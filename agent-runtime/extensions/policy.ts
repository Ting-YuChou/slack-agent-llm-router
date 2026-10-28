import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { execFile } from "node:child_process";
import path from "node:path";
import { Type } from "typebox";

import { filesOutsideScope } from "../dist/src/candidate-scope.js";
import { DEFAULT_DIFF_LIMITS, DIFF_SUMMARY_TOOL, checkWrite, diffSummary, measureDiff, writeCap, type DiffUsage } from "../dist/src/diff-budget.js";
import type { WriterContext } from "../dist/src/container-runtime.js";
import { classifyBash, isApprovedSkillRead, isTestConfigPath, parsePorcelainZ, validateWorkspacePath } from "../dist/src/policy.js";
import { MCP_SERVERS, allServerTools, codemodeMcpPrompt, namespacedMcpToolNames, type McpServerId } from "../dist/src/mcp-config.js";
import { renderPrompt } from "../dist/src/prompts.js";
import { BUDGET_STATUS_DIALOG, activeTimeNotice, usdBudgetNotice } from "../dist/src/budget-notice.js";
import { POLICY_ENTRY, policyDecisionEntry, type PolicyVerdict } from "../dist/src/policy-decision.js";

const APPROVAL_TIMEOUT_MS = 300_000;
const BUDGET_STATUS_TIMEOUT_MS = 5_000;

function safeCommands(): string[] {
  try {
    const parsed = JSON.parse(process.env.PI_AGENT_SAFE_COMMANDS_JSON ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === "string") : [];
  } catch {
    return [];
  }
}

function approvedSkillPaths(): string[] {
  try {
    const parsed = JSON.parse(process.env.PI_AGENT_SKILL_PATHS_JSON ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === "string") : [];
  } catch {
    return [];
  }
}

function writerContext(): WriterContext | undefined {
  if (!process.env.PI_AGENT_WRITER_JSON) return undefined;
  try {
    const parsed = JSON.parse(process.env.PI_AGENT_WRITER_JSON) as WriterContext;
    if (!Array.isArray(parsed.allowedPaths) || !parsed.allowedPaths.every((item) => typeof item === "string")) throw new Error("allowedPaths");
    return parsed;
  } catch { throw new Error("PI_AGENT_WRITER_JSON must describe allowedPaths, checks, rubric and diffLimits"); }
}

function scopePrompt(context: WriterContext): string {
  const limits = context.diffLimits;
  return renderPrompt("policy-scope", {
    allowed_paths: context.allowedPaths.join(", "), checks: context.checks.join(", ") || "(none)", rubric: context.rubric.join(", ") || "(none)",
    max_files: limits.maxFiles, max_kib: Math.floor(limits.maxBytes / 1024), max_file_kib: Math.floor(limits.maxSingleFileBytes / 1024),
  });
}

function previewWrite(tool: string, input: Record<string, unknown>): string {
  const filePath = typeof input.path === "string" ? input.path : "(unknown path)";
  if (tool === "write") {
    const content = typeof input.content === "string" ? input.content : "";
    return `${filePath}\n\n${content.slice(0, 3_500)}${content.length > 3_500 ? "\n… truncated" : ""}`;
  }
  const edits = Array.isArray(input.edits) ? input.edits : [];
  const summary = edits.slice(0, 10).map((edit) => {
    if (!edit || typeof edit !== "object") return "";
    const value = edit as Record<string, unknown>;
    return `- ${String(value.oldText ?? "").slice(0, 500)}\n+ ${String(value.newText ?? "").slice(0, 500)}`;
  }).join("\n");
  return `${filePath}\n\n${summary.slice(0, 3_500)}`;
}

/** Changed files in the worktree; undefined when git cannot say, which callers treat as tainted. */
function changedFiles(cwd: string): Promise<string[] | undefined> {
  return new Promise((resolve) => {
    execFile(
      "git",
      ["status", "--porcelain", "-z", "--untracked-files=all"],
      // The git metadata is mounted read-only, so status must not try to refresh the index.
      { cwd, env: { ...process.env, GIT_OPTIONAL_LOCKS: "0" }, maxBuffer: 8 * 1024 * 1024, timeout: 15_000 },
      (error, stdout) => resolve(error ? undefined : parsePorcelainZ(stdout)),
    );
  });
}

/** Names the reason a configured command needs approval because test configuration may have changed. */
async function testConfigTaint(cwd: string, writtenPaths: Set<string>): Promise<string | undefined> {
  const written = [...writtenPaths].find(isTestConfigPath);
  if (written) return `test configuration changed: ${written}`;
  const changed = await changedFiles(cwd);
  if (!changed) return "could not list changed files";
  const tainted = changed.find(isTestConfigPath);
  return tainted ? `test configuration changed: ${tainted}` : undefined;
}

/** The run's MCP servers, from the registrations the host passes in. */
function runMcpServers(value: string | undefined): McpServerId[] {
  if (!value) return [];
  let entries: unknown;
  try { entries = JSON.parse(value); } catch { entries = undefined; }
  if (!Array.isArray(entries)) throw new Error("PI_AGENT_MCP_SERVERS_JSON must be valid JSON");
  return entries.flatMap((entry) => typeof entry?.server === "string" && Object.hasOwn(MCP_SERVERS, entry.server) ? [entry.server as McpServerId] : []);
}

type BudgetStage = "none" | "notice" | "final";

/** Next stage reached: a notice at 75% of the budget, a final warning once two or fewer remain. */
function budgetStage(used: number, max: number): BudgetStage {
  if (max - used <= 2) return "final";
  return used >= Math.ceil(0.75 * max) ? "notice" : "none";
}

function budgetMessage(stage: BudgetStage, used: number, max: number, unit: string): string {
  const left = Math.max(0, max - used);
  return stage === "final"
    ? renderPrompt("budget-final", { used, max, unit, left })
    : renderPrompt("budget-notice", { used, max, unit, left });
}

export default function policyExtension(pi: ExtensionAPI) {
  // Read when Pi activates the extension, so one module instance can serve differently configured sessions.
  const MAX_TURNS = Number(process.env.PI_AGENT_MAX_TURNS ?? "20");
  const MAX_TOOL_CALLS = Number(process.env.PI_AGENT_MAX_TOOL_CALLS ?? "40");
  // Delivery-mode runs pass "none" (the owner approves the final diff instead). CI repair sessions skip that
  // card and keep per-write prompts. Anything but an explicit "none" prompts, so a typo never silently drops approval.
  const WRITE_APPROVAL = process.env.PI_AGENT_WRITE_APPROVAL === "none" ? "none" : "per_prompt";
  let turns = 0;
  let toolCalls = 0;
  const mcpServers = runMcpServers(process.env.PI_AGENT_MCP_SERVERS_JSON);
  const mcpToolServer = new Map(mcpServers.flatMap((server) => namespacedMcpToolNames(server, allServerTools(server)).map((name) => [name, server] as const)));
  /** Nested MCP calls per server. A new input does not reset it: the CodeGraph proxy's budget lasts as long as the container. */
  const nestedMcpCalls = new Map<McpServerId, number>();
  let writeApproved = false;
  const writtenPaths = new Set<string>();
  const budgetSent: Record<"turns" | "toolCalls" | "usd" | "time", BudgetStage> = { turns: "none", toolCalls: "none", usd: "none", time: "none" };
  /** The turn whose one budget_status dialog was already asked; the host would cancel a second. */
  let statusAskedTurn = -1;
  // The host counts these to tune the classifier; commands go out only as a basename and a hash.
  const decide = (tool: string, decision: PolicyVerdict, reason: string, extra: { approved?: boolean; command?: string; startedAt?: number } = {}) =>
    pi.appendEntry(POLICY_ENTRY, policyDecisionEntry({ tool, decision, reason, approved: extra.approved, command: extra.command, latencyMs: extra.startedAt === undefined ? undefined : Date.now() - extra.startedAt }));

  if (process.env.PI_AGENT_JEV_CLASSIFIER_MODE === "on") {
    pi.on("before_agent_start", (event) => {
      event.systemPromptOptions.sections.jev_classifier = renderPrompt("jev-classifier");
    });
  }
  if (process.env.PI_AGENT_MCP_SERVERS_JSON) {
    pi.on("before_agent_start", (event) => {
      event.systemPromptOptions.sections.mcp_codemode = codemodeMcpPrompt(mcpServers);
    });
  }

  const scope = writerContext();
  if (scope) {
    pi.on("before_agent_start", (event) => {
      event.systemPromptOptions.sections.scope = scopePrompt(scope);
    });
  }
  // The host checks the final diff against these same limits; checking each write lets the agent fix course.
  const diffLimits = scope?.diffLimits ?? DEFAULT_DIFF_LIMITS;
  const maxWriteBytes = writeCap(diffLimits);
  /** Bytes per path this session wrote; stands in for the worktree's diff only when git cannot measure it. */
  const writtenBytes: DiffUsage = new Map();

  pi.registerTool({
    name: DIFF_SUMMARY_TOOL,
    label: "Diff summary",
    description: "Show your whole change against the baseline: added and deleted lines per path (shell changes and new files included) and how many more files and bytes the diff limits allow. Read-only; use it before you finish.",
    parameters: Type.Object({}),
    async execute(_toolCallId, _input, _signal, _onUpdate, ctx) {
      const summary = await diffSummary(ctx.cwd, diffLimits);
      if (!summary) return { content: [{ type: "text", text: "diff_summary could not read the worktree's Git state." }], details: {}, isError: true };
      return { content: [{ type: "text", text: summary.text }], details: summary.details };
    },
  });

  pi.on("input", () => {
    turns = 0;
    toolCalls = 0;
    writeApproved = false;
    return undefined;
  });

  pi.on("turn_start", (_event, ctx) => {
    turns += 1;
    if (turns > MAX_TURNS) {
      // The host can only tell a harness limit from other aborts through this session entry.
      pi.appendEntry("pi_agent_limit", { kind: "turns", limit: MAX_TURNS });
      ctx.abort();
    }
  });

  pi.on("tool_call", async (event, ctx) => {
    // A codemode script's MCP calls are bounded by their server's own call budget (its gateway token or the
    // CodeGraph proxy), so they count toward the run's limit only once that budget is spent. The script itself
    // and its other nested calls still count.
    const server = event.parentToolCallId ? mcpToolServer.get(event.toolName) : undefined;
    const nested = server ? nestedMcpCalls.get(server) ?? 0 : 0;
    if (server && nested < MCP_SERVERS[server].maxCalls) nestedMcpCalls.set(server, nested + 1);
    else if (++toolCalls > MAX_TOOL_CALLS) {
      pi.appendEntry("pi_agent_limit", { kind: "tool_calls", limit: MAX_TOOL_CALLS });
      ctx.abort();
      return { block: true, reason: `Tool call limit of ${MAX_TOOL_CALLS} exceeded` };
    }

    if (["read", "write", "edit", "grep", "find", "ls", "lsp"].includes(event.toolName)) {
      const input = event.input as Record<string, unknown>;
      const requestedPath = typeof input.path === "string" ? input.path : ".";
      if (event.toolName === "read" && isApprovedSkillRead(requestedPath, approvedSkillPaths())) return undefined;
      const operation = event.toolName === "write" || event.toolName === "edit" ? "write" : "read";
      const checked = await validateWorkspacePath(ctx.cwd, requestedPath, operation);
      if (!checked.allowed) { decide(event.toolName, "block", checked.reason ?? "path_blocked"); return { block: true, reason: `Path blocked by policy: ${checked.reason}` }; }
    }

    if (event.toolName === "write" || event.toolName === "edit") {
      const target = path.posix.normalize(String((event.input as Record<string, unknown>).path ?? ""));
      if (scope && filesOutsideScope([target], scope.allowedPaths).length) {
        decide(event.toolName, "block", "outside_allowed_paths");
        return { block: true, reason: `outside_allowed_paths: ${target}; this task may only modify ${scope.allowedPaths.join(", ")}` };
      }
      const edits = event.toolName === "edit" && Array.isArray(event.input.edits) ? event.input.edits : [];
      const written = event.toolName === "write" ? (typeof event.input.content === "string" ? Buffer.byteLength(event.input.content) : 0)
        : edits.reduce((total: number, edit: unknown) => total + (edit && typeof edit === "object" && typeof (edit as { newText?: unknown }).newText === "string" ? Buffer.byteLength((edit as { newText: string }).newText) : 0), 0);
      if (written > maxWriteBytes) {
        decide(event.toolName, "block", "write_too_large");
        return { block: true, reason: `write_too_large: a single ${event.toolName} may carry at most ${maxWriteBytes} bytes; this one has ${written}` };
      }
      const usage = await measureDiff(ctx.cwd) ?? writtenBytes;
      const projected = event.toolName === "write" ? written : (usage.get(target) ?? 0) + written;
      const budget = checkWrite(usage, diffLimits, target, projected);
      if (!budget.allowed) {
        decide(event.toolName, "block", budget.reason);
        return { block: true, reason: budget.message };
      }
      if (!writeApproved && WRITE_APPROVAL === "per_prompt") {
        const startedAt = Date.now();
        const approved = await ctx.ui.confirm(
          `Allow workspace ${event.toolName}: ${target}?`,
          previewWrite(event.toolName, event.input),
          { timeout: APPROVAL_TIMEOUT_MS },
        );
        decide(event.toolName, "approval", "write_approval", { approved, startedAt });
        if (!approved) return { block: true, reason: "Workspace modification rejected or expired" };
        writeApproved = true;
      } else decide(event.toolName, "allow", WRITE_APPROVAL === "none" ? "write_approval_off" : "write_approved_for_prompt");
      writtenPaths.add(target);
      writtenBytes.set(target, projected);
      return undefined;
    }

    if (event.toolName === "bash") {
      const command = typeof event.input.command === "string" ? event.input.command : "";
      event.input.timeout = Math.min(
        typeof event.input.timeout === "number" ? event.input.timeout : 300_000,
        300_000,
      );
      const decision = classifyBash(command, safeCommands());
      if (decision.decision === "block") { decide("bash", "block", decision.reason, { command }); return { block: true, reason: `Command permanently blocked: ${decision.reason}` }; }
      if (decision.decision === "allow") { decide("bash", "allow", decision.reason, { command }); return undefined; }
      const taint = decision.decision === "allow_after_write_approval" ? await testConfigTaint(ctx.cwd, writtenPaths) : undefined;
      if (decision.decision === "allow_after_write_approval" && (writeApproved || WRITE_APPROVAL === "none") && !taint) { decide("bash", "allow", decision.reason, { command }); return undefined; }
      const startedAt = Date.now();
      // The title names the command on one line; the dialog body still shows all of it.
      const line = command.replace(/\s+/g, " ").trim();
      const approved = await ctx.ui.confirm(
        `Allow shell command: ${line.length > 120 ? `${line.slice(0, 120)}…` : line}?`,
        `${taint ? `Needs approval because ${taint}.\n\n` : ""}${command.slice(0, 4_000)}`,
        { timeout: APPROVAL_TIMEOUT_MS },
      );
      decide("bash", "approval", taint ? "test_config_changed" : decision.reason, { approved, command, startedAt });
      if (!approved) return { block: true, reason: "Shell command rejected or expired" };
    }
    return undefined;
  });

  pi.on("input", () => {
    budgetSent.turns = "none";
    budgetSent.toolCalls = "none";
    budgetSent.usd = "none";
    budgetSent.time = "none";
    statusAskedTurn = -1;
    return undefined;
  });

  // Limits abort without warning, so tell the agent while it can still verify and summarize its work.
  pi.on("tool_result", async (event, ctx) => {
    // Nested results go back to the calling script, not to the model.
    if (event.parentToolCallId) return undefined;
    const notices: string[] = [];
    for (const [key, used, max, unit] of [["toolCalls", toolCalls, MAX_TOOL_CALLS, "tool calls"], ["turns", turns, MAX_TURNS, "turns"]] as const) {
      const stage = budgetStage(used, max);
      if (stage === "none" || stage === budgetSent[key] || budgetSent[key] === "final") continue;
      budgetSent[key] = stage;
      notices.push(budgetMessage(stage, used, max, unit));
    }
    // One reply carries the dollars and, once it runs short, the active time. It is not asked after the final dollar warning.
    if (budgetSent.usd !== "final" && statusAskedTurn !== turns && typeof ctx.ui?.input === "function") {
      statusAskedTurn = turns;
      const reply = await ctx.ui.input(BUDGET_STATUS_DIALOG, "", { timeout: BUDGET_STATUS_TIMEOUT_MS });
      for (const [key, notice] of [["usd", usdBudgetNotice(reply)], ["time", activeTimeNotice(reply)]] as const) {
        if (!notice || notice.stage === budgetSent[key] || budgetSent[key] === "final") continue;
        budgetSent[key] = notice.stage;
        notices.push(notice.text);
      }
    }
    if (!notices.length) return undefined;
    return {
      content: [...event.content, { type: "text" as const, text: notices.join("\n") }],
      ...(event.structuredContent !== undefined ? { structuredContent: event.structuredContent } : {}),
    };
  });
}
