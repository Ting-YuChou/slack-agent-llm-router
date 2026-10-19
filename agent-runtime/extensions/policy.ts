import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { execFile } from "node:child_process";
import path from "node:path";

import { filesOutsideScope } from "../dist/src/candidate-scope.js";
import type { WriterContext } from "../dist/src/container-runtime.js";
import { classifyBash, isApprovedSkillRead, isTestConfigPath, parsePorcelainZ, validateWorkspacePath } from "../dist/src/policy.js";
import { codemodeMcpPrompt, type McpServerId } from "../dist/src/mcp-config.js";

const MAX_TURNS = Number(process.env.PI_AGENT_MAX_TURNS ?? "20");
const MAX_TOOL_CALLS = Number(process.env.PI_AGENT_MAX_TOOL_CALLS ?? "40");
const APPROVAL_TIMEOUT_MS = 300_000;
// Anything but an explicit "none" keeps per-write prompts, so a typo never silently drops approval.
const WRITE_APPROVAL = process.env.PI_AGENT_WRITE_APPROVAL === "none" ? "none" : "per_prompt";

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
  return [
    `Only modify files under these paths: ${context.allowedPaths.join(", ")}. Writes elsewhere are blocked, and out-of-scope changes fail delivery.`,
    `After you finish, trusted checks run in a separate verifier: ${context.checks.join(", ") || "(none)"}. Make them pass without weakening tests.`,
    `An independent reviewer grades these criteria: ${context.rubric.join(", ") || "(none)"}.`,
    `Delivery limits: at most ${limits.maxFiles} files changed, ${Math.floor(limits.maxBytes / 1024)} KiB total diff, ${Math.floor(limits.maxSingleFileBytes / 1024)} KiB per file. Prefer targeted edit calls over rewriting whole files.`,
  ].join(" ");
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

type BudgetStage = "none" | "notice" | "final";

/** Next stage reached: a notice at 75% of the budget, a final warning once two or fewer remain. */
function budgetStage(used: number, max: number): BudgetStage {
  if (max - used <= 2) return "final";
  return used >= Math.ceil(0.75 * max) ? "notice" : "none";
}

function budgetMessage(stage: BudgetStage, used: number, max: number, unit: string): string {
  if (stage === "final") {
    return `[harness] Final budget warning: ${used}/${max} ${unit} used, ${Math.max(0, max - used)} left. Stop exploring now: run the tests if you have not, then summarize what changed, what passed and what is left.`;
  }
  return `[harness] Budget notice: ${used}/${max} ${unit} used. Start to wrap up: finish the current change, run the tests and summarize what you did and what is left.`;
}

export default function policyExtension(pi: ExtensionAPI) {
  let turns = 0;
  let toolCalls = 0;
  let writeApproved = false;
  const writtenPaths = new Set<string>();
  const budgetSent: Record<"turns" | "toolCalls", BudgetStage> = { turns: "none", toolCalls: "none" };

  if (process.env.PI_AGENT_JEV_CLASSIFIER_MODE === "on") {
    pi.on("before_agent_start", (event) => {
      event.systemPromptOptions.sections.jev_classifier = [
        "For semantic classification, ranking, or yes/no probability that affects only your analysis, you may use codemode models.classify with openrouter/typesafe/jev-1.13.",
        "Prefer deterministic code for exact rules, and do not use classifier output as authorization for writes, shell commands, approvals, deployment, or deletion.",
        "Check stopReason and use the returned probabilities; continue without Jev if the classifier is unavailable.",
      ].join(" ");
    });
  }
  if (process.env.PI_AGENT_MCP_SERVERS_JSON) {
    pi.on("before_agent_start", (event) => {
      try {
        const servers = JSON.parse(process.env.PI_AGENT_MCP_SERVERS_JSON ?? "[]") as Array<{ server?: McpServerId }>;
        event.systemPromptOptions.sections.mcp_codemode = codemodeMcpPrompt(servers.flatMap((entry) => entry.server ? [entry.server] : []));
      } catch { throw new Error("PI_AGENT_MCP_SERVERS_JSON must be valid JSON"); }
    });
  }

  const scope = writerContext();
  if (scope) {
    pi.on("before_agent_start", (event) => {
      event.systemPromptOptions.sections.scope = scopePrompt(scope);
    });
  }

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
    toolCalls += 1;
    if (toolCalls > MAX_TOOL_CALLS) {
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
      if (!checked.allowed) return { block: true, reason: `Path blocked by policy: ${checked.reason}` };
    }

    if (event.toolName === "write" || event.toolName === "edit") {
      const target = path.posix.normalize(String((event.input as Record<string, unknown>).path ?? ""));
      if (scope && filesOutsideScope([target], scope.allowedPaths).length) {
        return { block: true, reason: `outside_allowed_paths: ${target}; this task may only modify ${scope.allowedPaths.join(", ")}` };
      }
      if (event.toolName === "write" && typeof event.input.content === "string" && Buffer.byteLength(event.input.content) > 256 * 1024) {
        return { block: true, reason: "Single write exceeds 256 KiB" };
      }
      if (!writeApproved && WRITE_APPROVAL === "per_prompt") {
        const approved = await ctx.ui.confirm(
          `Allow workspace ${event.toolName}?`,
          previewWrite(event.toolName, event.input),
          { timeout: APPROVAL_TIMEOUT_MS },
        );
        if (!approved) return { block: true, reason: "Workspace modification rejected or expired" };
        writeApproved = true;
      }
      writtenPaths.add(target);
      return undefined;
    }

    if (event.toolName === "bash") {
      const command = typeof event.input.command === "string" ? event.input.command : "";
      event.input.timeout = Math.min(
        typeof event.input.timeout === "number" ? event.input.timeout : 300_000,
        300_000,
      );
      const decision = classifyBash(command, safeCommands());
      if (decision.decision === "block") return { block: true, reason: `Command permanently blocked: ${decision.reason}` };
      if (decision.decision === "allow") return undefined;
      const taint = decision.decision === "allow_after_write_approval" ? await testConfigTaint(ctx.cwd, writtenPaths) : undefined;
      if (decision.decision === "allow_after_write_approval" && (writeApproved || WRITE_APPROVAL === "none") && !taint) return undefined;
      const approved = await ctx.ui.confirm(
        "Allow shell command?",
        `${taint ? `Needs approval because ${taint}.\n\n` : ""}${command.slice(0, 4_000)}`,
        { timeout: APPROVAL_TIMEOUT_MS },
      );
      if (!approved) return { block: true, reason: "Shell command rejected or expired" };
    }
    return undefined;
  });

  pi.on("input", () => {
    budgetSent.turns = "none";
    budgetSent.toolCalls = "none";
    return undefined;
  });

  // Limits abort without warning, so tell the agent while it can still verify and summarize its work.
  pi.on("tool_result", (event) => {
    // Nested results go back to the calling script, not to the model.
    if (event.parentToolCallId) return undefined;
    const notices: string[] = [];
    for (const [key, used, max, unit] of [["toolCalls", toolCalls, MAX_TOOL_CALLS, "tool calls"], ["turns", turns, MAX_TURNS, "turns"]] as const) {
      const stage = budgetStage(used, max);
      if (stage === "none" || stage === budgetSent[key] || budgetSent[key] === "final") continue;
      budgetSent[key] = stage;
      notices.push(budgetMessage(stage, used, max, unit));
    }
    if (!notices.length) return undefined;
    return {
      content: [...event.content, { type: "text" as const, text: notices.join("\n") }],
      ...(event.structuredContent !== undefined ? { structuredContent: event.structuredContent } : {}),
    };
  });
}
