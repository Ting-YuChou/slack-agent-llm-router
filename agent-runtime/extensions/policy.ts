import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { classifyBash, validateWorkspacePath } from "../dist/src/policy.js";

const MAX_TURNS = Number(process.env.PI_AGENT_MAX_TURNS ?? "20");
const MAX_TOOL_CALLS = Number(process.env.PI_AGENT_MAX_TOOL_CALLS ?? "40");
const APPROVAL_TIMEOUT_MS = 300_000;

function safeCommands(): string[] {
  try {
    const parsed = JSON.parse(process.env.PI_AGENT_SAFE_COMMANDS_JSON ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === "string") : [];
  } catch {
    return [];
  }
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

export default function policyExtension(pi: ExtensionAPI) {
  let turns = 0;
  let toolCalls = 0;
  let writeApproved = false;

  pi.on("input", () => {
    turns = 0;
    toolCalls = 0;
    writeApproved = false;
    return undefined;
  });

  pi.on("turn_start", (_event, ctx) => {
    turns += 1;
    if (turns > MAX_TURNS) ctx.abort();
  });

  pi.on("tool_call", async (event, ctx) => {
    toolCalls += 1;
    if (toolCalls > MAX_TOOL_CALLS) {
      ctx.abort();
      return { block: true, reason: `Tool call limit of ${MAX_TOOL_CALLS} exceeded` };
    }

    if (["read", "write", "edit", "grep", "find", "ls"].includes(event.toolName)) {
      const input = event.input as Record<string, unknown>;
      const requestedPath = typeof input.path === "string" ? input.path : ".";
      const operation = event.toolName === "write" || event.toolName === "edit" ? "write" : "read";
      const checked = await validateWorkspacePath(ctx.cwd, requestedPath, operation);
      if (!checked.allowed) return { block: true, reason: `Path blocked by policy: ${checked.reason}` };
    }

    if (event.toolName === "write" || event.toolName === "edit") {
      if (event.toolName === "write" && typeof event.input.content === "string" && Buffer.byteLength(event.input.content) > 256 * 1024) {
        return { block: true, reason: "Single write exceeds 256 KiB" };
      }
      if (!writeApproved) {
        const approved = await ctx.ui.confirm(
          `Allow workspace ${event.toolName}?`,
          previewWrite(event.toolName, event.input),
          { timeout: APPROVAL_TIMEOUT_MS },
        );
        if (!approved) return { block: true, reason: "Workspace modification rejected or expired" };
        writeApproved = true;
      }
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
      if (decision.decision === "allow_after_write_approval" && writeApproved) return undefined;
      const approved = await ctx.ui.confirm(
        "Allow shell command?",
        command.slice(0, 4_000),
        { timeout: APPROVAL_TIMEOUT_MS },
      );
      if (!approved) return { block: true, reason: "Shell command rejected or expired" };
    }
    return undefined;
  });
}
