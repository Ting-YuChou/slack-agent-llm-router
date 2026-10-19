import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { renderPrompt } from "../dist/src/prompts.js";

/** Must match RUN_CHECKS_TOOL / RUN_CHECKS_DIALOG in src/pi-rpc.ts; the host answers only this dialog. */
export const RUN_CHECKS_TOOL = "run_checks";
export const RUN_CHECKS_DIALOG = "pi-agent/run_checks";

interface ChecksContext { checks: string[]; defaultChecks: string[]; maxCalls: number }

function checksContext(): ChecksContext {
  try {
    const parsed = JSON.parse(process.env.PI_AGENT_CHECKS_JSON ?? "null") as ChecksContext;
    if (!Array.isArray(parsed?.checks) || !parsed.checks.length || !Array.isArray(parsed.defaultChecks) || !Number.isInteger(parsed.maxCalls)) throw new Error();
    return parsed;
  } catch { throw new Error("PI_AGENT_CHECKS_JSON must describe checks, defaultChecks and maxCalls"); }
}

export function checksPrompt(context: ChecksContext): string {
  return renderPrompt("checks", { tool: RUN_CHECKS_TOOL, checks: context.checks.join(", "), default_checks: context.defaultChecks.join(", "), max_calls: context.maxCalls });
}

export default function checks(pi: ExtensionAPI) {
  const context = checksContext();
  pi.registerTool({
    name: RUN_CHECKS_TOOL,
    label: "Run checks",
    description: `Run trusted verification checks (${context.checks.join(", ")}) on the current changes and get the failures back. At most ${context.maxCalls} calls per task.`,
    parameters: Type.Object({ checks: Type.Optional(Type.Array(Type.String({ maxLength: 80 }), { maxItems: 20, description: "Check names to run; omit for the default set" })) }),
    async execute(toolCallId, input, signal, _onUpdate, ctx) {
      const request = JSON.stringify({ tool_call_id: toolCallId, ...(input.checks?.length ? { checks: input.checks } : {}) });
      // The host runs the checks and answers the dialog; aborting the tool call dismisses it.
      const value = await ctx.ui.input(RUN_CHECKS_DIALOG, request, { signal });
      if (value === undefined) {
        const text = signal?.aborted ? "run_checks was aborted." : "run_checks is unavailable for this request.";
        return { content: [{ type: "text", text }], details: { status: "rejected", reason: "cancelled" }, isError: true };
      }
      let details: { status?: string } = {};
      try { details = JSON.parse(value) as { status?: string }; } catch { /* keep the raw text */ }
      return { content: [{ type: "text", text: value }], details, isError: details.status === "rejected" };
    },
  });
  pi.on("before_agent_start", (event) => {
    event.systemPromptOptions.sections.run_checks = checksPrompt(context);
  });
}
