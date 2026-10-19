import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

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
  return [
    `Use the ${RUN_CHECKS_TOOL} tool to run the trusted verification checks on your current changes before you finish: ${context.checks.join(", ")}.`,
    `With no arguments it runs ${context.defaultChecks.join(", ")}; pass checks to choose a subset. You have at most ${context.maxCalls} calls per task, and one call at a time.`,
    "Results list failing files and lines; fix them and run the checks again. Run tests through this tool rather than through bash.",
    "The same checks run again after you finish, so do not weaken or skip tests to make them pass.",
  ].join(" ");
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
