import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { validateSnapshotRead } from "../dist/src/policy.js";
import { renderPrompt } from "../dist/src/prompts.js";

/** Defense in depth: host mounts the workspace read-only and exposes only these tools. */
export default function discussion(pi: ExtensionAPI) {
  const root = process.env.PI_AGENT_WORKTREE;
  if (!root) throw new Error("Discussion workspace required");
  let calls = 0, turns = 0;
  pi.on("input", () => { calls = 0; turns = 0; });
  pi.on("before_agent_start", event => {
    event.systemPromptOptions.sections.discussion = renderPrompt("discussion", { root });
  });
  pi.on("turn_start", (_event, ctx) => { if (++turns > 20) ctx.abort(); });
  pi.on("tool_call", async (event, ctx) => {
    if (++calls > 40) { ctx.abort(); return { block: true, reason: "Discussion tool limit" }; }
    if (!["read", "grep", "find", "ls"].includes(event.toolName)) return { block: true, reason: "Discussion is read-only" };
    const input = event.input as Record<string, unknown>;
    if (input.path === undefined) input.path = root;
    if (!(await validateSnapshotRead(root, ctx.cwd, typeof input.path === "string" ? input.path : root, ["."])).allowed) return { block: true, reason: "Read outside discussion workspace" };
  });
}
