import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function workspaceSummaryPlugin(pi: ExtensionAPI) {
  pi.registerTool({
    name: "workspace_summary",
    label: "Workspace Summary",
    description: "Return a read-only summary of the current Git worktree status.",
    parameters: Type.Object({}),
    async execute() {
      const result = await pi.exec("git", ["status", "--short"]);
      return {
        content: [{ type: "text", text: result.stdout || "Working tree is clean" }],
        details: { exitCode: result.code },
      };
    },
  });
}
