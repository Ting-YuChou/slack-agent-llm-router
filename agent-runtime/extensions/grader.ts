import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import path from "node:path";
import { validateSnapshotRead } from "../dist/src/policy.js";
import { Type } from "typebox";
import { validateGrade } from "../dist/src/quality-gate.js";
import { verifyGradeEvidence } from "../dist/src/grade-evidence.js";
export default function grader(pi: ExtensionAPI) {
  const config = JSON.parse(process.env.PI_AGENT_GRADER_JSON ?? "null");
  if (!config) throw new Error("Grader context required");
  const root = process.env.PI_AGENT_WORKTREE!;
  let submitted = false;
  let calls = 0;
  pi.on("tool_call", async (event, ctx) => {
    // The verdict itself never counts against the read budget.
    if (event.toolName === "submit_grade" && !submitted) return;
    if (++calls > 40) { ctx.abort(); return { block: true, reason: "Grader tool limit" }; }

    if (!["read", "grep", "find", "ls", "submit_grade"].includes(event.toolName)) return { block: true, reason: "Grader is read-only" };
    if (submitted) return { block: true, reason: "Grade already submitted" };
    if (event.toolName !== "submit_grade") {
      const input = event.input as Record<string, unknown>;
      // The session cwd is outside the snapshot: relative and path-less reads target the snapshot root.
      const resolved = input.path = path.resolve(root, typeof input.path === "string" ? input.path : ".");
      if (!(await validateSnapshotRead(root,root,resolved)).allowed) return { block: true, reason: "Only candidate snapshot is readable" };
    }
  });
  pi.registerTool({
    name: "submit_grade", label: "Submit grade", description: "Submit the final independent verdict bound to this candidate. Evaluate every rubric criterion. Cite evidence as path or path:line inside the snapshot.",
    parameters: Type.Object({
      status: Type.Union([Type.Literal("passed"), Type.Literal("needs_changes"), Type.Literal("unknown")]),
      candidate_hash: Type.String(),
      criteria: Type.Array(Type.Object({ id: Type.String(), passed: Type.Boolean(), evidence: Type.Optional(Type.String()) })),
      findings: Type.Array(Type.Object({ path: Type.String(), line: Type.Integer(), severity: Type.Union([Type.Literal("error"), Type.Literal("warning")]), reason: Type.String() })),
    }),
    async execute(_id, input) {
      const result = validateGrade(input, config.candidate_hash, config.rubric);
      // Reject invented locations now, so the grader can correct them within this session.
      await verifyGradeEvidence(result, root, Array.isArray(config.deleted_paths) ? config.deleted_paths : []);
      submitted = true;
      return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
    },
  });
}
