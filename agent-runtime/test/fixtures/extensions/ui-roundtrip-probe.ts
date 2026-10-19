// Test-only Pi extension: echoes a large extension-UI input round trip back as a notification.
import { createHash } from "node:crypto";
export default function (pi: any) {
  pi.registerCommand("probe", { description: "extension UI round-trip probe", handler: async (_args: string, ctx: any) => {
    const value: string | undefined = await ctx.ui.input("pi-agent/run_checks", "P".repeat(16 * 1024));
    ctx.ui.notify(`len=${value?.length ?? -1};sha=${createHash("sha256").update(value ?? "").digest("hex")}`, "info");
  } });
}
