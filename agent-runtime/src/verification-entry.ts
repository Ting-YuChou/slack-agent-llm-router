// Runs only inside the credential-free verification container.
import { spawn } from "node:child_process";
import { containerDeps, prepareWorkspace, runModeChecks, startServices } from "./verification-runner.js";
const check = JSON.parse(process.argv[2]);
if (check.runner === "mode") {
  // isolation "per_mode": every check of one mode in this container, one stdout record each.
  await runModeChecks(check, containerDeps);
  process.exit(0);
}
// isolation "per_check": this container runs one check and exits with its code.
const cwd = prepareWorkspace(check.mode, check.baselinePaths, check.cwd);
let env: Record<string, string>;
try { env = await startServices(1); } catch { process.exit(125); }
const child = spawn(check.argv[0], check.argv.slice(1), { cwd, stdio: "inherit", env: { PATH: process.env.PATH ?? "", PYTHONDONTWRITEBYTECODE: "1", PYTHONNOUSERSITE: "1", ...env } });
child.on("error", () => process.exit(125));
child.on("exit", code => process.exit(code ?? 125));
