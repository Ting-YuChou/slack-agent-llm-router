// Runs only inside the credential-free verification container.
import { chmodSync, cpSync, existsSync, mkdirSync, readdirSync, rmSync, statSync, symlinkSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { createConnection } from "node:net";
const check = JSON.parse(process.argv[2]);
const workspace = "/workspace/check";
mkdirSync(workspace, { mode: 0o700 });
cpSync("/candidate", workspace, { recursive: true, force: true, preserveTimestamps: false, mode: 0 });
function writable(directory: string) {
  chmodSync(directory, 0o700);
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) writable(file);
    else if (entry.isFile()) chmodSync(file, statSync(file).mode & 0o111 ? 0o700 : 0o600);
    else throw new Error("Unsupported candidate file");
  }
}
writable(workspace);
if (check.mode === "baseline") for (const name of check.baselinePaths) {
  const source = path.resolve("/baseline", name), target = path.resolve(workspace, name);
  if (!source.startsWith("/baseline/") || !target.startsWith(workspace + "/")) throw new Error("Invalid baseline path");
  rmSync(target, { recursive: true, force: true });
  if (existsSync(source)) cpSync(source, target, { recursive: true, force: true });
}
writable(workspace);
mkdirSync(path.join(workspace, "agent-runtime"), { recursive: true });
if (!existsSync(path.join(workspace, "agent-runtime/node_modules"))) symlinkSync("/opt/pi/node_modules", path.join(workspace, "agent-runtime/node_modules"));
const cwd = path.resolve(workspace, check.cwd);
if (cwd !== workspace && !cwd.startsWith(workspace + "/")) throw new Error("Invalid check cwd");
// Redis is task-local and cannot reach any host/shared Redis instance.
const redis = spawn("redis-server", ["--bind", "127.0.0.1", "--port", "6379", "--save", "", "--appendonly", "no", "--dir", "/tmp"], { stdio: "ignore" });
redis.on("error", () => process.exit(125));
let ready = false;
for (let attempt = 0; attempt < 50 && !ready; attempt++) {
  ready = await new Promise<boolean>(resolve => { const socket = createConnection({ host: "127.0.0.1", port: 6379 }); socket.once("connect", () => { socket.destroy(); resolve(true); }); socket.once("error", () => resolve(false)); socket.setTimeout(100, () => { socket.destroy(); resolve(false); }); });
  if (!ready) await new Promise(resolve => setTimeout(resolve, 50));
}
if (!ready) process.exit(125);
const child = spawn(check.argv[0], check.argv.slice(1), { cwd, stdio: "inherit", env: { PATH: process.env.PATH, HOME: "/tmp", TMPDIR: "/tmp", PYTHONDONTWRITEBYTECODE: "1", REDIS_HOST: "127.0.0.1", REDIS_PORT: "6379", REDIS_URL: "redis://127.0.0.1:6379/0" } });
child.on("error", () => process.exit(125));
child.on("exit", code => { redis.kill(); process.exit(code ?? 125); });
