/**
 * A docker CLI stand-in: `run` speaks enough Pi RPC to configure, ask for credentials and settle each
 * prompt; every other subcommand is logged, and fails when a `fail-<subcommand>` file exists.
 */
export const FAKE_DOCKER = `#!/usr/bin/env node
const fs = require("fs"), path = require("path");
const dir = __dirname, log = (line) => fs.appendFileSync(path.join(dir, "log"), line + "\\n");
const [cmd] = process.argv.slice(2);
if (cmd === "run") {
  log("run"); fs.writeFileSync(path.join(dir, "run.pid"), String(process.pid));
  fs.appendFileSync(path.join(dir, "run-args"), process.argv.slice(3).join(" ") + "\\n");
  let model = {}, level = "", buf = "";
  const out = (record) => process.stdout.write(JSON.stringify(record) + "\\n");
  process.stdin.on("data", (chunk) => {
    buf += chunk; let i;
    while ((i = buf.indexOf("\\n")) >= 0) {
      const r = JSON.parse(buf.slice(0, i)); buf = buf.slice(i + 1);
      if (r.type === "set_model") { model = { provider: r.provider, id: r.modelId }; out({ id: r.id, type: "response", command: r.type, success: true }); }
      else if (r.type === "set_thinking_level") { level = r.level; out({ id: r.id, type: "response", command: r.type, success: true }); }
      else if (r.type === "get_state") out({ id: r.id, type: "response", command: r.type, success: true, data: { model, thinkingLevel: level } });
      else if (r.type === "prompt") { out({ id: r.id, type: "response", command: "prompt", success: true }); out({ type: "extension_ui_request", id: "cred-" + r.id, method: "input", title: "pi-agent/credential", placeholder: "" }); }
      else if (r.type === "extension_ui_response") {
        log("credential " + (r.value ? JSON.parse(r.value).OPENAI_API_KEY : "cancelled"));
        out({ type: "message_end", message: { role: "assistant", content: [{ type: "text", text: "done" }], stopReason: "stop" } });
        out({ type: "agent_settled" });
      }
    }
  });
  process.stdin.on("end", () => process.exit(0));
  process.on("SIGTERM", () => process.exit(0));
} else if (cmd === "inspect") process.stdout.write("true\\n");
else {
  log(cmd);
  if (fs.existsSync(path.join(dir, "fail-" + cmd))) process.exit(1);
  if (cmd === "stop") { try { process.kill(Number(fs.readFileSync(path.join(dir, "run.pid"), "utf8")), "SIGTERM"); } catch {} }
}
`;
