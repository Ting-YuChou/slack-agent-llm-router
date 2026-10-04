import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const runtimeRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const lockPath = path.join(runtimeRoot, "plugins.lock.json");
const skillLockPath = path.join(runtimeRoot, "skills.lock.json");
const image = process.argv[2] ?? "slack-pi-agent:1.0.1";
const lock = JSON.parse(readFileSync(lockPath, "utf8"));
const skillLock = JSON.parse(readFileSync(skillLockPath, "utf8"));

for (const plugin of lock.plugins ?? []) {
  const pluginPath = path.resolve(runtimeRoot, plugin.host_path);
  const digest = `sha256:${createHash("sha256").update(readFileSync(pluginPath)).digest("hex")}`;
  if (digest !== plugin.integrity) {
    throw new Error(`Refusing to lock an image while plugin checksum is invalid: ${plugin.id}`);
  }
}

for (const skill of skillLock.skills ?? []) {
  const skillPath = path.resolve(runtimeRoot, skill.host_path);
  const digest = `sha256:${createHash("sha256").update(readFileSync(skillPath)).digest("hex")}`;
  if (digest !== skill.integrity) {
    throw new Error(`Refusing to lock an image while skill checksum is invalid: ${skill.name}`);
  }
}

const imageDigest = execFileSync(
  "docker",
  ["image", "inspect", "--format={{.Id}}", image],
  { encoding: "utf8" },
).trim();
if (!/^sha256:[a-f0-9]{64}$/.test(imageDigest)) {
  throw new Error(`Docker returned an invalid image digest for ${image}`);
}

lock.image_digest = imageDigest;
const temporaryPath = `${lockPath}.tmp`;
writeFileSync(temporaryPath, `${JSON.stringify(lock, null, 2)}\n`, { mode: 0o600 });
renameSync(temporaryPath, lockPath);
process.stdout.write(`Locked ${image} at ${imageDigest}\n`);
