import { cpSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const runtimeRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(runtimeRoot, "node_modules", "brace-expansion");
const target = path.join(
  runtimeRoot,
  "node_modules",
  "@earendil-works",
  "pi-coding-agent",
  "node_modules",
  "brace-expansion",
);
const sourcePackage = JSON.parse(readFileSync(path.join(source, "package.json"), "utf8"));
if (sourcePackage.version !== "5.0.8") {
  throw new Error(`Expected brace-expansion 5.0.8, found ${sourcePackage.version}`);
}

rmSync(target, { recursive: true, force: true });
cpSync(source, target, { recursive: true, force: false });
const installedPackage = JSON.parse(readFileSync(path.join(target, "package.json"), "utf8"));
if (installedPackage.version !== "5.0.8") {
  throw new Error("Failed to patch Pi's vendored brace-expansion dependency");
}

const lockPath = path.join(runtimeRoot, "package-lock.json");
const lock = JSON.parse(readFileSync(lockPath, "utf8"));
const lockKey = "node_modules/@earendil-works/pi-coding-agent/node_modules/brace-expansion";
const lockEntry = lock.packages?.[lockKey];
if (!lockEntry) throw new Error("Pi vendored brace-expansion is missing from package-lock.json");
Object.assign(lockEntry, {
  version: "5.0.8",
  resolved: "https://registry.npmjs.org/brace-expansion/-/brace-expansion-5.0.8.tgz",
  integrity: "sha512-JZyDyq3D4AUifKTPOB7DELf6XsB3WdPuNxCtob1vFXPsSXhdAiHBWJ/tJ8HAc9aH84BK+5JFZLNkJKx3G9kzQg==",
});
writeFileSync(lockPath, `${JSON.stringify(lock, null, 2)}\n`, { mode: 0o600 });
process.stdout.write("Patched Pi vendored brace-expansion to 5.0.8\n");
