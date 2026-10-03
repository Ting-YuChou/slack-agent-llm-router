import { cpSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const runtimeRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const lockPath = path.join(runtimeRoot, "package-lock.json");
const lock = JSON.parse(readFileSync(lockPath, "utf8"));
const piPackagePath = path.join(runtimeRoot, "node_modules", "@earendil-works", "pi-coding-agent", "package.json");
const piPackage = JSON.parse(readFileSync(piPackagePath, "utf8"));
const piLockEntry = lock.packages?.["node_modules/@earendil-works/pi-coding-agent"];
if (!piLockEntry) throw new Error("Pi package is missing from package-lock.json");

const patches = [
  {
    name: "brace-expansion",
    version: "5.0.12",
    resolved: "https://registry.npmjs.org/brace-expansion/-/brace-expansion-5.0.12.tgz",
    integrity: "sha512-YovQ3rzhaLMIrDjNDMkNS01tea93qhEhG5xy8f6+R0l+dw3Ki+5sCoIoI942iuLZTHWogWktgwVDhU09iNEimQ==",
  },
  {
    name: "undici",
    version: "8.11.2",
    resolved: "https://registry.npmjs.org/undici/-/undici-8.11.2.tgz",
    integrity: "sha512-u4UB2/IrKdU6lFxumHmmo1a3fCQO5tzQllRorfoRS63txhrB7xTpSn1PftwC4qEHkOaqP95fCWW4lJzwErwzhQ==",
  },
];

for (const patch of patches) {
  const source = path.join(runtimeRoot, "node_modules", patch.name);
  const target = path.join(
    runtimeRoot,
    "node_modules",
    "@earendil-works",
    "pi-coding-agent",
    "node_modules",
    patch.name,
  );
  const sourcePackage = JSON.parse(readFileSync(path.join(source, "package.json"), "utf8"));
  if (sourcePackage.version !== patch.version) {
    throw new Error(`Expected ${patch.name} ${patch.version}, found ${sourcePackage.version}`);
  }
  rmSync(target, { recursive: true, force: true });
  cpSync(source, target, { recursive: true, force: false });
  const installedPackage = JSON.parse(readFileSync(path.join(target, "package.json"), "utf8"));
  if (installedPackage.version !== patch.version) {
    throw new Error(`Failed to patch Pi's vendored ${patch.name} dependency`);
  }
  const lockKey = `node_modules/@earendil-works/pi-coding-agent/node_modules/${patch.name}`;
  const lockEntry = lock.packages?.[lockKey];
  if (lockEntry) {
    Object.assign(lockEntry, {
      version: patch.version,
      resolved: patch.resolved,
      integrity: patch.integrity,
    });
  }
  if (typeof piPackage.dependencies?.[patch.name] === "string") {
    piPackage.dependencies[patch.name] = patch.version;
    piLockEntry.dependencies[patch.name] = patch.version;
  }
}

writeFileSync(piPackagePath, `${JSON.stringify(piPackage, null, 2)}\n`, { mode: 0o644 });
writeFileSync(lockPath, `${JSON.stringify(lock, null, 2)}\n`, { mode: 0o600 });
process.stdout.write(`Patched Pi vendored dependencies: ${patches.map((patch) => `${patch.name}@${patch.version}`).join(", ")}\n`);
