import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

test("agent runtime exposes an HTTP server module", () => {
  const modulePath = fileURLToPath(new URL("../src/server.js", import.meta.url));

  assert.equal(existsSync(modulePath), true);
});

test("runtime pins the Pi coding agent package", () => {
  const packagePath = fileURLToPath(new URL("../../package.json", import.meta.url));
  const packageJson = JSON.parse(readFileSync(packagePath, "utf8"));

  assert.equal(packageJson.dependencies["@earendil-works/pi-coding-agent"], "0.83.0");
  assert.equal(packageJson.dependencies["brace-expansion"], "5.0.9");
  assert.equal(packageJson.dependencies.undici, "8.10.0");
});

test("runtime ships explicit policy extension, plugin and skill locks, and container image", () => {
  for (const relative of [
    "../../extensions/policy.ts",
    "../../plugins.lock.json",
    "../../skills.lock.json",
    "../../skills/test-gap/SKILL.md",
    "../../Dockerfile.agent",
  ]) {
    assert.equal(existsSync(fileURLToPath(new URL(relative, import.meta.url))), true, relative);
  }
});

test("agent image copies every TypeScript build configuration it invokes", () => {
  const dockerfile = readFileSync(
    fileURLToPath(new URL("../../Dockerfile.agent", import.meta.url)),
    "utf8",
  );

  assert.match(dockerfile, /COPY tsconfig\.json tsconfig\.extensions\.json/);
  assert.match(dockerfile, /COPY skills \.\/skills/);
  assert.match(dockerfile, /COPY skills\.lock\.json \.\/skills\.lock\.json/);
  assert.match(dockerfile, /ENV PATH="\/opt\/pi\/node_modules\/\.bin:/);
});

test("demo image setup refreshes the deployment image lock", () => {
  const scriptPath = fileURLToPath(new URL("../../scripts/lock-image.mjs", import.meta.url));
  assert.equal(existsSync(scriptPath), true);
  const source = readFileSync(scriptPath, "utf8");
  assert.match(source, /plugin checksum is invalid/);
  assert.match(source, /skill checksum is invalid/);
  assert.match(source, /docker.*image.*inspect/s);
});
