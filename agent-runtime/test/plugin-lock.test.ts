import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { verifyPluginLock } from "../src/plugin-lock.js";

test("accepts exact-version allowlisted plugins with matching sha256", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-plugin-"));
  const content = "export default () => {};\n";
  await writeFile(path.join(root, "safe.ts"), content);
  const digest = createHash("sha256").update(content).digest("hex");
  await writeFile(
    path.join(root, "plugins.lock.json"),
    JSON.stringify({
      schema_version: 1,
      image_digest: "sha256:image",
      plugins: [{
        id: "safe",
        source: "npm:@example/safe",
        version: "1.2.3",
        integrity: `sha256:${digest}`,
        container_path: "/opt/pi/plugins/safe.ts",
        host_path: "safe.ts",
        enabled_tools: ["safe_tool"],
        risk_class: "read_only",
      }],
    }),
  );

  const result = await verifyPluginLock(path.join(root, "plugins.lock.json"), {
    rootDir: root,
    actualImageDigest: "sha256:image",
  });

  assert.equal(result.healthy, true);
  assert.deepEqual(result.plugins.map((plugin) => plugin.id), ["safe"]);
});

test("rejects checksum, image, floating version, and undeclared path mismatches", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-plugin-bad-"));
  await writeFile(path.join(root, "bad.ts"), "changed");
  await writeFile(
    path.join(root, "plugins.lock.json"),
    JSON.stringify({
      schema_version: 1,
      image_digest: "sha256:expected",
      plugins: [{
        id: "bad",
        source: "npm:@example/bad",
        version: "^1.0.0",
        integrity: "sha256:deadbeef",
        container_path: "/tmp/bad.ts",
        host_path: "bad.ts",
        enabled_tools: [],
        risk_class: "read_only",
      }],
    }),
  );

  const result = await verifyPluginLock(path.join(root, "plugins.lock.json"), {
    rootDir: root,
    actualImageDigest: "sha256:actual",
  });

  assert.equal(result.healthy, false);
  assert.match(result.errors.join("\n"), /image digest|exact version|container path|checksum/i);
});
