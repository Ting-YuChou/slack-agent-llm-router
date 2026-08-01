import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { verifySkillLock } from "../src/skill-lock.js";

test("accepts explicitly allowlisted skills with matching metadata and sha256", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-skill-"));
  const content = "---\nname: test-gap\ndescription: Find a missing test.\n---\n\nDo the work.\n";
  await mkdir(path.join(root, "skills", "test-gap"), { recursive: true });
  await writeFile(path.join(root, "skills", "test-gap", "SKILL.md"), content);
  const digest = createHash("sha256").update(content).digest("hex");
  await writeFile(
    path.join(root, "skills.lock.json"),
    JSON.stringify({
      schema_version: 1,
      skills: [{
        name: "test-gap",
        version: "0.1.0",
        integrity: `sha256:${digest}`,
        container_path: "/opt/pi/skills/test-gap/SKILL.md",
        host_path: "skills/test-gap/SKILL.md",
        risk_class: "workspace_write",
      }],
    }),
  );

  const result = await verifySkillLock(path.join(root, "skills.lock.json"), { rootDir: root });

  assert.equal(result.healthy, true);
  assert.deepEqual(result.skills.map((skill) => skill.name), ["test-gap"]);
});

test("rejects duplicates, floating versions, path mismatches, and invalid checksums", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-skill-bad-"));
  await writeFile(
    path.join(root, "skills.lock.json"),
    JSON.stringify({
      schema_version: 1,
      skills: [
        {
          name: "unsafe",
          version: "latest",
          integrity: "sha256:deadbeef",
          container_path: "/tmp/SKILL.md",
          host_path: "../SKILL.md",
          risk_class: "workspace_write",
        },
        {
          name: "unsafe",
          version: "0.1.0",
          integrity: `sha256:${"0".repeat(64)}`,
          container_path: "/opt/pi/skills/different/SKILL.md",
          host_path: "missing.md",
          risk_class: "workspace_write",
        },
      ],
    }),
  );

  const result = await verifySkillLock(path.join(root, "skills.lock.json"), { rootDir: root });

  assert.equal(result.healthy, false);
  assert.match(result.errors.join("\n"), /duplicate|exact version|container path|escapes|checksum|missing/i);
});

test("rejects a host skill lock that differs from the manifest embedded in the image", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-skill-image-"));
  const lockPath = path.join(root, "skills.lock.json");
  await writeFile(lockPath, JSON.stringify({ schema_version: 1, skills: [] }));

  const result = await verifySkillLock(lockPath, {
    rootDir: root,
    imageLockContent: JSON.stringify({ schema_version: 1, skills: [{ name: "different" }] }),
  });

  assert.equal(result.healthy, false);
  assert.match(result.errors.join("\n"), /embedded in agent image/i);
});
