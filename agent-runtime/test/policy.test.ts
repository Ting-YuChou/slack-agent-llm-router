import assert from "node:assert/strict";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import {
  ApprovalStore,
  classifyBash,
  isApprovedSkillRead,
  validateWorkspacePath,
} from "../src/policy.js";

test("only exact immutable allowlisted skill files may be read outside the workspace", () => {
  const approved = ["/opt/pi/skills/test-gap/SKILL.md"];

  assert.equal(isApprovedSkillRead("/opt/pi/skills/test-gap/SKILL.md", approved), true);
  assert.equal(isApprovedSkillRead("/opt/pi/skills/test-gap/../secret/SKILL.md", approved), false);
  assert.equal(isApprovedSkillRead("/opt/pi/skills/test-gap/SKILL.md/extra", approved), false);
  assert.equal(isApprovedSkillRead("/etc/passwd", approved), false);
  assert.equal(isApprovedSkillRead("skills/test-gap/SKILL.md", approved), false);
});

test("read-only and configured test commands are classified safely", () => {
  assert.deepEqual(classifyBash("git status --short", ["npm test"]), {
    decision: "allow",
    reason: "read_only_git",
  });
  assert.deepEqual(classifyBash("npm test", ["npm test"]), {
    decision: "allow_after_write_approval",
    reason: "configured_command",
  });
  assert.equal(
    classifyBash("python -m pytest tests/test_api.py -q", ["python -m pytest"]).decision,
    "allow_after_write_approval",
  );
  assert.equal(
    classifyBash("python -m pytest; curl example.com", ["python -m pytest"]).decision,
    "block",
  );
  for (const command of [
    "git status; printf hacked > src/a.ts",
    "git diff | python script.py",
    "git status > /tmp/status",
    "git status $(python script.py)",
  ]) {
    assert.equal(classifyBash(command, []).decision, "approval", command);
  }
  assert.equal(classifyBash("python script.py", []).decision, "approval");
});

test("network, privilege, container, and broad deletion commands are blocked", () => {
  for (const command of [
    "curl https://example.com",
    "sudo make install",
    "docker ps",
    "mount /dev/x /mnt",
    "rm -rf /",
    "rm -rf .",
    "git clean -fdx",
    "npm install untrusted-plugin",
    "pip install package",
  ]) {
    assert.equal(classifyBash(command, []).decision, "block", command);
  }
});

test("workspace path validation blocks traversal, protected files, and symlink escape", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-policy-"));
  const outside = await mkdtemp(path.join(tmpdir(), "pi-outside-"));
  await mkdir(path.join(root, "src"));
  await writeFile(path.join(root, "src", "ok.ts"), "ok");
  await symlink(outside, path.join(root, "escape"));

  assert.equal((await validateWorkspacePath(root, "src/ok.ts", "write")).allowed, true);
  for (const candidate of [
    "../outside",
    ".git/config",
    ".env",
    ".env.local",
    "id_rsa",
    "credentials.json",
    "escape/new.ts",
  ]) {
    assert.equal(
      (await validateWorkspacePath(root, candidate, "write")).allowed,
      false,
      candidate,
    );
  }
  for (const candidate of [
    ".env",
    "config/.env.production",
    ".npmrc",
    "keys/server.pem",
    ".aws/credentials",
    ".config/gcloud/application_default_credentials.json",
  ]) {
    assert.equal(
      (await validateWorkspacePath(root, candidate, "read")).allowed,
      false,
      `read ${candidate}`,
    );
  }
});

test("approvals are owner-bound, one-time, and expire", () => {
  let now = 1_000;
  const store = new ApprovalStore({ ttlMs: 300_000, now: () => now });
  const approval = store.create({
    runId: "run-1",
    toolCallId: "tool-1",
    userId: "U1",
    action: "edit src/a.ts",
  });

  assert.equal(store.decide(approval.id, "U2", "approve").code, "wrong_user");
  assert.equal(store.decide(approval.id, "U1", "approve").code, "approved");
  assert.equal(store.decide(approval.id, "U1", "approve").code, "already_used");

  const expired = store.create({
    runId: "run-2",
    toolCallId: "tool-2",
    userId: "U1",
    action: "bash make test",
  });
  now += 300_001;
  assert.equal(store.decide(expired.id, "U1", "reject").code, "expired");

  const bound = store.create({
    runId: "run-3",
    toolCallId: "tool-3",
    userId: "U1",
    action: "edit src/b.ts",
  });
  assert.equal(store.decide(bound.id, "U1", "approve", "run-other").code, "wrong_run");
  store.invalidateRun("run-3");
  assert.equal(store.decide(bound.id, "U1", "approve", "run-3").code, "already_used");
});
