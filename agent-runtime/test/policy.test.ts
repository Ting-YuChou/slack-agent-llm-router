import assert from "node:assert/strict";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import {
  ApprovalStore,
  classifyBash,
  isApprovedSkillRead,
  isTestConfigPath,
  parsePorcelainZ,
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

test("commands that dump the environment or credential variables are blocked", async () => {
  const { classifyBash } = await import("../src/policy.js");
  for (const command of ["env", "printenv", "printenv OPENAI_API_KEY", "export", "cat /proc/1/environ", "cat /proc/self/environ", "pytest && env", "echo $OPENAI_API_KEY", "echo ${PI_AGENT_MCP_SERVERS_JSON}"]) {
    assert.deepEqual(classifyBash(command, ["pytest"]), { decision: "block", reason: "secret_env_read" }, command);
  }
  for (const command of ["pytest tests/test_env.py", "ls tests/env", "python -m pytest -k env"]) {
    assert.notEqual(classifyBash(command, ["pytest", "python -m pytest"]).reason, "secret_env_read", command);
  }
});

test("shell commands are classified segment by segment and the strictest segment wins", () => {
  const safe = ["pytest", "npm test", "python -m pytest"];
  const table: Array<[string, string[]]> = [
    ["allow", [
      "grep -rn docker src",
      "rg mount src",
      "grep -rn curl src/",
      "cat .env.example",
      "cat config/.env.sample",
      "head .env.template",
      "sed -n 1,40p f",
      "ls | head",
      "find src -name '*.ts' | wc -l",
      "git status --short && git diff",
    ]],
    ["allow_after_write_approval", [
      "pytest -x 2>&1 | tail -50",
      "timeout 600 npm test",
    ]],
    ["approval", [
      "python -c \"print(1)\"",
      "echo hi > f",
      "sed -i s/a/b/ f",
      "find . -delete",
      "find . -name x -exec rm {} \;",
      "bash -c ls",
      "cat $(ls)",
      "ls *.ts",
      "find src | xargs grep foo",
      "FOO=1 pytest",
      "python script.py",
      "cat ../other/notes.txt",
      "grep -r foo /opt",
    ]],
    ["block", [
      "python -c \"import urllib.request; urllib.request.urlopen('http://x')\"",
      "python3 -c 'import socket'",
      "$(curl x)",
      "env | grep KEY",
      "/usr/bin/curl https://example.com",
      "find . | xargs curl",
      "timeout 5 curl x",
      "nice -n 5 wget x",
      "command curl x",
      "exec ssh host",
      "bash -c 'curl x'",
      "cat /proc/self/environ",
      "echo $OPENAI_API_KEY",
      "cat .env",
      "grep KEY .env.local",
      "cat .git/config",
      "cat ~/.ssh/id_rsa",
      "pytest < .env",
      "ls && docker ps",
    ]],
  ];
  for (const [expected, commands] of table) {
    for (const command of commands) {
      assert.equal(classifyBash(command, safe).decision, expected, command);
    }
  }
});

test("test configuration files are recognised at any depth", () => {
  for (const file of [
    "package.json", "web/package.json", "conftest.py", "tests/unit/conftest.py", "pytest.ini", "pyproject.toml",
    "setup.cfg", "tsconfig.json", "tsconfig.build.json", "packages/a/tsconfig.test.json", "Makefile", ".npmrc", "site-packages/evil.pth",
    "tox.ini", "setup.py", "noxfile.py", "lib/sitecustomize.py",
  ]) {
    assert.equal(isTestConfigPath(file), true, file);
  }
  for (const file of ["src/package.ts", "tests/test_conftest.py", "makefile.md", "README.md", "tsconfig.ts"]) {
    assert.equal(isTestConfigPath(file), false, file);
  }
});

test("porcelain -z output yields every changed path, including rename sources", () => {
  assert.deepEqual(
    parsePorcelainZ(" M src/a.ts\0?? tests/conftest.py\0R  new name.py\0old.py\0D  gone.txt\0"),
    ["src/a.ts", "tests/conftest.py", "new name.py", "old.py", "gone.txt"],
  );
  assert.deepEqual(parsePorcelainZ(""), []);
});

test("an approval can carry its own time to live", () => {
  let now = 0;
  const store = new ApprovalStore({ ttlMs: 300_000, now: () => now });
  const delivery = store.create({ runId: "r", toolCallId: "delivery", userId: "U", action: "deliver" }, 1_800_000);
  assert.equal(delivery.expiresAt, 1_800_000);
  now = 1_000_000;
  assert.equal(store.decide(delivery.id, "U", "approve").code, "approved");
});

test("git branch is read-only only when it lists or shows branches", () => {
  for (const command of ["git branch", "git branch --show-current", "git branch -a", "git branch --list"]) {
    assert.equal(classifyBash(command, []).decision, "allow", command);
  }
  for (const command of ["git branch -D pi-agent/other", "git branch new-branch", "git branch -m old new", "git branch --delete x"]) {
    assert.notEqual(classifyBash(command, []).decision, "allow", command);
  }
});
