import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { diskUsageBytes, diskQuotaFromEnv, sessionQuotaFromEnv } from "../src/disk-usage.js";

test("disk usage counts the bytes written under a directory", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "du-"));
  try {
    await writeFile(path.join(root, "big.bin"), Buffer.alloc(512 * 1024));
    assert.ok(await diskUsageBytes(root) >= 512 * 1024);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("the worktree quota defaults to 2 GiB every 15 s and rejects nonsense", () => {
  const quota = diskQuotaFromEnv({});
  assert.equal(quota.bytes, 2 * 1024 ** 3);
  assert.equal(quota.intervalMs, 15_000);
  assert.equal(diskQuotaFromEnv({ PI_AGENT_WORKTREE_QUOTA_BYTES: "1048576" }).bytes, 1048576);
  assert.throws(() => diskQuotaFromEnv({ PI_AGENT_WORKTREE_QUOTA_BYTES: "-1" }), /PI_AGENT_WORKTREE_QUOTA_BYTES/);
});

test("the session quota defaults to 256 MiB every 15 s, is set apart from the worktree's, and rejects nonsense", () => {
  const quota = sessionQuotaFromEnv({});
  assert.equal(quota.bytes, 256 * 1024 ** 2);
  assert.equal(quota.intervalMs, 15_000);
  assert.equal(quota.usage, diskUsageBytes);
  assert.equal(sessionQuotaFromEnv({ PI_AGENT_SESSION_QUOTA_BYTES: "1048576", PI_AGENT_WORKTREE_QUOTA_BYTES: "4096" }).bytes, 1048576);
  assert.equal(diskQuotaFromEnv({ PI_AGENT_SESSION_QUOTA_BYTES: "1048576" }).bytes, 2 * 1024 ** 3);
  for (const raw of ["-1", "0", "1.5", "lots"]) {
    assert.throws(() => sessionQuotaFromEnv({ PI_AGENT_SESSION_QUOTA_BYTES: raw }), /PI_AGENT_SESSION_QUOTA_BYTES/);
  }
});
