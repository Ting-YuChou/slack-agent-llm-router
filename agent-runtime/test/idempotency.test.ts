import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { IdempotencyCache, IdempotencyKeyReusedError } from "../src/idempotency.js";

test("a repeated key returns the first result without running the work again", async () => {
  const cache = new IdempotencyCache();
  let runs = 0;
  const work = async () => ({ run_id: `R${++runs}` });
  assert.deepEqual(await cache.run("k", "body-a", work), { run_id: "R1" });
  assert.deepEqual(await cache.run("k", "body-a", work), { run_id: "R1" });
  assert.equal(runs, 1);
});

test("concurrent retries share one execution", async () => {
  const cache = new IdempotencyCache();
  let runs = 0; let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const work = async () => { runs++; await gate; return { run_id: "R1" }; };
  const both = Promise.all([cache.run("k", "body", work), cache.run("k", "body", work)]);
  release();
  assert.deepEqual(await both, [{ run_id: "R1" }, { run_id: "R1" }]);
  assert.equal(runs, 1);
});

test("reusing a key for a different request is refused", async () => {
  const cache = new IdempotencyCache();
  await cache.run("k", "body-a", async () => ({ run_id: "R1" }));
  await assert.rejects(cache.run("k", "body-b", async () => ({ run_id: "R2" })), IdempotencyKeyReusedError);
});

test("a failed request is not remembered, so its retry runs", async () => {
  const cache = new IdempotencyCache();
  await assert.rejects(cache.run("k", "body", async () => { throw new Error("busy"); }));
  assert.deepEqual(await cache.run("k", "body", async () => ({ run_id: "R2" })), { run_id: "R2" });
});

test("keys expire after their TTL and the cache stays bounded", async () => {
  let now = 0;
  const cache = new IdempotencyCache({ ttlMs: 1000, max: 2, now: () => now });
  let runs = 0; const work = async () => ++runs;
  await cache.run("a", "x", work); await cache.run("b", "x", work); await cache.run("c", "x", work);
  assert.equal(cache.size, 2, "the oldest key is evicted");
  now = 2000;
  assert.equal(await cache.run("c", "x", work), 4, "an expired key runs again");
});

test("remembered results survive a restart through the state file", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "idempotency-"));
  try {
    const file = path.join(root, "idempotency.json");
    const first = new IdempotencyCache({ file });
    await first.run("k", "body", async () => ({ run_id: "R1" }));
    const restarted = new IdempotencyCache({ file });
    await restarted.load();
    let ran = false;
    assert.deepEqual(await restarted.run("k", "body", async () => { ran = true; return { run_id: "R2" }; }), { run_id: "R1" });
    assert.equal(ran, false);
  } finally { await rm(root, { recursive: true, force: true }); }
});
