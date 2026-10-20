import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { RESOURCE_LIMIT_FLAGS } from "../src/container-flags.js";
import { buildAgentDockerArgs } from "../src/container-runtime.js";
import { dockerFixerRunner } from "../src/fixers.js";
import { DockerVerifier, VerifierScheduler, validateProfile, type RunContainer } from "../src/verification.js";

/** True when `flags` appears in `args` as one contiguous run. */
function contains(args: string[], flags: readonly string[]): boolean {
  return args.some((_, start) => flags.every((flag, offset) => args[start + offset] === flag));
}

test("the shared limits reap zombies, forbid swap and cap open files and file size", () => {
  const rendered = RESOURCE_LIMIT_FLAGS.join(" ");
  for (const expected of ["--init", "--memory 2g", "--memory-swap 2g", "--pids-limit 256", "--ulimit nofile=1024:1024", "--ulimit fsize=1073741824"]) {
    assert.ok(rendered.includes(expected), expected);
  }
});

test("writer, verifier and fixer containers all carry the shared limits", async () => {
  const writer = buildAgentDockerArgs({
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", gatewayToken: "t", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  });
  assert.ok(contains(writer, RESOURCE_LIMIT_FLAGS), "writer");

  const fixerCalls: string[][] = [];
  await dockerFixerRunner(`sha256:${"a".repeat(64)}`, async ({ args }) => { fixerCalls.push(args); return { output: "", exit: 0, timedOut: false }; })(
    { name: "fmt", argv: ["fmt"], cwd: ".", timeoutMs: 1000 }, "/tmp/ws", new AbortController().signal);
  assert.ok(contains(fixerCalls[0] ?? [], RESOURCE_LIMIT_FLAGS), "fixer");

  const root = await mkdtemp(path.join(tmpdir(), "flags-"));
  try {
    await mkdir(path.join(root, "candidate"));
    const profile = validateProfile({ id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"], checks: [{ name: "lint", argv: ["lint"], cwd: ".", timeoutMs: 1000 }] });
    const verifierCalls: string[][] = [];
    const runContainer: RunContainer = async ({ args }) => { verifierCalls.push(args); return { output: "", exit: 0, timedOut: false }; };
    await new DockerVerifier(profile, { runContainer, removeContainer: async () => {}, scheduler: new VerifierScheduler(1) })
      .verify({ hash: "h", path: path.join(root, "candidate") }, new AbortController().signal, ["candidate"], profile.checks);
    assert.ok(verifierCalls.length > 0 && verifierCalls.every(args => contains(args, RESOURCE_LIMIT_FLAGS)), "verifier");
  } finally { await rm(root, { recursive: true, force: true }); }
});
