import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

import { assertImageManifest, assertImageMatchesHost, compareImageManifest, computeHostSources, computeImageManifest, computeImageSources, imageManifestScript, sourceManifestScript } from "../src/image-manifest.js";
import { AGENT_TEST_IMAGE } from "./support/docker-images.js";

const exec = promisify(execFile);
const enabled = process.env.PI_AGENT_DOCKER_INTEGRATION === "1";
const sha = (text: string) => `sha256:${createHash("sha256").update(text).digest("hex")}`;

/** A fake /opt/pi under `root`, as the agent image lays it out. */
async function fakeImageTree(root: string) {
  const at = (file: string) => path.join(root, file);
  for (const dir of ["opt/pi/extensions", "opt/pi/plugins", "opt/pi/skills/test-gap", "opt/pi/dist/src", "opt/pi/node_modules/x"]) await mkdir(at(dir), { recursive: true });
  await writeFile(at("opt/pi/extensions/policy.ts"), "export default 1;\n");
  await writeFile(at("opt/pi/plugins/lsp.ts"), "lsp\n");
  await writeFile(at("opt/pi/skills/test-gap/SKILL.md"), "# skill\n");
  await writeFile(at("opt/pi/skills.lock.json"), "{}\n");
  await writeFile(at("opt/pi/dist/src/policy.js"), "export const a = 1;\n");
  await writeFile(at("opt/pi/dist/src/policy.js.map"), "{}\n");
  await writeFile(at("opt/pi/node_modules/x/index.js"), "ignored\n");
}

const runScript = async (root: string) => JSON.parse((await exec("node", ["-e", imageManifestScript(root)], { encoding: "utf8" })).stdout) as Record<string, string>;

test("the in-image manifest hashes each loaded file and the compiled sources as one tree", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "image-manifest-"));
  try {
    await fakeImageTree(root);
    const manifest = await runScript(root);
    assert.deepEqual(Object.keys(manifest).sort(), [
      "/opt/pi/dist/src", "/opt/pi/extensions/policy.ts", "/opt/pi/plugins/lsp.ts", "/opt/pi/skills.lock.json", "/opt/pi/skills/test-gap/SKILL.md",
    ]);
    assert.equal(manifest["/opt/pi/extensions/policy.ts"], sha("export default 1;\n"));
    assert.match(manifest["/opt/pi/dist/src"], /^tree-sha256:[a-f0-9]{64}$/);
    // Source maps do not run; compiled JavaScript does.
    await writeFile(path.join(root, "opt/pi/dist/src/policy.js.map"), "{\"changed\":true}\n");
    assert.equal((await runScript(root))["/opt/pi/dist/src"], manifest["/opt/pi/dist/src"]);
    await writeFile(path.join(root, "opt/pi/dist/src/policy.js"), "export const a = 2;\n");
    assert.notEqual((await runScript(root))["/opt/pi/dist/src"], manifest["/opt/pi/dist/src"]);
    await symlink("/etc/passwd", path.join(root, "opt/pi/extensions/evil.ts"));
    assert.equal((await runScript(root))["/opt/pi/extensions/evil.ts"], "symlink:/etc/passwd");
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("any changed, added or removed file in the image is reported", () => {
  const locked = { "/opt/pi/extensions/policy.ts": sha("a"), "/opt/pi/dist/src": `tree-sha256:${"0".repeat(64)}` };
  assert.deepEqual(compareImageManifest(locked, { ...locked }), []);
  assert.match(compareImageManifest(locked, { ...locked, "/opt/pi/extensions/policy.ts": sha("b") }).join("\n"), /changed: \/opt\/pi\/extensions\/policy\.ts/);
  assert.match(compareImageManifest(locked, { ...locked, "/opt/pi/extensions/extra.ts": sha("c") }).join("\n"), /added: \/opt\/pi\/extensions\/extra\.ts/);
  const { ["/opt/pi/dist/src"]: _tree, ...withoutTree } = locked;
  assert.match(compareImageManifest(locked, withoutTree).join("\n"), /removed: \/opt\/pi\/dist\/src/);
  for (const missing of [undefined, null, [], "x", {}]) assert.match(compareImageManifest(missing, locked).join("\n"), /no image_manifest/);
});

test("the manifest is read from the image by digest, offline and read-only, and malformed output is refused", async () => {
  const calls: string[][] = [];
  const manifest = { "/opt/pi/extensions/policy.ts": sha("a") };
  assert.deepEqual(await computeImageManifest("sha256:abc", async (args) => { calls.push(args); return JSON.stringify(manifest); }), manifest);
  const args = calls[0].join(" ");
  assert.match(args, /^run --rm --network none --read-only --entrypoint node sha256:abc -e /);
  for (const output of ["", "not json", "[]", JSON.stringify({ "/etc/passwd": sha("a") }), JSON.stringify({ "/opt/pi/x": "md5:1" }), "{}"]) {
    await assert.rejects(computeImageManifest("sha256:abc", async () => output), /image manifest/i, output);
  }
});

test("startup refuses an image whose loaded code differs from the lock", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "image-manifest-lock-"));
  const lockPath = path.join(dir, "plugins.lock.json");
  const image = { "/opt/pi/extensions/policy.ts": sha("policy") };
  try {
    await writeFile(lockPath, JSON.stringify({ schema_version: 1, image_manifest: image }));
    await assertImageManifest(lockPath, "sha256:abc", async () => JSON.stringify(image));
    await assert.rejects(assertImageManifest(lockPath, "sha256:abc", async () => JSON.stringify({ "/opt/pi/extensions/policy.ts": sha("policy!") })), /changed: \/opt\/pi\/extensions\/policy\.ts/);
    await writeFile(lockPath, JSON.stringify({ schema_version: 1 }));
    await assert.rejects(assertImageManifest(lockPath, "sha256:abc", async () => JSON.stringify(image)), /no image_manifest/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

/** An agent-runtime checkout as Dockerfile.agent copies it to /opt/pi, with what the image never copies beside it. */
async function fakeRuntimeTree(base: string) {
  const at = (file: string) => path.join(base, file);
  for (const dir of ["src", "prompts", "extensions", "plugins", "skills/test-gap", "node_modules/x", "dist/src", "test"]) await mkdir(at(dir), { recursive: true });
  await writeFile(at("src/policy.ts"), "export const a = 1;\n");
  await writeFile(at("prompts/writer.md"), "# writer\n");
  await writeFile(at("extensions/policy.ts"), "export default 1;\n");
  await writeFile(at("plugins/lsp.ts"), "lsp\n");
  await writeFile(at("skills/test-gap/SKILL.md"), "# skill\n");
  await writeFile(at("skills.lock.json"), "{}\n");
  await writeFile(at("node_modules/x/index.js"), "ignored\n");
  await writeFile(at("dist/src/policy.js"), "ignored\n");
  await writeFile(at("test/policy.test.ts"), "ignored\n");
}

const runSourceScript = async (base: string) => JSON.parse((await exec("node", ["-e", sourceManifestScript(base)], { encoding: "utf8" })).stdout) as Record<string, string>;

test("the sources an image was built from hash the same on the host as inside the image", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "image-sources-"));
  try {
    const host = path.join(root, "agent-runtime");
    const image = path.join(root, "image/opt/pi");
    await fakeRuntimeTree(host);
    await fakeRuntimeTree(image);
    const fromHost = await computeHostSources(host);
    assert.deepEqual(Object.keys(fromHost).sort(), [
      "/opt/pi/extensions/policy.ts", "/opt/pi/plugins/lsp.ts", "/opt/pi/prompts/writer.md", "/opt/pi/skills.lock.json", "/opt/pi/skills/test-gap/SKILL.md", "/opt/pi/src/policy.ts",
    ]);
    assert.equal(fromHost["/opt/pi/src/policy.ts"], sha("export const a = 1;\n"));
    // The same script runs in the image, rooted at /opt/pi there.
    assert.deepEqual(await runSourceScript(image), fromHost);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("startup refuses an image built from a different agent-runtime tree, even when the lock matches it", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "image-stale-"));
  try {
    await fakeRuntimeTree(root);
    const image = await computeHostSources(root);
    const calls: string[][] = [];
    await assertImageMatchesHost("sha256:abc", root, async (args) => { calls.push(args); return JSON.stringify(image); });
    assert.match(calls[0].join(" "), /^run --rm --network none --read-only --entrypoint node sha256:abc -e /);
    // An image built before a source change: its compiled policy and extensions are older than the host's.
    await writeFile(path.join(root, "src/policy.ts"), "export const a = 2;\n");
    await writeFile(path.join(root, "extensions/credentials.ts"), "export default 2;\n");
    await assert.rejects(
      assertImageMatchesHost("sha256:abc", root, async () => JSON.stringify(image)),
      (error: Error) => /sha256:abc/.test(error.message)
        && /changed: src\/policy\.ts/.test(error.message)
        && /missing from the image: extensions\/credentials\.ts/.test(error.message)
        && /make demo-agent-images/.test(error.message),
    );
    for (const output of ["", "not json", "{}", JSON.stringify({ "/etc/passwd": sha("a") })]) {
      await assert.rejects(computeImageSources("sha256:abc", async () => output), /image sources/i, output);
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("a freshly built agent image carries exactly the host's agent-runtime sources", { skip: !enabled, timeout: 180_000 }, async () => {
  const id = (await exec("docker", ["image", "inspect", "--format={{.Id}}", AGENT_TEST_IMAGE], { encoding: "utf8" })).stdout.trim();
  await assertImageMatchesHost(id, path.resolve(import.meta.dirname, "../.."));
});

test("a one-byte change to the policy extension inside a real image fails the manifest check", { skip: !enabled, timeout: 180_000 }, async () => {
  const id = (await exec("docker", ["image", "inspect", "--format={{.Id}}", AGENT_TEST_IMAGE], { encoding: "utf8" })).stdout.trim();
  const original = await computeImageManifest(id);
  const policy = original["/opt/pi/extensions/policy.ts"];
  assert.match(policy, /^sha256:/);
  assert.match(original["/opt/pi/dist/src"], /^tree-sha256:/);
  const dir = await mkdtemp(path.join(tmpdir(), "image-manifest-tamper-"));
  const tag = `slack-pi-agent-tampered:${Math.random().toString(16).slice(2, 10)}`;
  try {
    await writeFile(path.join(dir, "Dockerfile"), `FROM ${AGENT_TEST_IMAGE}\nRUN printf ' ' >> /opt/pi/extensions/policy.ts\n`);
    await exec("docker", ["build", "-q", "-t", tag, dir], { encoding: "utf8" });
    const tampered = await computeImageManifest(tag);
    assert.deepEqual(compareImageManifest(original, tampered), ["changed: /opt/pi/extensions/policy.ts"]);
  } finally {
    await exec("docker", ["image", "rm", "-f", tag]).catch(() => undefined);
    await rm(dir, { recursive: true, force: true });
  }
});
