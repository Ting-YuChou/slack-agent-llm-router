import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

/** What the agent container loads: each file is hashed, except compiled sources, which get one tree digest. */
export const IMAGE_MANIFEST_FILES = ["/opt/pi/extensions", "/opt/pi/plugins", "/opt/pi/skills", "/opt/pi/skills.lock.json"] as const;
export const IMAGE_MANIFEST_TREES = ["/opt/pi/dist/src"] as const;
/** What Dockerfile.agent copies from agent-runtime/ and builds the image from; compared with the host's tree at startup. */
export const IMAGE_SOURCE_FILES = ["/opt/pi/src", "/opt/pi/prompts", "/opt/pi/extensions", "/opt/pi/plugins", "/opt/pi/skills", "/opt/pi/skills.lock.json"] as const;
const IMAGE_ROOT = "/opt/pi";

export type ImageManifest = Record<string, string>;
export type DockerRunner = (args: string[]) => Promise<string>;

const exec = promisify(execFile);
const dockerRunner: DockerRunner = async (args) => (await exec("docker", args, { encoding: "utf8", maxBuffer: 16 * 1024 * 1024, env: { PATH: process.env.PATH } })).stdout;

/**
 * Runs inside the image (node only, no network): hashes IMAGE_MANIFEST_FILES per file and the compiled
 * JavaScript under IMAGE_MANIFEST_TREES as one digest. Symlinks are recorded, never followed. `root`
 * prefixes the paths read, for tests; the manifest's keys are always the in-image paths.
 */
export function imageManifestScript(root = ""): string {
  return manifestScript(root + IMAGE_ROOT, IMAGE_MANIFEST_FILES, IMAGE_MANIFEST_TREES);
}

/**
 * Hashes the agent-runtime sources Dockerfile.agent copies into the image, per file, under their in-image
 * keys. `base` is where they are read: /opt/pi inside the image, the agent-runtime directory on the host.
 */
export function sourceManifestScript(base = IMAGE_ROOT): string {
  return manifestScript(base, IMAGE_SOURCE_FILES, []);
}

/** One walker for both manifests, so the host and the image can only disagree about bytes, never about how they hash. */
function manifestScript(base: string, files: readonly string[], trees: readonly string[]): string {
  return `
const fs = require("fs"), crypto = require("crypto");
const BASE = ${JSON.stringify(base)}, PREFIX = ${JSON.stringify(IMAGE_ROOT)}, FILES = ${JSON.stringify(files)}, TREES = ${JSON.stringify(trees)};
const read = (at) => BASE + at.slice(PREFIX.length);
const hash = (data) => crypto.createHash("sha256").update(data).digest("hex");
const walk = (at, visit) => {
  let stat; try { stat = fs.lstatSync(read(at)); } catch { return; }
  if (stat.isSymbolicLink()) return visit(at, "symlink:" + fs.readlinkSync(read(at)));
  if (stat.isDirectory()) { for (const name of fs.readdirSync(read(at)).sort()) walk(at + "/" + name, visit); return; }
  visit(at, "sha256:" + hash(fs.readFileSync(read(at))));
};
const manifest = {};
for (const file of FILES) walk(file, (at, digest) => { manifest[at] = digest; });
for (const tree of TREES) {
  const entries = [];
  walk(tree, (at, digest) => { if (at.endsWith(".js") || digest.startsWith("symlink:")) entries.push(at + "\\0" + digest); });
  if (entries.length) manifest[tree] = "tree-sha256:" + hash(entries.join("\\n"));
}
process.stdout.write(JSON.stringify(manifest));
`;
}

const DIGEST = /^(?:sha256:[a-f0-9]{64}|tree-sha256:[a-f0-9]{64}|symlink:.{1,4096})$/;

/** The manifest of the image as it is, read from the image itself (pass its digest), offline and read-only. */
export async function computeImageManifest(image: string, run: DockerRunner = dockerRunner): Promise<ImageManifest> {
  return parseManifest(await runInImage(image, imageManifestScript(), run), `Invalid image manifest from ${image}`);
}

/** The agent-runtime sources the image was built from, read from the image itself, offline and read-only. */
export async function computeImageSources(image: string, run: DockerRunner = dockerRunner): Promise<ImageManifest> {
  return parseManifest(await runInImage(image, sourceManifestScript(), run), `Invalid image sources from ${image}`);
}

/** The same sources in the host's agent-runtime checkout, hashed by the same script. */
export async function computeHostSources(runtimeRoot: string): Promise<ImageManifest> {
  const { stdout } = await exec(process.execPath, ["-e", sourceManifestScript(runtimeRoot)], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  return parseManifest(stdout, `Invalid host sources under ${runtimeRoot}`);
}

function runInImage(image: string, script: string, run: DockerRunner): Promise<string> {
  return run(["run", "--rm", "--network", "none", "--read-only", "--entrypoint", "node", image, "-e", script]);
}

function parseManifest(output: string, invalid: string): ImageManifest {
  let parsed: unknown;
  try { parsed = JSON.parse(output); } catch { parsed = undefined; }
  if (!isManifest(parsed)) throw new Error(invalid);
  return parsed;
}

/** Every difference between the locked manifest and the image's, as readable errors; none when they match. */
export function compareImageManifest(expected: unknown, actual: ImageManifest): string[] {
  if (!isManifest(expected)) return ["plugins.lock.json has no image_manifest; run npm run lock-image"];
  return diffManifests(expected, actual, { removed: "removed", added: "added" }, (at) => at);
}

function diffManifests(expected: ImageManifest, actual: ImageManifest, words: { removed: string; added: string }, name: (at: string) => string): string[] {
  const errors: string[] = [];
  for (const at of [...new Set([...Object.keys(expected), ...Object.keys(actual)])].sort()) {
    if (!(at in actual)) errors.push(`${words.removed}: ${name(at)}`);
    else if (!(at in expected)) errors.push(`${words.added}: ${name(at)}`);
    else if (expected[at] !== actual[at]) errors.push(`changed: ${name(at)}`);
  }
  return errors;
}

/** Every source that differs between the host's agent-runtime tree and the one the image was built from. */
export function compareSources(host: ImageManifest, image: ImageManifest): string[] {
  return diffManifests(host, image, { removed: "missing from the image", added: "only in the image" }, (at) => path.posix.relative(IMAGE_ROOT, at));
}

function isManifest(value: unknown): value is ImageManifest {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const entries = Object.entries(value);
  return entries.length > 0 && entries.every(([at, digest]) => at.startsWith("/opt/pi/") && typeof digest === "string" && DIGEST.test(digest));
}

/** Refuses to run an image whose loaded code differs from what plugins.lock.json recorded at lock time. */
export async function assertImageManifest(lockPath: string, image: string, run: DockerRunner = dockerRunner): Promise<void> {
  const lock = JSON.parse(await readFile(lockPath, "utf8")) as { image_manifest?: unknown };
  const errors = compareImageManifest(lock.image_manifest, await computeImageManifest(image, run));
  if (errors.length) throw new Error(`Agent image does not match plugins.lock.json: ${errors.slice(0, 5).join("; ")}`);
}

/**
 * Refuses an image built from a different agent-runtime tree than the one running. The lock pins whatever
 * image was locked, so an image built before a change to the policy or the credential hook would still
 * pass it, then fail on every model call; this names the stale files instead.
 */
export async function assertImageMatchesHost(image: string, runtimeRoot: string, run: DockerRunner = dockerRunner): Promise<void> {
  const [host, built] = await Promise.all([computeHostSources(runtimeRoot), computeImageSources(image, run)]);
  const errors = compareSources(host, built);
  if (errors.length) {
    throw new Error(`Agent image ${image} was built from a different agent-runtime tree (${errors.slice(0, 5).join("; ")}${errors.length > 5 ? `; ${errors.length - 5} more` : ""}); rebuild and relock it: make demo-agent-images`);
  }
}
