import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";

/** What the agent container loads: each file is hashed, except compiled sources, which get one tree digest. */
export const IMAGE_MANIFEST_FILES = ["/opt/pi/extensions", "/opt/pi/plugins", "/opt/pi/skills", "/opt/pi/skills.lock.json"] as const;
export const IMAGE_MANIFEST_TREES = ["/opt/pi/dist/src"] as const;

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
  return `
const fs = require("fs"), crypto = require("crypto");
const ROOT = ${JSON.stringify(root)}, FILES = ${JSON.stringify(IMAGE_MANIFEST_FILES)}, TREES = ${JSON.stringify(IMAGE_MANIFEST_TREES)};
const hash = (data) => crypto.createHash("sha256").update(data).digest("hex");
const walk = (at, visit) => {
  let stat; try { stat = fs.lstatSync(ROOT + at); } catch { return; }
  if (stat.isSymbolicLink()) return visit(at, "symlink:" + fs.readlinkSync(ROOT + at));
  if (stat.isDirectory()) { for (const name of fs.readdirSync(ROOT + at).sort()) walk(at + "/" + name, visit); return; }
  visit(at, "sha256:" + hash(fs.readFileSync(ROOT + at)));
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
  const output = await run(["run", "--rm", "--network", "none", "--read-only", "--entrypoint", "node", image, "-e", imageManifestScript()]);
  let parsed: unknown;
  try { parsed = JSON.parse(output); } catch { parsed = undefined; }
  if (!isManifest(parsed)) throw new Error(`Invalid image manifest from ${image}`);
  return parsed;
}

/** Every difference between the locked manifest and the image's, as readable errors; none when they match. */
export function compareImageManifest(expected: unknown, actual: ImageManifest): string[] {
  if (!isManifest(expected)) return ["plugins.lock.json has no image_manifest; run npm run lock-image"];
  const errors: string[] = [];
  for (const at of [...new Set([...Object.keys(expected), ...Object.keys(actual)])].sort()) {
    if (!(at in actual)) errors.push(`removed: ${at}`);
    else if (!(at in expected)) errors.push(`added: ${at}`);
    else if (expected[at] !== actual[at]) errors.push(`changed: ${at}`);
  }
  return errors;
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
