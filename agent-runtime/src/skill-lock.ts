import { createHash } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import path from "node:path";

export interface LockedSkill {
  name: string;
  version: string;
  integrity: string;
  container_path: string;
  host_path: string;
  risk_class: "read_only" | "workspace_write" | "privileged";
}

export interface SkillLockResult {
  healthy: boolean;
  errors: string[];
  skills: LockedSkill[];
}

const EXACT_SEMVER = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;
const GIT_COMMIT = /^[0-9a-f]{40}$/i;
const SHA256 = /^sha256:([0-9a-f]{64})$/i;

export async function verifySkillLock(
  lockPath: string,
  options: { rootDir?: string; imageLockContent?: string } = {},
): Promise<SkillLockResult> {
  let lockContent: string;
  let parsed: unknown;
  try {
    lockContent = await readFile(lockPath, "utf8");
    parsed = JSON.parse(lockContent);
  } catch {
    return { healthy: false, errors: ["skill lock is missing or invalid JSON"], skills: [] };
  }
  if (!isRecord(parsed) || parsed.schema_version !== 1 || !Array.isArray(parsed.skills)) {
    return { healthy: false, errors: ["skill lock schema is invalid"], skills: [] };
  }

  const errors: string[] = [];
  if (options.imageLockContent !== undefined && options.imageLockContent !== lockContent) {
    errors.push("skills.lock.json does not match the manifest embedded in agent image");
  }
  const root = await realpath(options.rootDir ?? path.dirname(lockPath));
  const skills: LockedSkill[] = [];
  const seenNames = new Set<string>();
  const seenContainerPaths = new Set<string>();
  const seenHostPaths = new Set<string>();

  for (const candidate of parsed.skills) {
    if (!isLockedSkill(candidate)) {
      errors.push("skill entry has an invalid schema");
      continue;
    }
    skills.push(candidate);
    if (seenNames.has(candidate.name)) errors.push(`duplicate skill name: ${candidate.name}`);
    if (seenContainerPaths.has(candidate.container_path)) errors.push(`duplicate skill container path: ${candidate.container_path}`);
    if (seenHostPaths.has(candidate.host_path)) errors.push(`duplicate skill host path: ${candidate.host_path}`);
    seenNames.add(candidate.name);
    seenContainerPaths.add(candidate.container_path);
    seenHostPaths.add(candidate.host_path);

    if (!EXACT_SEMVER.test(candidate.version) && !GIT_COMMIT.test(candidate.version)) {
      errors.push(`skill ${candidate.name} must use an exact version or Git commit`);
    }
    const expectedContainerPath = `/opt/pi/skills/${candidate.name}/SKILL.md`;
    if (candidate.container_path !== expectedContainerPath) {
      errors.push(`skill ${candidate.name} container path must be ${expectedContainerPath}`);
    }

    const hostPath = path.resolve(root, candidate.host_path);
    if (hostPath === root || !hostPath.startsWith(`${root}${path.sep}`)) {
      errors.push(`skill ${candidate.name} host path escapes the lock root`);
      continue;
    }
    if (path.basename(hostPath) !== "SKILL.md") {
      errors.push(`skill ${candidate.name} host path must point to SKILL.md`);
      continue;
    }

    const match = SHA256.exec(candidate.integrity);
    if (!match) {
      errors.push(`skill ${candidate.name} checksum is not sha256`);
      continue;
    }
    try {
      if ((await lstat(hostPath)).isSymbolicLink()) {
        errors.push(`skill ${candidate.name} file must not be a symlink`);
        continue;
      }
      const canonicalHostPath = await realpath(hostPath);
      if (!canonicalHostPath.startsWith(`${root}${path.sep}`)) {
        errors.push(`skill ${candidate.name} host path escapes the lock root`);
        continue;
      }
      const content = await readFile(canonicalHostPath, "utf8");
      const digest = createHash("sha256").update(content).digest("hex");
      if (digest !== match[1].toLowerCase()) errors.push(`skill ${candidate.name} checksum mismatch`);
      if (frontmatterName(content) !== candidate.name) errors.push(`skill ${candidate.name} frontmatter name mismatch`);
    } catch {
      errors.push(`skill ${candidate.name} file is missing`);
    }
  }

  return { healthy: errors.length === 0, errors, skills };
}

function frontmatterName(content: string): string | undefined {
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(content);
  if (!match) return undefined;
  return /^name:\s*([a-z0-9-]+)\s*$/m.exec(match[1])?.[1];
}

function isLockedSkill(value: unknown): value is LockedSkill {
  if (!isRecord(value)) return false;
  return (
    typeof value.name === "string" && /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value.name) &&
    typeof value.version === "string" &&
    typeof value.integrity === "string" &&
    typeof value.container_path === "string" &&
    typeof value.host_path === "string" &&
    ["read_only", "workspace_write", "privileged"].includes(String(value.risk_class))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
