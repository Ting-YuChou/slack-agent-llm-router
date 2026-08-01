import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import path from "node:path";

export interface LockedPlugin {
  id: string;
  source: string;
  version: string;
  integrity: string;
  container_path: string;
  host_path: string;
  enabled_tools: string[];
  risk_class: "read_only" | "workspace_write" | "privileged";
}

export interface PluginLockResult {
  healthy: boolean;
  errors: string[];
  plugins: LockedPlugin[];
  imageDigest?: string;
}

const EXACT_SEMVER = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;
const GIT_COMMIT = /^[0-9a-f]{40}$/i;
const SHA256 = /^sha256:([0-9a-f]{64})$/i;

export async function verifyPluginLock(
  lockPath: string,
  options: { rootDir?: string; actualImageDigest: string },
): Promise<PluginLockResult> {
  const errors: string[] = [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(lockPath, "utf8"));
  } catch {
    return { healthy: false, errors: ["plugin lock is missing or invalid JSON"], plugins: [] };
  }
  if (!isRecord(parsed) || parsed.schema_version !== 1 || !Array.isArray(parsed.plugins)) {
    return { healthy: false, errors: ["plugin lock schema is invalid"], plugins: [] };
  }
  const lockImageDigest = typeof parsed.image_digest === "string" ? parsed.image_digest : "";
  if (!lockImageDigest || lockImageDigest !== options.actualImageDigest) {
    errors.push("agent image digest does not match plugins.lock.json");
  }
  const root = await realpath(options.rootDir ?? path.dirname(lockPath));
  const plugins: LockedPlugin[] = [];
  const seenIds = new Set<string>();
  for (const candidate of parsed.plugins) {
    if (!isLockedPlugin(candidate)) {
      errors.push("plugin entry has an invalid schema");
      continue;
    }
    plugins.push(candidate);
    if (seenIds.has(candidate.id)) errors.push(`duplicate plugin id: ${candidate.id}`);
    seenIds.add(candidate.id);
    if (!EXACT_SEMVER.test(candidate.version) && !GIT_COMMIT.test(candidate.version)) {
      errors.push(`plugin ${candidate.id} must use an exact version or Git commit`);
    }
    if (!candidate.container_path.startsWith("/opt/pi/plugins/")) {
      errors.push(`plugin ${candidate.id} container path is outside /opt/pi/plugins`);
    }
    const hostPath = path.resolve(root, candidate.host_path);
    if (hostPath !== root && !hostPath.startsWith(`${root}${path.sep}`)) {
      errors.push(`plugin ${candidate.id} host path escapes the lock root`);
      continue;
    }
    const match = SHA256.exec(candidate.integrity);
    if (!match) {
      errors.push(`plugin ${candidate.id} checksum is not sha256`);
      continue;
    }
    try {
      const digest = createHash("sha256").update(await readFile(hostPath)).digest("hex");
      if (digest !== match[1].toLowerCase()) errors.push(`plugin ${candidate.id} checksum mismatch`);
    } catch {
      errors.push(`plugin ${candidate.id} file is missing`);
    }
  }
  return { healthy: errors.length === 0, errors, plugins, imageDigest: lockImageDigest };
}

function isLockedPlugin(value: unknown): value is LockedPlugin {
  if (!isRecord(value)) return false;
  return (
    typeof value.id === "string" && /^[A-Za-z0-9._-]+$/.test(value.id) &&
    typeof value.source === "string" &&
    typeof value.version === "string" &&
    typeof value.integrity === "string" &&
    typeof value.container_path === "string" &&
    typeof value.host_path === "string" &&
    Array.isArray(value.enabled_tools) && value.enabled_tools.every((tool) => typeof tool === "string") &&
    ["read_only", "workspace_write", "privileged"].includes(String(value.risk_class))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
