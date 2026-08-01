import { randomUUID } from "node:crypto";
import { lstat, realpath } from "node:fs/promises";
import path from "node:path";

export type PolicyDecision = "allow" | "allow_after_write_approval" | "approval" | "block";

export interface BashPolicyResult {
  decision: PolicyDecision;
  reason: string;
}

const BLOCKED_COMMAND_PATTERNS: Array<[RegExp, string]> = [
  [/\b(?:curl|wget|nc|ncat|ssh|scp|rsync|ftp)\b/i, "network_command"],
  [/\b(?:sudo|su|doas)\b/i, "privilege_escalation"],
  [/\b(?:docker|podman|kubectl|nerdctl|mount|umount)\b/i, "host_or_container_control"],
  [/\brm\s+(?:-[^\s]*r[^\s]*f?|-rf|-fr)\s+(?:\/|\.|\.\/|\*)\s*(?:$|[;&|])/i, "broad_delete"],
  [/\bgit\s+clean\b/i, "broad_delete"],
  [/\b(?:npm\s+(?:install|i|add|update)|pnpm\s+(?:install|add|update)|yarn\s+(?:install|add|upgrade)|pip3?\s+install|apt(?:-get)?\s+install)\b/i, "runtime_install"],
  [/(?:^|[\s'"/])(?:\.git|\.env(?:\.[^\s/]*)?|\.ssh|\.aws|id_rsa|id_ed25519|credentials\.json)(?:[\s'"/]|$)/i, "protected_path"],
  [/(?:^|\s)(?:~\/|\/Users\/|\/home\/|\/root\/|\/etc\/|\/var\/run\/docker\.sock)/i, "host_path"],
];

const READ_ONLY_GIT = /^git\s+(?:status|diff|log|show|branch(?:\s+--show-current)?|rev-parse)(?:\s|$)/i;

export function classifyBash(command: string, safeCommands: string[]): BashPolicyResult {
  const normalized = command.trim().replace(/\s+/g, " ");
  for (const [pattern, reason] of BLOCKED_COMMAND_PATTERNS) {
    if (pattern.test(normalized)) return { decision: "block", reason };
  }
  if (/[;&|`$<>\n\r]/.test(normalized)) {
    return { decision: "approval", reason: "shell_operator" };
  }
  if (READ_ONLY_GIT.test(normalized)) {
    return { decision: "allow", reason: "read_only_git" };
  }
  if (
    safeCommands.some((safe) => normalized === safe.trim() || normalized.startsWith(`${safe.trim()} `))
  ) {
    return { decision: "allow_after_write_approval", reason: "configured_command" };
  }
  return { decision: "approval", reason: "unlisted_bash" };
}

const PROTECTED_SEGMENTS = new Set([".git", ".ssh", ".aws", ".azure", ".gnupg"]);
const PROTECTED_BASENAMES = [
  /^\.env(?:\..+)?$/i,
  /^(?:id_rsa|id_ed25519|credentials(?:\.json)?|service-account\.json|\.npmrc|\.pypirc|\.netrc)$/i,
  /\.(?:pem|key|p12|pfx)$/i,
];

export async function validateWorkspacePath(
  workspaceRoot: string,
  requestedPath: string,
  operation: "read" | "write",
): Promise<{ allowed: boolean; reason?: string; resolvedPath?: string }> {
  const root = await realpath(workspaceRoot);
  if (!requestedPath || path.isAbsolute(requestedPath)) {
    return { allowed: false, reason: "absolute_or_empty_path" };
  }
  const normalized = path.normalize(requestedPath);
  if (normalized === ".." || normalized.startsWith(`..${path.sep}`)) {
    return { allowed: false, reason: "path_traversal" };
  }
  const slashPath = normalized.split(path.sep).join("/");
  const basename = path.basename(normalized);
  const segments = slashPath.split("/");
  const protectedDirectory = segments.some((segment) => PROTECTED_SEGMENTS.has(segment))
    || slashPath === ".config/gcloud"
    || slashPath.startsWith(".config/gcloud/");
  if (protectedDirectory || PROTECTED_BASENAMES.some((pattern) => pattern.test(basename))) {
    return { allowed: false, reason: "protected_path" };
  }

  const candidate = path.resolve(root, normalized);
  let existing = candidate;
  while (existing !== root) {
    try {
      await lstat(existing);
      break;
    } catch {
      existing = path.dirname(existing);
    }
  }
  const existingReal = await realpath(existing);
  if (existingReal !== root && !existingReal.startsWith(`${root}${path.sep}`)) {
    return { allowed: false, reason: "symlink_escape" };
  }
  const resolved = path.join(existingReal, path.relative(existing, candidate));
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    return { allowed: false, reason: "workspace_escape" };
  }
  return { allowed: true, resolvedPath: resolved };
}

export interface ApprovalRequest {
  id: string;
  runId: string;
  toolCallId: string;
  userId: string;
  action: string;
  expiresAt: number;
  used: boolean;
}

export class ApprovalStore {
  private readonly approvals = new Map<string, ApprovalRequest>();
  private readonly ttlMs: number;
  private readonly now: () => number;

  constructor(options: { ttlMs?: number; now?: () => number } = {}) {
    this.ttlMs = options.ttlMs ?? 300_000;
    this.now = options.now ?? Date.now;
  }

  create(input: Omit<ApprovalRequest, "id" | "expiresAt" | "used">): ApprovalRequest {
    const approval: ApprovalRequest = {
      ...input,
      id: randomUUID(),
      expiresAt: this.now() + this.ttlMs,
      used: false,
    };
    this.approvals.set(approval.id, approval);
    return { ...approval };
  }

  decide(
    id: string,
    userId: string,
    decision: "approve" | "reject",
    runId?: string,
  ): { code: "approved" | "rejected" | "not_found" | "wrong_user" | "wrong_run" | "expired" | "already_used"; approval?: ApprovalRequest } {
    const approval = this.approvals.get(id);
    if (!approval) return { code: "not_found" };
    if (approval.userId !== userId) return { code: "wrong_user" };
    if (runId !== undefined && approval.runId !== runId) return { code: "wrong_run" };
    if (approval.used) return { code: "already_used" };
    if (this.now() > approval.expiresAt) {
      approval.used = true;
      return { code: "expired" };
    }
    approval.used = true;
    return { code: decision === "approve" ? "approved" : "rejected", approval: { ...approval } };
  }

  invalidateRun(runId: string): void {
    for (const approval of this.approvals.values()) {
      if (approval.runId === runId) approval.used = true;
    }
  }
}
