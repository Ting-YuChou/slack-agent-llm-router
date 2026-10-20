import { randomUUID } from "node:crypto";
import { lstat, realpath } from "node:fs/promises";
import path from "node:path";

import { tokenizeShell, type ShellSegment } from "./shell-tokenize.js";

export type PolicyDecision = "allow" | "allow_after_write_approval" | "approval" | "block";

export interface BashPolicyResult {
  decision: PolicyDecision;
  reason: string;
}

const BLOCKED_COMMAND_PATTERNS: Array<[RegExp, string]> = [
  // Best effort only: Pi runs as PID 1, so the gateway's own limits are the real credential boundary.
  [/(?:^|[;&|(]\s*)(?:env|printenv|export|set|declare\s+-x)(?:\s|$)|\/proc\/[^\s/]+\/environ|\$\{?[A-Za-z_]*(?:KEY|TOKEN|SECRET|PASSWORD|_JSON)\b/i, "secret_env_read"],
  [/\b(?:curl|wget|nc|ncat|ssh|scp|rsync|ftp)\b/i, "network_command"],
  [/\b(?:sudo|su|doas)\b/i, "privilege_escalation"],
  [/\b(?:docker|podman|kubectl|nerdctl|mount|umount)\b/i, "host_or_container_control"],
  [/\brm\s+(?:-[^\s]*r[^\s]*f?|-rf|-fr)\s+(?:\/|\.|\.\/|\*)\s*(?:$|[;&|])/i, "broad_delete"],
  [/\bgit\s+clean\b/i, "broad_delete"],
  [/\b(?:npm\s+(?:install|i|add|update)|pnpm\s+(?:install|add|update)|yarn\s+(?:install|add|upgrade)|pip3?\s+install|apt(?:-get)?\s+install)\b/i, "runtime_install"],
  [/(?:^|[\s'"/])(?:\.git|\.env(?:\.[^\s/]*)?|\.ssh|\.aws|id_rsa|id_ed25519|credentials\.json)(?:[\s'"/]|$)/i, "protected_path"],
  [/(?:^|\s)(?:~\/|\/Users\/|\/home\/|\/root\/|\/etc\/|\/var\/run\/docker\.sock)/i, "host_path"],
];

// `git branch` only when it lists: a bare name, -D, -m or --delete would change refs.
const READ_ONLY_GIT = /^git\s+(?:(?:status|diff|log|show|rev-parse)(?:\s|$)|branch(?:\s+(?:--show-current|-a|--all|-r|--remotes|-l|--list|-v|-vv|--verbose))*\s*$)/i;

export function isApprovedSkillRead(requestedPath: string, approvedPaths: string[]): boolean {
  if (!path.posix.isAbsolute(requestedPath) || path.posix.normalize(requestedPath) !== requestedPath) return false;
  return approvedPaths.some((approvedPath) => (
    approvedPath.startsWith("/opt/pi/skills/") &&
    approvedPath.endsWith("/SKILL.md") &&
    approvedPath === requestedPath
  ));
}

const DECISION_RANK: Record<PolicyDecision, number> = { allow: 0, allow_after_write_approval: 1, approval: 2, block: 3 };

function stricter(left: BashPolicyResult, right: BashPolicyResult): BashPolicyResult {
  return !left.reason || DECISION_RANK[right.decision] > DECISION_RANK[left.decision] ? right : left;
}

function legacyBlock(text: string): BashPolicyResult | undefined {
  for (const [pattern, reason] of BLOCKED_COMMAND_PATTERNS) {
    if (pattern.test(text)) return { decision: "block", reason };
  }
  return undefined;
}

const BLOCKED_PROGRAMS = new Map<string, string>([
  ...["env", "printenv", "export", "set", "declare", "typeset"].map((name) => [name, "secret_env_read"] as [string, string]),
  ...["curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "rsync", "ftp", "telnet", "socat"].map((name) => [name, "network_command"] as [string, string]),
  ...["sudo", "su", "doas"].map((name) => [name, "privilege_escalation"] as [string, string]),
  ...["docker", "podman", "kubectl", "nerdctl", "mount", "umount"].map((name) => [name, "host_or_container_control"] as [string, string]),
]);

/** Programs that only read the files they are given and write to stdout. */
const READ_ONLY_PROGRAMS = new Set([
  "cat", "head", "tail", "wc", "ls", "grep", "egrep", "fgrep", "rg", "find", "sed",
  "echo", "printf", "pwd", "cut", "tr", "diff", "stat", "basename", "dirname", "true",
]);
/** Flags that make an interpreter run the next argument as code. */
const INLINE_CODE_FLAGS: Record<string, RegExp> = {
  python: /^-[A-Za-z]*c$/, python3: /^-[A-Za-z]*c$/,
  node: /^(?:-e|-p|--eval|--print)$/, perl: /^-[A-Za-z]*[eE]$/, ruby: /^-[A-Za-z]*e$/, php: /^-r$/,
  sh: /^-[A-Za-z]*c$/, bash: /^-[A-Za-z]*c$/, zsh: /^-[A-Za-z]*c$/, dash: /^-[A-Za-z]*c$/, ksh: /^-[A-Za-z]*c$/,
};
const SHELLS = new Set(["sh", "bash", "zsh", "dash", "ksh"]);
const INLINE_NETWORK = /\b(?:urllib|requests|httpx|aiohttp|http\.client|socket|websocket|fetch|XMLHttpRequest|child_process|subprocess|os\.system|LWP|Net::HTTP|open-uri)\b|https?:\/\//i;
const ALLOWED_TEMPLATE_ENV = /^\.env\.(?:example|sample|template)$/i;
const FIND_ACTIONS = /^-(?:exec|execdir|ok|okdir|delete|fprint|fprint0|fprintf|fls)$/;

/** Strips `timeout`, `nice`, `nohup`, `xargs`, `command` and `exec` and reports whether xargs fed arguments. */
function unwrap(argv: string[]): { argv: string[]; viaXargs: boolean; query?: boolean } {
  let rest = argv;
  let viaXargs = false;
  for (;;) {
    const name = path.posix.basename(rest[0] ?? "");
    if (name === "timeout") {
      let index = 1;
      while (rest[index]?.startsWith("-")) index += ["-s", "--signal", "-k", "--kill-after"].includes(rest[index]!) ? 2 : 1;
      rest = rest.slice(index + 1);
    } else if (name === "nice") {
      let index = 1;
      while (rest[index]?.startsWith("-")) index += ["-n", "--adjustment"].includes(rest[index]!) ? 2 : 1;
      rest = rest.slice(index);
    } else if (name === "nohup") {
      rest = rest.slice(rest[1] === "--" ? 2 : 1);
    } else if (name === "xargs") {
      viaXargs = true;
      let index = 1;
      while (rest[index]?.startsWith("-") && rest[index] !== "--") {
        index += /^-[InLPdsaE]$/.test(rest[index]!) ? 2 : 1;
      }
      if (rest[index] === "--") index += 1;
      rest = rest.slice(index);
      if (!rest.length) rest = ["echo"];
    } else if (name === "command") {
      if (rest[1] === "-v" || rest[1] === "-V") return { argv: rest, viaXargs, query: true };
      rest = rest.slice(rest[1] === "-p" ? 2 : 1);
    } else if (name === "exec") {
      let index = 1;
      while (rest[index]?.startsWith("-")) index += rest[index] === "-a" ? 2 : 1;
      rest = rest.slice(index);
    } else {
      return { argv: rest, viaXargs };
    }
  }
}

/** Path checks for arguments of programs that are otherwise allowed to read freely. */
function readArgumentPolicy(argument: string): BashPolicyResult | undefined {
  if (/\/proc\/[^/]+\/environ/.test(argument)) return { decision: "block", reason: "secret_env_read" };
  if (/^~|(?:^|[\s:=])(?:\/Users\/|\/home\/|\/root\/|\/etc\/|\/var\/run\/docker\.sock)/.test(argument)) {
    return { decision: "block", reason: "host_path" };
  }
  for (const piece of argument.split(/[:=,]/)) {
    const segments = piece.split("/").filter(Boolean);
    const protectedPiece = segments.some((segment) => PROTECTED_SEGMENTS.has(segment))
      || /(?:^|\/)\.config\/gcloud(?:\/|$)/.test(piece)
      || segments.some((segment) => !ALLOWED_TEMPLATE_ENV.test(segment) && PROTECTED_BASENAMES.some((pattern) => pattern.test(segment)));
    if (protectedPiece) return { decision: "block", reason: "protected_path" };
  }
  if (argument.startsWith("/") && argument !== "/dev/null") return { decision: "approval", reason: "absolute_path" };
  if (argument.split(/[/:=,]/).includes("..")) return { decision: "approval", reason: "path_traversal" };
  return undefined;
}

function readOnlyProgramPolicy(name: string, args: string[]): BashPolicyResult {
  if (name === "find" && args.some((arg) => FIND_ACTIONS.test(arg))) return { decision: "approval", reason: "find_action" };
  if (name === "rg" && args.some((arg) => arg === "--pre" || arg.startsWith("--pre="))) return { decision: "approval", reason: "rg_preprocessor" };
  if (name === "sed") {
    if (args.some((arg) => /^-[^-]*i/.test(arg) || arg.startsWith("--in-place"))) return { decision: "approval", reason: "in_place_edit" };
    const options = args.filter((arg) => arg.startsWith("-"));
    const operands = args.filter((arg) => !arg.startsWith("-"));
    // Only quiet print scripts such as `1,40p` or `/re/p`; anything else may write or execute.
    const printOnly = /^(?:(?:\d+|\$)(?:,(?:\d+|\$|\+\d+))?|\/[^/]*\/(?:,\/[^/]*\/)?)p$/;
    if (!options.length || options.some((option) => option !== "-n") || !operands.length || !printOnly.test(operands[0]!)) {
      return { decision: "approval", reason: "sed_script" };
    }
  }
  return { decision: "allow", reason: "read_only_command" };
}

function classifySegment(segment: ShellSegment, safeCommands: string[], depth: number): BashPolicyResult {
  // An empty reason marks "no restriction yet"; the segment's own verdict then supplies it.
  let result: BashPolicyResult = { decision: "allow", reason: "" };
  for (const redirect of segment.redirects) {
    if (redirect.kind === "dup") continue;
    const checked = readArgumentPolicy(redirect.target);
    if (checked?.decision === "block") return checked;
    if (redirect.kind === "write" && redirect.target !== "/dev/null") result = stricter(result, { decision: "approval", reason: "write_redirect" });
    else if (checked) result = stricter(result, checked);
  }

  let argv = segment.argv;
  const assignments = argv.findIndex((word) => !/^[A-Za-z_][A-Za-z0-9_]*=/.test(word));
  if (assignments !== 0) {
    if (assignments < 0) return stricter(result, { decision: "approval", reason: "env_assignment" });
    argv = argv.slice(assignments);
    result = stricter(result, { decision: "approval", reason: "env_assignment" });
  }
  const unwrapped = unwrap(argv);
  argv = unwrapped.argv;
  const text = argv.join(" ");
  const name = path.posix.basename(argv[0] ?? "");
  const args = argv.slice(1);

  const blocked = BLOCKED_PROGRAMS.get(name);
  if (blocked) return { decision: "block", reason: blocked };
  if (unwrapped.query) return stricter(result, { decision: "allow", reason: "command_lookup" });

  const inlineFlag = INLINE_CODE_FLAGS[name.replace(/^(python3)\.\d+$/, "$1")];
  if (inlineFlag) {
    const index = args.findIndex((arg) => inlineFlag.test(arg));
    if (index >= 0) {
      const code = args[index + 1] ?? "";
      if (INLINE_NETWORK.test(code)) return { decision: "block", reason: "network_command" };
      const legacy = legacyBlock(code.replace(/\s+/g, " "));
      if (legacy) return legacy;
      if (SHELLS.has(name) && depth < 3) {
        const inner = classifyBash(code, safeCommands, depth + 1);
        if (inner.decision === "block") return inner;
      }
      return stricter(result, { decision: "approval", reason: "inline_code" });
    }
  }

  if (READ_ONLY_PROGRAMS.has(name) && argv[0] === name) {
    for (const arg of args) {
      const checked = readArgumentPolicy(arg);
      if (checked?.decision === "block") return checked;
      if (checked) result = stricter(result, checked);
    }
    result = stricter(result, readOnlyProgramPolicy(name, args));
    if (segment.unquotedGlob) result = stricter(result, { decision: "approval", reason: "unquoted_glob" });
    if (unwrapped.viaXargs) result = stricter(result, { decision: "approval", reason: "xargs_arguments" });
    return result;
  }

  const legacy = legacyBlock(text);
  if (legacy) return legacy;
  if (READ_ONLY_GIT.test(text) && argv[0] === "git" && !args.some((arg) => arg.startsWith("--output"))) {
    for (const arg of args) {
      const checked = readArgumentPolicy(arg);
      if (checked?.decision === "block") return checked;
    }
    return stricter(result, { decision: "allow", reason: "read_only_git" });
  }
  if (safeCommands.some((safe) => text === safe.trim() || text.startsWith(`${safe.trim()} `))) {
    return stricter(result, { decision: "allow_after_write_approval", reason: "configured_command" });
  }
  return stricter(result, { decision: "approval", reason: "unlisted_bash" });
}

/**
 * Classifies a shell command segment by segment; the most restrictive segment decides.
 * Commands the tokenizer refuses fall back to the legacy patterns, which can only block or ask.
 */
export function classifyBash(command: string, safeCommands: string[], depth = 0): BashPolicyResult {
  const normalized = command.trim().replace(/\s+/g, " ");
  const parsed = tokenizeShell(command);
  if (!parsed.ok) {
    return legacyBlock(normalized) ?? { decision: "approval", reason: "shell_operator" };
  }
  let result: BashPolicyResult | undefined;
  for (const segment of parsed.segments) {
    const decision = classifySegment(segment, safeCommands, depth);
    result = result ? stricter(result, decision) : decision;
    if (result.decision === "block") return result;
  }
  return result!;
}

const PROTECTED_SEGMENTS = new Set([".git", ".ssh", ".aws", ".azure", ".gnupg"]);
const PROTECTED_BASENAMES = [
  /^\.env(?:\..+)?$/i,
  /^(?:id_rsa|id_ed25519|credentials(?:\.json)?|service-account\.json|\.npmrc|\.pypirc|\.netrc)$/i,
  /\.(?:pem|key|p12|pfx)$/i,
];

const TEST_CONFIG_BASENAMES = /^(?:package\.json|conftest\.py|pytest\.ini|pyproject\.toml|setup\.cfg|setup\.py|tox\.ini|noxfile\.py|sitecustomize\.py|usercustomize\.py|Makefile|\.npmrc|tsconfig.*\.json|.*\.pth)$/;

/** Files that change what a configured test command executes, at any depth. */
export function isTestConfigPath(filePath: string): boolean {
  return TEST_CONFIG_BASENAMES.test(path.posix.basename(filePath.replaceAll("\\", "/")));
}

/** Parses `git status --porcelain -z`; a rename or copy entry is followed by its source path. */
export function parsePorcelainZ(output: string): string[] {
  const fields = output.split("\0").filter(Boolean);
  const paths: string[] = [];
  for (let index = 0; index < fields.length; index += 1) {
    const field = fields[index]!;
    paths.push(field.slice(3));
    if (field[0] === "R" || field[0] === "C") {
      const source = fields[index + 1];
      if (source !== undefined) paths.push(source);
      index += 1;
    }
  }
  return paths;
}

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

/** Snapshot tools run from an empty session cwd and need absolute mount paths. */
export async function validateSnapshotRead(root: string, cwd: string, requested?: string, scopes: string[] = ['.']) {
  const absolute = path.resolve(cwd,requested ?? '.');
  const relative = path.relative(path.resolve(root),absolute) || '.';
  const checked = await validateWorkspacePath(root,relative,'read');
  if (!checked.allowed) return checked;
  const canonicalRoot = await realpath(root);
  if (!scopes.some(scope => { const allowed = path.resolve(canonicalRoot,scope); return checked.resolvedPath === allowed || checked.resolvedPath?.startsWith(allowed + path.sep); })) return { allowed: false,reason: 'task_scope' };
  return checked;
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

  create(input: Omit<ApprovalRequest, "id" | "expiresAt" | "used">, ttlMs = this.ttlMs): ApprovalRequest {
    const approval: ApprovalRequest = {
      ...input,
      id: randomUUID(),
      expiresAt: this.now() + ttlMs,
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
