import path from "node:path";

import { redact } from "./telemetry.js";
import type { VerificationCheck, VerificationResult } from "./quality-gate.js";

/** One diagnostic a trusted check reported, with its path relative to the repository root. */
export interface CheckFailure { file?: string; line?: number; column?: number; code?: string; message: string }

const WORKSPACE = "/workspace/check/";
const MAX_FAILURES = 50;
const MAX_MESSAGE = 500;
const EXCERPT_BYTES = 8 * 1024;
const ANSI = /\u001b\[[0-9;?]*[ -/]*[@-~]|\u001b[@-Z\\-_]/g;

export function stripAnsi(text: string): string { return text.replace(ANSI, ""); }

/** Maps a reported path to a repository-relative one; anything that escapes the workspace is dropped. */
function repoPath(file: string, cwd: string): string | undefined {
  let value = file.trim();
  if (value.startsWith(WORKSPACE)) value = value.slice(WORKSPACE.length);
  else if (path.posix.isAbsolute(value)) return undefined;
  else value = path.posix.join(cwd, value);
  value = path.posix.normalize(value);
  if (!value || value === "." || value.startsWith("../") || value === ".." || path.posix.isAbsolute(value) || /[\x00-\x1f]/.test(value)) return undefined;
  return value;
}

type Parser = (line: string, next: () => string | undefined) => (Omit<CheckFailure, "file"> & { file: string }) | null;

const parsers: Parser[] = [
  // tsc: src/a.ts(12,5): error TS2322: message
  line => { const m = /^(.+?)\((\d+),(\d+)\): error (TS\d+): (.+)$/.exec(line); return m ? { file: m[1], line: +m[2], column: +m[3], code: m[4], message: m[5] } : null; },
  // tsc --pretty: src/a.ts:12:5 - error TS2322: message
  line => { const m = /^(.+?):(\d+):(\d+) - error (TS\d+): (.+)$/.exec(line); return m ? { file: m[1], line: +m[2], column: +m[3], code: m[4], message: m[5] } : null; },
  // mypy: src/a.py:7: error: message  [code]   (column optional)
  line => {
    const m = /^(.+?\.pyi?):(\d+)(?::(\d+))?: error: (.+?)(?:\s+\[([\w-]+)\])?$/.exec(line);
    return m ? { file: m[1], line: +m[2], ...(m[3] ? { column: +m[3] } : {}), ...(m[5] ? { code: m[5] } : {}), message: m[4] } : null;
  },
  // flake8: src/a.py:12:80: E501 message
  line => { const m = /^(.+?):(\d+):(\d+): ([A-Z]+\d+) (.+)$/.exec(line); return m ? { file: m[1], line: +m[2], column: +m[3], code: m[4], message: m[5] } : null; },
  // black
  line => { const m = /^would reformat (.+)$/.exec(line); return m ? { file: m[1], message: "would reformat" } : null; },
  line => {
    const m = /^error: cannot format (.+?): (.+)$/.exec(line); if (!m) return null;
    const at = /Cannot parse: (\d+):(\d+)/.exec(m[2]);
    return { file: m[1], ...(at ? { line: +at[1], column: +at[2] } : {}), message: m[2] };
  },
  // compileall: *** Error compiling '<file>'... then a SyntaxError block
  (line, next) => {
    const m = /^\*\*\* Error compiling '(.+)'\.\.\.$/.exec(line); if (!m) return null;
    let lineNo: number | undefined;
    for (let i = 0, item = next(); item !== undefined && i < 8; i++, item = next()) {
      const at = /^\s*File ".*", line (\d+)/.exec(item); if (at) lineNo = +at[1];
      const error = /^(\w*Error): (.+)$/.exec(item);
      if (error) return { file: m[1], ...(lineNo ? { line: lineNo } : {}), code: error[1], message: error[2] };
    }
    return { file: m[1], ...(lineNo ? { line: lineNo } : {}), message: "compile error" };
  },
];

/** Extracts file/line diagnostics from tsc, flake8, mypy, black and compileall output. */
export function parseDiagnostics(output: string, cwd = "."): CheckFailure[] {
  const lines = stripAnsi(output).split(/\r?\n/);
  const failures: CheckFailure[] = [];
  for (let index = 0; index < lines.length && failures.length < MAX_FAILURES; index++) {
    const line = lines[index]; let cursor = index;
    const next = () => lines[++cursor];
    for (const parse of parsers) {
      const found = parse(line, next);
      if (!found) { cursor = index; continue; }
      index = cursor;
      const file = repoPath(found.file, cwd);
      if (file) failures.push({ file, ...(found.line ? { line: found.line } : {}), ...(found.column ? { column: found.column } : {}), ...(found.code ? { code: found.code } : {}), message: String(redact(found.message)).slice(0, MAX_MESSAGE) });
      break;
    }
  }
  return failures;
}

/** The redacted, ANSI-free tail of a check log, cut on line boundaries. */
export function excerpt(output: string, maxBytes = EXCERPT_BYTES): string {
  const lines = String(redact(stripAnsi(output))).replace(/\s+$/, "").split(/\r?\n/);
  const kept: string[] = []; let bytes = 0;
  for (let index = lines.length - 1; index >= 0; index--) {
    const size = Buffer.byteLength(lines[index]) + (kept.length ? 1 : 0);
    if (bytes + size > maxBytes) break;
    kept.unshift(lines[index]); bytes += size;
  }
  return kept.join("\n");
}

function location(failure: CheckFailure): string {
  return [failure.file, failure.line, failure.column].filter(value => value !== undefined).join(":");
}

function heading(check: VerificationCheck): string {
  const detail = [check.timed_out ? "timed out" : check.exit_code !== null ? `exit ${check.exit_code}` : "no exit code",
    typeof check.failed === "number" ? `${check.failed} failed` : "", check.cause ? `cause ${check.cause}` : ""].filter(Boolean).join(", ");
  return `## ${check.name}: ${check.status} (${detail})`;
}

/**
 * Bounded writer-facing feedback for checks that did not pass. Host artifact paths never
 * appear; whatever does not fit is counted in a trailing "N more omitted" line.
 */
export function formatCheckFeedback(verification: VerificationResult, options: { maxBytes?: number } = {}): string {
  const maxBytes = options.maxBytes ?? 24 * 1024;
  const passed = verification.checks.filter(check => check.status === "passed").map(check => check.name);
  const blocks: string[][] = [];
  for (const check of verification.checks.filter(item => item.status !== "passed")) {
    const block = [heading(check)];
    for (const failure of check.failures ?? []) block.push(`- ${[location(failure), failure.code, failure.message].filter(Boolean).join(" ")}`);
    if (check.excerpt) block.push("Log tail:", check.excerpt);
    blocks.push(block);
  }
  const out: string[] = [`Verification: ${verification.status}`];
  if (passed.length) out.push(`Passed: ${passed.join(", ")}`);
  const reserve = 64; let bytes = Buffer.byteLength(out.join("\n")); let omitted = 0;
  const add = (text: string) => {
    const size = Buffer.byteLength(text) + 1;
    if (bytes + size > maxBytes - reserve) return false;
    out.push(text); bytes += size; return true;
  };
  for (const block of blocks) {
    if (!add(block[0])) { omitted += block.length; continue; }
    for (const [index, line] of block.slice(1).entries()) {
      if (add(line)) continue;
      // A log tail that does not fit is shortened to the space left rather than dropped whole.
      const room = maxBytes - reserve - bytes - 1;
      if (line === block.at(-1) && block[index] === "Log tail:" && room > 256) { const tail = excerpt(line, room); if (tail && add(tail)) continue; }
      omitted += 1;
    }
  }
  if (omitted) out.push(`${omitted} more omitted`);
  return out.join("\n");
}
