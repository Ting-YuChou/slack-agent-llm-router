import { lstat, readFile } from "node:fs/promises";
import path from "node:path";

import type { Grade } from "./quality-gate.js";

const MAX_CITED_FILE_BYTES = 4 * 1024 * 1024;
// A relative path with a directory part or an extension (or a well-known bare name), optionally `:line`.
const CITATION = /(?<![\w./-])((?:[\w.-]+\/)*[\w-][\w.-]*)(?::(\d+))?/g;
const BARE_NAMES = new Set(["Makefile", "Dockerfile", "Procfile", "Gemfile", "Rakefile", "LICENSE"]);

/** Files the candidate deletes, from its `git diff` patch; graders may cite them although the snapshot lacks them. */
export function deletedPaths(patch: string | undefined): string[] {
  if (!patch) return [];
  const deleted: string[] = [];
  const lines = patch.split("\n");
  for (let i = 0; i < lines.length - 1; i++) {
    const header = /^diff --git a\/(.+) b\/\1$/.exec(lines[i]!);
    if (header && lines[i + 1]!.startsWith("deleted file mode")) deleted.push(header[1]!);
  }
  return deleted.slice(0, 500);
}

/** Line count of a regular file inside the snapshot, or undefined when it is missing, a symlink, or outside the root. */
async function lineCount(root: string, relative: string, cache: Map<string, number | undefined>): Promise<number | undefined> {
  if (cache.has(relative)) return cache.get(relative);
  let count: number | undefined;
  const target = path.resolve(root, relative);
  if (!path.isAbsolute(relative) && target.startsWith(path.resolve(root) + path.sep)) {
    try {
      const info = await lstat(target);
      if (info.isFile() && info.size <= MAX_CITED_FILE_BYTES) {
        const text = await readFile(target, "utf8");
        count = text.length === 0 ? 0 : text.split("\n").length - (text.endsWith("\n") ? 1 : 0);
      }
    } catch { /* missing files are simply not evidence */ }
  }
  cache.set(relative, count);
  return count;
}

function citations(text: string): Array<{ path: string; line?: number }> {
  const found: Array<{ path: string; line?: number }> = [];
  for (const match of text.matchAll(CITATION)) {
    const cited = match[1]!.replace(/[.]+$/, "");
    if (!cited.includes("/") && !/\.\w+$/.test(cited) && !BARE_NAMES.has(cited)) continue;
    found.push({ path: cited, line: match[2] ? Number(match[2]) : undefined });
  }
  return found.slice(0, 50);
}

/** Rejects a verdict whose citations do not exist in the snapshot, so a grader cannot pass work on invented evidence. */
export async function verifyGradeEvidence(grade: Grade, snapshotRoot: string, deleted: string[] = []): Promise<void> {
  const removed = new Set(deleted);
  const cache = new Map<string, number | undefined>();
  const exists = async (cited: string, line?: number): Promise<"ok" | "missing" | "out_of_range"> => {
    if (removed.has(cited)) return "ok";
    const lines = await lineCount(snapshotRoot, cited, cache);
    if (lines === undefined) return "missing";
    return line === undefined || (line >= 1 && line <= lines) ? "ok" : "out_of_range";
  };
  for (const finding of grade.findings) {
    const state = await exists(finding.path, finding.line);
    if (state !== "ok") throw new Error(`Grader evidence ${finding.path}:${finding.line} does not exist in the candidate snapshot`);
  }
  for (const criterion of grade.criteria) {
    if (!criterion.passed) continue;
    let grounded = false;
    for (const cited of citations(criterion.evidence ?? "")) {
      const state = await exists(cited.path, cited.line);
      if (state === "out_of_range") throw new Error(`Grader evidence ${cited.path}:${cited.line} for criterion ${criterion.id} is outside the file`);
      if (state === "ok") grounded = true;
    }
    if (!grounded) throw new Error(`Grader evidence for passed criterion ${criterion.id} must cite a file in the candidate snapshot (path or path:line)`);
  }
}
