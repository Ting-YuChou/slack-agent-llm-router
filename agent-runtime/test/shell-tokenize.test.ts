import assert from "node:assert/strict";
import { test } from "node:test";

import { tokenizeShell } from "../src/shell-tokenize.js";

test("the tokenizer splits lists and pipelines and keeps quoted text literal", () => {
  const parsed = tokenizeShell("grep -rn 'a b' src | wc -l && echo \"x\\\"y\" ; ls\nfind . -name '*.ts'");
  assert.equal(parsed.ok, true);
  if (!parsed.ok) return;
  assert.deepEqual(parsed.segments.map((segment) => segment.argv), [
    ["grep", "-rn", "a b", "src"],
    ["wc", "-l"],
    ["echo", "x\"y"],
    ["ls"],
    ["find", ".", "-name", "*.ts"],
  ]);
  assert.equal(parsed.segments[4]!.unquotedGlob, false);
});

test("the tokenizer records redirects and unquoted globs", () => {
  const parsed = tokenizeShell("pytest -x 2>&1 >out.txt < in.txt; ls *.ts &> /dev/null");
  assert.equal(parsed.ok, true);
  if (!parsed.ok) return;
  assert.deepEqual(parsed.segments[0]!.redirects, [
    { kind: "dup", target: "1" },
    { kind: "write", target: "out.txt" },
    { kind: "read", target: "in.txt" },
  ]);
  assert.deepEqual(parsed.segments[1]!.redirects, [{ kind: "write", target: "/dev/null" }]);
  assert.equal(parsed.segments[1]!.unquotedGlob, true);
});

test("the tokenizer refuses expansion, substitution, groups, heredocs and background jobs", () => {
  for (const command of [
    "echo $HOME", "echo \"$HOME\"", "echo `id`", "echo $(id)", "diff <(ls) b", "cat <<EOF", "cat <<< x",
    "(ls)", "{ ls; }", "sleep 1 &", "ls |& cat", "echo 'open", "ls &&", "| ls", "ls && ; ls",
  ]) {
    assert.equal(tokenizeShell(command).ok, false, command);
  }
  assert.deepEqual(tokenizeShell("echo '$HOME'"), { ok: true, segments: [{ argv: ["echo", "$HOME"], redirects: [], unquotedGlob: false }] });
});
