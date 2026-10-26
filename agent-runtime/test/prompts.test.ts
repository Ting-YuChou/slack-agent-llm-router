import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";

import { PROMPT_IDS, PROMPTS_VERSION, ROLE_PROMPTS, promptRef, promptTelemetry, renderPrompt, type PromptVars } from "../src/prompts.js";

const root = path.resolve(import.meta.dirname, "../..");
const goldenDir = path.join(root, "test/golden/prompts");
const fixtures = JSON.parse(readFileSync(path.join(goldenDir, "fixtures.json"), "utf8")) as Record<string, PromptVars>;

interface PromptGenerator { generatePromptsModule(promptsDir: string): string }
const generator = async () => await import(pathToFileURL(path.join(root, "scripts/build-prompts.mjs")).href) as PromptGenerator;

test("every prompt renders byte-identical to the golden captured before the move", () => {
  assert.deepEqual([...PROMPT_IDS].sort(), Object.keys(fixtures).sort());
  for (const id of PROMPT_IDS) {
    assert.equal(renderPrompt(id, fixtures[id]), readFileSync(path.join(goldenDir, `${id}.txt`), "utf8"), id);
  }
});

test("each prompt has a markdown source and a golden", () => {
  const sources = readdirSync(path.join(root, "prompts")).filter(name => name.endsWith(".md")).map(name => name.slice(0, -3)).sort();
  assert.deepEqual(sources, [...PROMPT_IDS].sort());
  const goldens = readdirSync(goldenDir).filter(name => name.endsWith(".txt")).map(name => name.slice(0, -4)).sort();
  assert.deepEqual(goldens, sources);
});

test("prompts.generated.ts is in sync with prompts/", async () => {
  const { generatePromptsModule } = await generator();
  assert.equal(readFileSync(path.join(root, "src/prompts.generated.ts"), "utf8"), generatePromptsModule(path.join(root, "prompts")),
    "run npm --prefix agent-runtime run build-prompts");
});

test("prompt refs carry a 12-character content hash and the version covers every prompt", () => {
  assert.match(PROMPTS_VERSION, /^[0-9a-f]{12}$/);
  for (const id of PROMPT_IDS) assert.match(promptRef(id), new RegExp(`^${id}@[0-9a-f]{12}$`));
  assert.notEqual(promptRef("investigator"), promptRef("planner"));
});

test("renderPrompt rejects missing, unused and unknown variables and unknown ids", () => {
  const vars = fixtures.discussion!;
  assert.throws(() => renderPrompt("discussion", {}), /discussion.*missing.*root/);
  assert.throws(() => renderPrompt("discussion", { ...vars, extra: "x" }), /discussion.*unused.*extra/);
  assert.throws(() => renderPrompt("jev-classifier", { root: "/x" }), /unused.*root/);
  assert.throws(() => renderPrompt("no-such-prompt" as never, {}), /Unknown prompt/);
});

test("substituted values are inserted verbatim and never re-expanded", () => {
  assert.equal(renderPrompt("discussion", { root: "{{root}} $& $1" }).includes("evidence in {{root}} $& $1."), true);
  assert.match(renderPrompt("ci-repair", fixtures["ci-repair"]!), /\{\{not_a_var\}\}$/);
});

test("the generator rejects malformed placeholders", async () => {
  const { generatePromptsModule } = await generator();
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const dir = mkdtempSync(path.join(tmpdir(), "prompts-"));
  writeFileSync(path.join(dir, "bad.md"), "Hello {{ name }}\n");
  assert.throws(() => generatePromptsModule(dir), /bad.*placeholder/);
});

test("telemetry names the prompt version and the prompts each role runs with", () => {
  const refs = (...ids: Parameters<typeof promptRef>[0][]) => ({ prompts_version: PROMPTS_VERSION, prompts: ids.map(promptRef) });
  assert.deepEqual(promptTelemetry("grader"), refs("grader", "grader-scope"));
  assert.deepEqual(promptTelemetry("planner"), refs("planner", "repo-guidance"));
  for (const role of ["code_explorer", "ci_investigator", "test_analyst"]) assert.deepEqual(promptTelemetry(role), refs("investigator", "repo-guidance"));
  assert.deepEqual(promptTelemetry("discussion"), refs("discussion"));
  const writer = promptTelemetry("implementation").prompts;
  for (const id of ["policy-scope", "checks", "repair", "scope-repair", "budget-notice", "budget-final"] as const) assert.ok(writer.includes(promptRef(id)), id);
  assert.deepEqual(promptTelemetry("no-such-role"), { prompts_version: PROMPTS_VERSION, prompts: [] });
});

test("AGENTS.md guidance goes to the planner and investigators, never to the grader or the writer", () => {
  for (const role of ["planner", "code_explorer", "ci_investigator", "test_analyst"]) assert.ok(ROLE_PROMPTS[role]!.includes("repo-guidance"), role);
  for (const role of ["grader", "implementation", "discussion"]) assert.ok(!ROLE_PROMPTS[role]!.includes("repo-guidance"), role);
});

test("the writer prompt set includes the diff-limit repair prompt", () => {
  assert.ok(ROLE_PROMPTS.implementation!.includes("diff-limit-repair"));
  assert.ok(promptTelemetry("implementation").prompts.includes(promptRef("diff-limit-repair")));
});

test("every prompt rendered in the runtime or shipped under prompts/ belongs to some role", () => {
  const sources = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap(entry =>
    entry.isDirectory() ? sources(path.join(dir, entry.name)) : entry.name.endsWith(".ts") ? [path.join(dir, entry.name)] : []);
  const rendered = new Set<string>();
  for (const file of [...sources(path.join(root, "src")), ...sources(path.join(root, "extensions"))]) {
    for (const call of readFileSync(file, "utf8").matchAll(/renderPrompt\(([^,)]*)/g)) {
      for (const literal of call[1]!.matchAll(/["']([a-z][a-z0-9-]*)["']/g)) rendered.add(literal[1]!);
    }
  }
  for (const id of ["diff-limit-repair", "planner", "investigator", "policy-scope", "repo-guidance"]) assert.ok(rendered.has(id), `scan missed ${id}`);
  for (const id of rendered) assert.ok((PROMPT_IDS as string[]).includes(id), `${id} is not a prompt id`);
  const shipped = readdirSync(path.join(root, "prompts")).filter(name => name.endsWith(".md")).map(name => name.slice(0, -3));
  const assigned = new Set<string>(Object.values(ROLE_PROMPTS).flat());
  const unassigned = [...new Set([...rendered, ...shipped])].filter(id => !assigned.has(id)).sort();
  assert.deepEqual(unassigned, []);
});
