import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { validateProfile } from "../src/verification.js";

const repoFile = (relative: string) => fileURLToPath(new URL(`../../../${relative}`, import.meta.url));
const readJson = (relative: string) => JSON.parse(readFileSync(repoFile(relative), "utf8"));

test("the example verification profile passes validateProfile", () => {
  const profile = validateProfile(readJson("config/verification/slack-router.example.json"));

  assert.equal(profile.id, "slack-router-v1");
});

test("the example profile protects AGENTS.md as a baseline path", () => {
  const profile = validateProfile(readJson("config/verification/slack-router.example.json"));

  assert.ok(profile.baselinePaths.includes("AGENTS.md"), "AGENTS.md must be a baseline path");
});

test("every eval task profile passes validateProfile once given an image", () => {
  const tasks = readdirSync(repoFile("eval/tasks"));
  assert.ok(tasks.length > 0);
  for (const id of tasks) {
    const task = readJson(`eval/tasks/${id}/task.json`);
    const profile = { id: `eval-${id}`, repository: `fixture/${id}`, image: `sha256:${"0".repeat(64)}`, ...task.profile };
    assert.doesNotThrow(() => validateProfile(profile), id);
  }
});
