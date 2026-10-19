import assert from "node:assert/strict";
import { test } from "node:test";

import { assertCandidateScope, filesOutsideScope, ScopeViolationError } from "../src/candidate-scope.js";

test("files outside allowed roots are listed; '.' allows everything", () => {
  assert.deepEqual(filesOutsideScope(["src/a.ts", "src", "srcx/b.ts", "docs/c.md"], ["src/"]), ["srcx/b.ts", "docs/c.md"]);
  assert.deepEqual(filesOutsideScope(["anything/at/all"], ["."]), []);
});

test("a scope violation names the violating files so the writer can repair them", () => {
  assert.doesNotThrow(() => assertCandidateScope(["src/a.ts"], ["src"]));
  assert.throws(() => assertCandidateScope(["src/a.ts", "docs/c.md", ".github/x.yml"], ["src"]),
    (error: unknown) => error instanceof ScopeViolationError && JSON.stringify(error.files) === JSON.stringify(["docs/c.md", ".github/x.yml"]));
});
