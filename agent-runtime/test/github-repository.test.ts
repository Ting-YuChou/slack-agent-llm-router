import assert from "node:assert/strict";
import { test } from "node:test";

import { parseGitHubRepositories, repositoryFromRemote } from "../src/github-repository.js";

test("GitHub repository allowlist is normalized and rejects duplicates or invalid entries", () => {
  assert.deepEqual(parseGitHubRepositories("Acme/Widgets,octo/repo-2"), ["Acme/Widgets", "octo/repo-2"]);
  assert.throws(() => parseGitHubRepositories("acme/widgets,ACME/WIDGETS"), /duplicate/);
  assert.throws(() => parseGitHubRepositories("https://github.com/acme/widgets"), /owner\/repository/);
});

test("GitHub HTTPS and SSH remotes resolve to repository slugs", () => {
  assert.equal(repositoryFromRemote("https://github.com/acme/widgets.git"), "acme/widgets");
  assert.equal(repositoryFromRemote("git@github.com:acme/widgets.git"), "acme/widgets");
  assert.equal(repositoryFromRemote("ssh://git@github.com/acme/widgets.git"), "acme/widgets");
  assert.equal(repositoryFromRemote("https://example.com/acme/widgets.git"), null);
});
