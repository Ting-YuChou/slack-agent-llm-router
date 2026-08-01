---
name: test-gap
description: Find one meaningful automated-test gap, add a regression test, and make the smallest safe implementation change needed to pass it. Use for requests to inspect test coverage, close an edge-case gap, or improve confidence in an existing behavior.
---

# Test Gap

Close one concrete coverage gap without broad refactoring or speculative changes.

## Workflow

1. Read repository guidance and the smallest relevant source and test files.
2. Identify one behavior that is important, currently untested, and supported by the existing contract.
3. Explain the chosen gap briefly before editing.
4. Add a focused regression test first and run it to confirm it fails for the expected reason.
5. If the test exposes a defect, make the smallest implementation change that satisfies the established contract. If no production change is needed, do not invent one.
6. Run the focused test, then the nearest practical broader test suite.
7. Summarize the gap, changed files, test results, and any remaining risk.

## Guardrails

- Stay inside the mounted workspace and obey all tool approvals and blocked-path rules.
- Do not install dependencies, access the network, alter credentials, or modify protected files.
- Do not weaken assertions merely to make a test pass.
- Avoid unrelated formatting, cleanup, or refactors.
- Stop and report clearly if the intended behavior is ambiguous or the required command is blocked.
