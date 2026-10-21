Repair-driving eval tasks, kept out of `eval/tasks` so the main suite's baseline
(`eval/baseline.json`) keeps measuring first-attempt behaviour.

Each task here sets `requires_repair: true`: a run passes only if it meets its
`expect` **and** went through at least one repair. When the task's profile turns
`writerReuse` on, every repair attempt must also report `startup.reused: true`,
so a cold-started repair fails the run with "repair did not reuse the writer".

`hidden-rule-repair` gets its repair deterministically: the prompt names only the
dash bug, while the `acceptance` check (the writer sees check names, never their
argv) also requires `&` to be spelled out as `and`. `run_checks` is always
refused (`writerChecks.reserveMs` equals the time limit), so the first attempt
fails verification and the failing assertion reaches the repair prompt.

It spends model budget. The first run records its own baseline:

    python scripts/agent_eval.py --live-openai --tasks-dir eval/repair-tasks \
      --baseline eval/repair-baseline.json --repeats 3 --max-cost-usd 2 --update-baseline
