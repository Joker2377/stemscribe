Reply in traditional chinese.

Keep code decouple, split things into readable functions.

Write comment for each function.

## Conventions
- Each stage is a standalone script run in its own venv (`envs/<name>`); stages share code only by plain `from <module> import ...` inside `pipeline/`. Keep pure logic (no file / model I/O) in functions so it can be tested.
- Tests go in `tests/test_<module>.py` as plain `test_*` functions with `assert` (pytest, no fixtures). New non-trivial logic gets at least one test.
- Before committing, run and make pass: `envs/tools/Scripts/ruff.exe check .` and `envs/tools/Scripts/python.exe -m pytest`. If training code changed, also `envs/separator/Scripts/python.exe -m pytest`.
- Lint config lives in `pyproject.toml` (line length 150). Don't add `# noqa` without a reason in the comment.
- CI (`.github/workflows/ci.yml`) runs the same lint and tests on push to main and on PRs.
- Commit messages: one short imperative summary line; body only when the why isn't obvious.
