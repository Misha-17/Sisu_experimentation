# Validation of this build

This build changes the earlier V10 export in four ways: Gemma 4 replaces gpt-oss as the default generator, thinking can be turned on with matching budgets, ablation switches were added, and the evaluation pipeline (`sisu_eval`) was added.

## Checked

- **161 tests passed** on Linux with Python 3.13, with numpy, pypdf, python-docx, beautifulsoup4 and pytest installed and **no PyTorch**. The E5 encoder is imported lazily, so retrieval falls back to its declared keyword lane in tests.
  - 119 original runtime and model-setup tests, unchanged.
  - 15 tests for thinking budgets, the Gemma `think` flag, ablation switches, the config check that refuses a thinking deadline too short for planning, and a check that the launch profiles equal the evaluated variants.
  - 6 tests for the unattended runner (`scripts/run_full_eval.py`): command order, setup and pilot switches, the results zip, and the status health check that stops a broken run.
  - 21 tests for the evaluation pipeline. They cover link resolution for the three project configs and mining from recorded GitHub GraphQL and Discourse responses. They build a dated git repository and check that snapshots exclude docs written after the question. Then they run four variants end to end with the real index, engine and access manager and a scripted model. Judging, the report, the two review CSV round trips, the bootstrap and kappa, the access-control test (zero leaks), and isolation of variants from shell settings are also covered.

## Not checked here

- **No live model call.** Ollama was not available in the build environment. Gemma 4 with thinking on and JSON-schema output together is untested. Every call's status, parse errors and thinking length are recorded per question, so the first pilot run (`run --limit 20`) shows whether it works.
- **No live mining.** The build environment could not reach the GitHub API or Discourse forums. The miners were tested only against recorded responses, and category names in the project files are checked when the miner runs.
- **The runner script was only tested in dry-run mode** (commands and order), not executed end to end against Ollama.
- **No run with the real E5 encoder** inside the evaluation, and no Windows rerun of the suite. The CI workflow (`.github/workflows/tests.yml`) runs the suite on Ubuntu and Windows with Python 3.11 and 3.13 once pushed.
- Answer quality on real corpora is what `sisu_eval` measures. These tests do not establish it.
