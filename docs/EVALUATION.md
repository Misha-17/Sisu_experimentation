# Evaluation pipeline

`python -m sisu_eval` evaluates the assistant on **real questions with real documents and no human participants**:

1. **Mine** public threads where a newcomer asked a question and a maintainer answered with a link to the project's docs. The maintainer's reply is the reference answer and the linked page (and section) is the gold evidence.
2. **Snapshot** the docs from git **as they stood when the question was asked** (start of its month by default). Docs written later cannot leak in. If the linked page or section did not exist yet, the question is *unanswerable at its date*, and the correct behaviour is to refuse. Those questions are also asked again against current docs (`--head-check`).
3. **Run** every variant: the two launch profiles (`normal`, `thinking`), a control with thinking off but the thinking budgets (`normal_long_budget`), ablations, and a plain RAG baseline.
4. **Judge** answer correctness against the maintainer's reply, and check whether each answer is supported by its cited passages. The default judge is `gpt-oss:20b`, a different model family from the Gemma 4 generator.
5. **Calibrate** the judge against two human raters (Cohen's kappa) before reporting its numbers. Rating is done by co-authors, not study participants.
6. **ACL test**: a user who may only see half the documents asks every question. Leaks should be zero.
7. **Report** means with 95% bootstrap CIs and paired differences against `normal`.

All data goes to `eval_data/<project>/` (ignored by Git).

For the two-machine workflow (questions collected on a laptop, runs on the GPU PC with one command: `scripts/run_full_eval.py`), follow [RUNNING.md](../RUNNING.md). The commands below are the individual steps that script runs. `python -m sisu_eval status --project fastapi` shows progress, speed and failed model calls per variant at any time.

## Projects

| Project | Docs | Questions from | Onboarding type |
|---|---|---|---|
| `fastapi` | `docs/en/docs` in fastapi/fastapi | GitHub Discussions, category Questions | newcomer to a framework |
| `pydantic` | `docs` in pydantic/pydantic | GitHub Discussions (Question / Questions / Q&A) | newcomer to a library |
| `cpython-devguide` | python/devguide (reStructuredText, indexed as text) | discuss.python.org (core-dev, documentation, help) | new contributor |

The miner checks category names against the live site. If none match, it stops and prints the real names. Yields differ a lot between projects: mine a few hundred threads first and look at `snapshots.json` before running models. To add a project, copy one of the JSON files in `sisu_eval/projects/`. A link pattern needs a named group `path` that maps a docs URL to a file under `root`.

## Setup (once)

PowerShell 7 (Windows). In Windows PowerShell 5.1, run the commands one line at a time instead of chaining with `&&`.

```powershell
py -3.11 -m venv .venv && .\.venv\Scripts\python.exe -m pip install --upgrade pip && .\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt && .\.venv\Scripts\python.exe scripts/download_encoder.py && ollama pull gemma4:12b && ollama pull gemma4:e4b && ollama pull gpt-oss:20b
$env:GITHUB_TOKEN = (gh auth token)
```

Linux / macOS:

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && .venv/bin/python -m pip install -r requirements-dev.txt && .venv/bin/python scripts/download_encoder.py && ollama pull gemma4:12b && ollama pull gemma4:e4b && ollama pull gpt-oss:20b && export GITHUB_TOKEN=$(gh auth token)
```

GitHub Discussions are only available through the GraphQL API, which needs a token. Any token that can read public repositories works. Discourse forums need no token.

## Full run for one project

Windows (PowerShell 7):

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m sisu_eval mine --project fastapi --limit 400 && & $py -m sisu_eval snapshot --project fastapi && & $py -m sisu_eval review-export --project fastapi
```

Linux / macOS:

```bash
.venv/bin/python -m sisu_eval mine --project fastapi --limit 400 && .venv/bin/python -m sisu_eval snapshot --project fastapi && .venv/bin/python -m sisu_eval review-export --project fastapi
```

Open `eval_data/fastapi/review.csv` and go through it with a co-author. Set `keep` to `n` for threads where the linked docs do not actually answer the question (for example, the maintainer linked a general page and solved it in prose). Set `answerable_override` only when the automatic date check is wrong. Then:

```bash
.venv/bin/python -m sisu_eval review-import --project fastapi --csv eval_data/fastapi/review.csv && .venv/bin/python -m sisu_eval index --project fastapi && .venv/bin/python -m sisu_eval run --project fastapi --variants normal --limit 20
```

**Time the 20-question pilot before anything else.** `eval_data/fastapi/runs/normal.jsonl` records `wall_s` per question. Multiply by the number of questions and variants to get the full cost. The thinking profile allows up to 420 s per answer. Its real time on your GPU is unknown until you measure it, so pilot it too:

```bash
.venv/bin/python -m sisu_eval run --project fastapi --variants thinking --limit 20
```

Then run everything. Runs resume where they stopped, so you can interrupt and restart:

```bash
.venv/bin/python -m sisu_eval run --project fastapi --variants normal,thinking,normal_long_budget,plain_rag --head-check && .venv/bin/python -m sisu_eval run --project fastapi --variants normal_e4b,lexical_only,dense_only,no_planning,no_review,no_neighbors && .venv/bin/python -m sisu_eval judge --project fastapi && .venv/bin/python -m sisu_eval calibrate-export --project fastapi --n 100
```

Two raters fill `eval_data/fastapi/calibration/calibration_sheet.csv` **independently**. They are blind to variant and judge verdict. Allowed values: `correct`, `partially_correct`, `incorrect` for correctness, and `fully_supported`, `partially_supported`, `unsupported` for support. Then:

```bash
.venv/bin/python -m sisu_eval calibrate --project fastapi --csv eval_data/fastapi/calibration/calibration_sheet.csv && .venv/bin/python -m sisu_eval acl --project fastapi && .venv/bin/python -m sisu_eval report --project fastapi && .venv/bin/python -m sisu_eval export-public --project fastapi
```

Results: `eval_data/fastapi/report.md`, `report.json`, `metrics.csv`.

## Variants

Defined in `sisu_eval/variants.json`. `normal` and `thinking` are the same values as the two launch profiles in `run_assistant.py`; a test fails if they drift apart.

| Variant | What changes |
|---|---|
| `normal` | Launch profile `normal`: Gemma 4 12B, thinking off, 95 s |
| `thinking` | Launch profile `thinking`: thinking on, +3,072 tokens per call, 4x call caps, 420 s, 240 s request timeout, 65,536 answer tokens |
| `normal_long_budget` | Thinking off with the thinking budgets. `thinking` vs this isolates thinking from extra time |
| `normal_e4b` | Gemma 4 E4B generator |
| `lexical_only` / `dense_only` | One retrieval lane (passages unchanged) |
| `no_planning` | Original question as the only facet |
| `no_review` | No source review, no revision |
| `no_neighbors` | No adjacent passages or blocks |
| `plain_rag` | Same retrieval and access control, one generation call citing [P#], no planning, binding or review |

## Metrics

Answerable questions (gold page present at the question's date):

- `correct_lenient` / `correct_strict`: judge verdict vs the maintainer's reply. Lenient scores correct = 1, partially correct = 0.5, refusal = 0. Strict counts only correct.
- `evidence_doc_recall`: the gold page was among the passages shown to the model.
- `evidence_section_recall`: same, at the linked heading (only for links with an anchor).
- `citation_gold_hit`: the answer cites the gold page (answered questions only).
- `fully_supported`: judge finds every statement backed by the cited passages (answered questions only).
- `false_refusal`: no answer although the gold page existed.

Unanswerable at their date: `correct_refusal_unanswerable`, the share where the system declined.

Cost: median and p90 seconds, `within_95s` (share that would meet the normal profile's deadline), model calls, output tokens (Ollama `eval_count`, includes thinking), thinking characters, call errors.

## Data handling

Mined question and answer text stays in `eval_data/` on your machine. For a paper or artifact, release `public_release.jsonl` from `export-public`: thread URLs, commit hashes and gold paths, with no post text and no usernames. Check each forum's terms before publishing anything else. GitHub API use falls under GitHub's terms, and Discourse sites set their own. The miner sends a generic User-Agent, waits between requests, retries on 429 with `Retry-After`, and caches responses in `eval_data/<project>/api_cache/` so reruns do not refetch.

## Known limits

- The gold evidence is the page the maintainer linked. Another page may also answer the question, so recall is a lower bound.
- The maintainer's reply can contain advice that is not in the docs, so a correct docs-based answer can be graded partial.
- `answerable_at_snapshot` comes from file and heading existence. A page can exist and still not contain the answer yet. The review pass is where you catch that.
- Month granularity (the default) means a snapshot can be up to a month older than the question, so a page added in the same month counts as not yet existing. Use `--granularity week` or `day` for closer dates at the cost of more snapshots to store and index (each snapshot is a full copy of the docs).
- An LLM judge has its own errors and biases. Report its kappa against your raters next to every number it produces.
