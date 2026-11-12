# Running the SISU evaluation

Two machines, two people:

- **Part A, Shivansh's laptop (no GPU needed):** collect the questions, review them, send the `eval_data` folder to the office.
- **Part B, the office PC with the RTX 5080:** run everything with one command and send back the result zips.

Mined questions contain forum text. Send `eval_data` by OneDrive, Teams or email, **never through GitHub**.

---

## Part A: Shivansh (Mac)

**A1. Python environment, once (about 2 minutes).** The last command runs the test suite. It should end with `157 passed`.

```bash
cd ~/Downloads/Sisu_experimentation && git pull && conda create -y -n sisu python=3.11 && conda activate sisu && pip install -r requirements-test.txt && python -m pytest tests -q
```

**A2. GitHub token,** so the miner can read GitHub Discussions. With the GitHub CLI:

```bash
brew install gh && gh auth login && export GITHUB_TOKEN=$(gh auth token)
```

No Homebrew? Create a classic token at https://github.com/settings/tokens. It needs no scopes, because the miner only reads public discussions. Then type `export GITHUB_TOKEN=` followed by the token.

`export` only lasts for the current terminal. In a new terminal, run `conda activate sisu && export GITHUB_TOKEN=$(gh auth token)` again.

**A3. Collect questions and build the dated doc snapshots** for all three projects. This takes 30 to 90 minutes, mostly waiting on the APIs. It's safe to rerun because responses are cached.

```bash
cd ~/Downloads/Sisu_experimentation && conda activate sisu && python -m sisu_eval mine --project fastapi --limit 400 && python -m sisu_eval snapshot --project fastapi && python -m sisu_eval review-export --project fastapi && python -m sisu_eval mine --project pydantic --limit 400 && python -m sisu_eval snapshot --project pydantic && python -m sisu_eval review-export --project pydantic && python -m sisu_eval mine --project cpython-devguide --limit 400 && python -m sisu_eval snapshot --project cpython-devguide && python -m sisu_eval review-export --project cpython-devguide
```

Each `snapshot` prints how many questions were kept. If a project keeps fewer than about 100, tell Claude before going on.

**A4. Package and send.** You can send right away. The review in A5 only affects scoring, which happens later on your laptop, so the office machine doesn't have to wait for it.

```bash
cd ~/Downloads/Sisu_experimentation && zip -rq ~/Downloads/eval_data_for_office.zip eval_data -x "eval_data/*/repo/*" "eval_data/*/api_cache/*" && ls -lh ~/Downloads/eval_data_for_office.zip
```

**A5. Review, with a co-author.** Open `eval_data/fastapi/review.csv`, `eval_data/pydantic/review.csv` and `eval_data/cpython-devguide/review.csv`:
- Set `keep` to `n` where the linked docs don't actually answer the question.
- Set `answerable_override` only where the automatic date check is wrong.

Then import the reviews:

```bash
cd ~/Downloads/Sisu_experimentation && conda activate sisu && python -m sisu_eval review-import --project fastapi --csv eval_data/fastapi/review.csv && python -m sisu_eval review-import --project pydantic --csv eval_data/pydantic/review.csv && python -m sisu_eval review-import --project cpython-devguide --csv eval_data/cpython-devguide/review.csv
```

**A6. When result zips arrive from the office.** Unzip them into the repo folder; they unpack into `eval_data/`. Two raters then fill `eval_data/fastapi/calibration/calibration_sheet.csv` independently, and the same file for each other project once its zip arrives. Then build the report. The loop unpacks the zips oldest first, so a later `all` zip replaces an earlier `core` zip:

```bash
cd ~/Downloads/Sisu_experimentation && conda activate sisu && for f in $(ls -tr ~/Downloads/fastapi-*.zip); do unzip -oq "$f"; done && python -m sisu_eval calibrate --project fastapi --csv eval_data/fastapi/calibration/calibration_sheet.csv && python -m sisu_eval report --project fastapi && open eval_data/fastapi/report.md
```

---

## Part B: the office PC (Windows, RTX 5080)

You need about 50 GB of free disk space. Run the commands below in **PowerShell**, one line at a time.

**B1. Install the tools, once.** Skip any that are already installed.

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
winget install -e --id Ollama.Ollama
```

Close and reopen PowerShell, then start the **Ollama** app from the Start menu. It should show an icon in the system tray.

**B2. Get the code.**

```powershell
git clone https://github.com/Misha-17/Sisu_experimentation.git C:\sisu
cd C:\sisu
```

**B3. Put Shivansh's data in place.** Save `eval_data_for_office.zip` into Downloads first.

```powershell
Expand-Archive -Path "$HOME\Downloads\eval_data_for_office.zip" -DestinationPath C:\sisu -Force
```

**B4. Pilot run.** The first time, this installs packages and downloads about 30 GB of models, then answers 20 questions with each of two settings. Expect 1 to 3 hours.

```powershell
cd C:\sisu
py -3.11 scripts\run_full_eval.py --pilot-only
```

At the end it prints a table like this example (your numbers will differ):

```text
variant                     done  answered  median s   p90 s  bad calls  hours left
normal                    20/412        17        48      81     0 (0%)         5.2
thinking                  20/412        15       190     344     2 (3%)        20.7
```

**Send a photo or copy of that table to Shivansh and wait for the go-ahead.**

**B5. Full run.** This runs for days. The script stops Windows from sleeping while it runs, but keep the PC plugged in.

```powershell
cd C:\sisu
py -3.11 scripts\run_full_eval.py --skip-setup
```

- **Stopping and restarting is safe.** Ctrl+C, a crash or a reboot loses nothing. Run the same command again and it continues where it stopped.
- **Results:** after each phase, a zip appears in `C:\sisu\results\`, for example `fastapi-core-20261205-1430.zip`. Send each new zip to Shivansh.
- **Progress:** in a second PowerShell window, run `cd C:\sisu` and then `.\.venv\Scripts\python.exe -m sisu_eval status --project fastapi`.
- **Logs:** everything is written to `C:\sisu\logs\`.
- **Code updates:** if Shivansh says the code changed, run `git pull`, then start B5 again.

### If something goes wrong

| Message | What to do |
|---|---|
| `Ollama is not running` | Start the Ollama app, then rerun the same command |
| `py is not recognized` | Reinstall Python 3.11 from B1 and tick "py launcher" |
| `STOP: variant normal looks broken` | Don't continue. Send the newest file in `C:\sisu\logs\` to Shivansh |
| `No questions found under eval_data` | Redo B3 and check that `C:\sisu\eval_data\fastapi\questions.jsonl` exists |
| Very slow, `ollama ps` shows `CPU` | Close other programs that use the GPU (games, other models), then rerun |
| Disk full | Free space on C:, then rerun |
