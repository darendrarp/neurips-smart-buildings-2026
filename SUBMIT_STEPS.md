# SUBMIT_STEPS — How to submit (for the user, in the browser)

Everything below needs a browser and the user's own Kaggle account. Nothing
here could be done from the VM (no Kaggle account, no live browser), and no
step was performed on the user's behalf.

## Before you start

- **File to upload:** `submission_notebook.ipynb` from this repo
  (https://github.com/darendrarp/neurips-smart-buildings-2026).
  Download it to your computer first (repo page → `submission_notebook.ipynb`
  → Raw → save, or clone the repo).
- **Cost:** free. No Kaggle paid tier is needed.
- **Time needed:** ~15 minutes for the first submission.

## Step 1 — Create / log in to Kaggle

1. Go to https://www.kaggle.com/ and log in (or create a free account).
   Use your own identity — do not create a second account (one account per
   person, per the rules).

## Step 2 — Join the competition and accept the rules

1. Go to https://www.kaggle.com/competitions/neurips-2026-smart-buildings
2. Click **Join Competition** and accept the Official Rules.
3. ⚠️ **Check the Timeline page first** (Overview → Timeline, same URL,
   Timeline tab):
   - **Entry Deadline** — the last day to join/accept the rules. This
     could NOT be verified from the VM; if it has passed, you cannot
     enter. Check it before doing anything else.
   - **Final Submission Deadline** — listed as 15 January 2027 in the
     rules, but the exact TIME was not verifiable from the VM. Kaggle
     deadlines are conventionally 11:59 PM UTC (16 Jan 06:59 WIB) — do
     not rely on that without checking the Timeline page.
   - **Team Merger Deadline** — only relevant if joining a team; solo
     entry needs nothing here.

## Step 3 — Check the official submission format (important)

The starter kit says the submission template is on the competition page,
but the exact grading contract could not be verified from the VM (the
Data tab errored and public notebooks' code was not readable as text).
Before submitting, confirm ONE of these:

- Open the competition's **Code** tab and look for an official starter
  notebook / submission template. If it defines the entry point
  differently (e.g. a different class name or file), adapt the notebook
  accordingly — the `Policy` class in our notebook implements the
  interface from the official starter kit
  (`Policy(building_meta_data, action_spec)` + `get_action(obs)`), and an
  alias `TransferPolicy = Policy` is included in case the harness looks
  for the starter-kit class name.
- If there is no starter notebook, our notebook as-is is the submission:
  it defines `Policy` at top level and runs a smoke test.

## Step 4 — Upload the notebook

1. On the competition page, go to the **Code** tab → **New Notebook**
   (or https://www.kaggle.com/code → New Notebook).
2. Upload `submission_notebook.ipynb` (File → Upload notebook, or drag
   the file in). Title it e.g. "Smart Buildings — Adaptive Preconditioning".
3. Make sure the notebook is attached to the competition (it should show
   under the competition's Code tab; if you created it from the
   competition page it is attached automatically).

## Step 5 — Run All and verify

1. Click **Save Version** → **Save & Run All** (or Run All, then Save).
2. Wait for the run (~1–2 minutes; the notebook is small and needs no
   input data). The last code cell must print:
   `smoke test passed: 50 steps, all actions valid`
3. If any cell errors, do NOT submit — copy the error and send it back
   for a fix.

## Step 6 — Submit to the competition

1. In the notebook viewer, click **Submit to Competition** (or from the
   notebook's version menu, "Submit").
2. Confirm. The notebook is scored on the private leaderboard
   (6 unseen buildings × 5 unseen weather stations). Scoring takes a
   while — check the **Leaderboard** tab later:
   https://www.kaggle.com/competitions/neurips-2026-smart-buildings/leaderboard
3. Limits: max **5 submissions per day**, max **2 final submissions**.
   Use early submissions to learn the real score; save final-submission
   picks for the best versions near the deadline.

## Step 7 — Before the final deadline

1. Select up to **2 final submissions** for judging (Leaderboard →
   "Select final submissions") before the Final Submission Deadline.
   If you select none, Kaggle uses a default — pick deliberately.
2. If the score is near the top 8, expect winner verification: prize
   acceptance documents + IRS Form W-8BEN (non-US winner) and possibly a
   recorded/panel call with the sponsors (remote). Keep the repo public
   and reproducible — winners must deliver code + documentation.

## Quick reference (URLs)

| Step | URL |
| --- | --- |
| Competition | https://www.kaggle.com/competitions/neurips-2026-smart-buildings |
| Rules | https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules |
| Timeline (verify deadlines!) | Competition page → Overview → Timeline tab |
| Leaderboard | https://www.kaggle.com/competitions/neurips-2026-smart-buildings/leaderboard |
| Code / notebooks | https://www.kaggle.com/competitions/neurips-2026-smart-buildings/notebooks |
| Official site | https://smart-buildings-2026.github.io/competition/ |
| Starter kit | https://github.com/smart-buildings-2026/starter-kit |
| This repo | https://github.com/darendrarp/neurips-smart-buildings-2026 |

## What to report back after submitting

- The public leaderboard score of the first submission (and its rank).
- Any error during Run All or submission (copy the exact message).
- The verified Entry Deadline + exact Final Submission time from the
  Timeline page (so the record can be corrected).
