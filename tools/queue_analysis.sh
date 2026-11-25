#!/usr/bin/env bash
# The last job of every experiment queue: when the queue's jobs are over, start an unattended Claude Code run that
# analyses the results, writes them into the note and uploads the journal write-up. Without it a queue that
# finishes after its session has ended is read by nobody (2026-11-22: RL on SmolVLA, experiments 51 and 52).
#
# Queue it LAST and without -d / -D, so it also runs when an earlier job failed (a failure needs reading too):
#
#   tsp -L analysis bash tools/queue_analysis.sh --experiment 58 \
#       --note notes/my-experiment-20261201.md \
#       --dir artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/my_experiment_20261201 \
#       --predictions notes/my-experiment-20261201-predictions.json --jobs 300-342
#
#   --experiment  the experiment's number in the journal (python tools/journal_writeup.py record); required
#   --note        the experiment's note, where the results section goes; required
#   --dir         the folder the jobs wrote their results to; required
#   --predictions the file of predictions registered before the run, if there is one
#   --jobs        the task-spooler ids of the queue's jobs, as text for the reader ("300-342")
#   --dry-run     print the command and the instructions, start nothing
#   --smoke       start the run with a harmless check instead of an analysis (tests login, tools and the journal)
#
# The job returns at once: the analysis runs detached, so it never holds the GPU queue. Its log is
# <dir>/analysis_<time>.log. It may read anything, write under notes/, upload to the journal and commit; it cannot
# run Python tools other than tools/journal_writeup.py, so it cannot train, evaluate, use the GPU or move the arm.
# Login: the run uses a long-lived token kept in ~/.config/so-arm-101/analysis-token (or ANALYSIS_TOKEN_FILE), made
# once with `claude setup-token`. The stored terminal login is not enough on this PC: the desktop app rewrites it
# without a refresh token, so it expires. The run starts with a clean environment, so nothing of the session that
# queued it (the desktop app's own login and address) leaks into it.
# ANALYSIS_TIMEOUT (default 60m) and ANALYSIS_BUDGET_USD (default 20) bound one run; ANALYSIS_MODEL picks the model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="" note="" dir="" predictions="" jobs="" dry=0 smoke=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) experiment="$2"; shift 2 ;;
    --note) note="$2"; shift 2 ;;
    --dir) dir="$2"; shift 2 ;;
    --predictions) predictions="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    --smoke) smoke=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$experiment" || -z "$note" || -z "$dir" ]]; then
  echo "Usage: tools/queue_analysis.sh --experiment NN --note notes/X.md --dir <results folder> [--predictions FILE] [--jobs IDS] [--dry-run|--smoke]" >&2
  exit 2
fi
cd "$ROOT"
[[ -d "$dir" ]] || { echo "No results folder $dir" >&2; exit 2; }
CLAUDE="$(command -v claude || true)"; [[ -n "$CLAUDE" ]] || CLAUDE="$HOME/.local/bin/claude"
[[ -x "$CLAUDE" ]] || { echo "The claude command was not found (looked on PATH and in ~/.local/bin)." >&2; exit 3; }

token_file="${ANALYSIS_TOKEN_FILE:-$HOME/.config/so-arm-101/analysis-token}"
clean=(env -i "HOME=$HOME" "PATH=$PATH" "TERM=${TERM:-xterm}" "LANG=${LANG:-C.UTF-8}")
[[ -s "$token_file" ]] && clean+=("CLAUDE_CODE_OAUTH_TOKEN=$(tr -d '[:space:]' < "$token_file")")
log="$dir/analysis_$(date +%Y%m%dT%H%M%S).log"
if [[ $smoke -eq 1 ]]; then
  prompt="This is a test of the unattended analysis job, not an analysis. Do exactly this and nothing else: (1) run \`python3 tools/journal_writeup.py record\` and report its last line; (2) run \`tsp -l\` and report how many lines it printed; (3) try to run \`python3 -c 'print(1)'\` and report whether it was allowed (it should be refused); (4) say whether the file $note exists. Change no file. Answer in four short lines."
else
  prompt="You are running unattended as the last job of an experiment queue in the SO-ARM-101 repository. Nobody is watching and nobody will answer a question, so do not ask any: decide, do the work, and write down what you decided.

The queue for experiment $experiment has ended. Its results are in $dir. Its note is $note.${predictions:+ The predictions registered before the run are in $predictions.}${jobs:+ Its task-spooler jobs were $jobs.}

Do this:
1. Run \`tsp -l\` and find the queue's jobs. Note which finished, which failed (a non-zero exit) and which were skipped. Read the results folder: the report, matrix, summary and result files, and the logs of any job that failed.
2. Follow the experiment-analyze-explain skill (.claude/skills/experiment-analyze-explain/SKILL.md) on what the files hold. You cannot take new measurements here: if the analysis needs one, say which, and put it under the follow-ups.
3. If predictions were registered, mark every one of them held, failed, half or open, from the files. Read all of them, not only the headline ones, and say plainly which failed.
4. Add a dated results section to the note: the tables with their baseline rows, what held and what failed, what it means for the decision the experiment was for, the limits, and what to run next. If the note says the experiment is queued or running, correct that.
5. Write notes/journal-writeups/$experiment.json (rules and fields: notes/journal-writeups/README.md), then \`python3 tools/journal_writeup.py put $experiment\`, then set the record's outcome with \`python3 tools/journal_writeup.py update $experiment <patch file under notes/journal-writeups/>\` (status, a one-line answer, up to eight numbers). If the queue failed part-way, write up what exists, name the failed stage as trouble in run.stages, and leave the record's status as running.
6. Commit the note and the write-up (git add the two files by name, then git commit). Do not push.

Hard limits: every number comes from a file you read; never estimate one. Do not start training, evaluation, captures or any GPU job, do not queue anything, and do not run anything that talks to the robot arm. You can only run tools/journal_writeup.py, tsp -l, ls, wc and git status, log, diff, add and commit; that is deliberate. Finish with five lines: what the experiment found, what failed, what you could not do, the commit, and whether the upload went through."
fi

allowed=(Read Glob Grep Skill "Edit(notes/**)"
  "Bash(python3 tools/journal_writeup.py:*)" "Bash(python tools/journal_writeup.py:*)"
  "Bash(tsp -l)" "Bash(tsp -l:*)" "Bash(ls:*)" "Bash(wc:*)"
  "Bash(git status:*)" "Bash(git log:*)" "Bash(git diff:*)" "Bash(git add:*)" "Bash(git commit:*)")
cmd=("${clean[@]}" timeout "${ANALYSIS_TIMEOUT:-60m}" "$CLAUDE" -p "$prompt" --permission-mode dontAsk
  --max-budget-usd "${ANALYSIS_BUDGET_USD:-20}" --allowedTools "${allowed[@]}")
[[ -n "${ANALYSIS_MODEL:-}" ]] && cmd+=(--model "$ANALYSIS_MODEL")

if [[ $dry -eq 1 ]]; then
  echo "Would run, detached, with its log in $log:"; echo "  timeout ${ANALYSIS_TIMEOUT:-60m} $CLAUDE -p <instructions> ...  (token file: $([[ -s "$token_file" ]] && echo found || echo missing))"
  echo "Allowed tools: ${allowed[*]}"; echo; echo "$prompt"
  exit 0
fi
# A logged-out CLI would fail inside the detached run, where nobody sees it. Check first, so the failure is this
# job's exit code in `tsp -l`.
if ! check="$("${clean[@]}" timeout 90 "$CLAUDE" -p "Reply with the single word ok." --permission-mode dontAsk 2>&1 < /dev/null)"; then
  echo "The claude command could not start a run, so experiment $experiment was NOT analysed:" >&2
  echo "  ${check##*$'\n'}" >&2
  echo "Make a long-lived token once: run \`claude setup-token\` in a terminal and save the token it prints as $token_file (chmod 600). Then run this job again." >&2
  exit 4
fi
{ echo "queue_analysis: experiment $experiment, started $(date -Is)"; echo "note: $note"; echo "results: $dir"; echo; } > "$log"
setsid nohup "${cmd[@]}" >> "$log" 2>&1 < /dev/null &
echo "Analysis of experiment $experiment started (pid $!). Its log: $log"
