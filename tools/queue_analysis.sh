#!/usr/bin/env bash
# The last job of every experiment queue: when the queue's jobs are over, have OpenClaw analyse the results, then
# put its analysis into the note, upload the journal write-up and commit. Without it a queue that finishes after
# its session has ended is read by nobody (2026-11-22: RL on SmolVLA, experiments 51 and 52).
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
#   --dry-run     print what would be sent and the instructions, start nothing
#   --smoke       send OpenClaw a harmless file task instead of an analysis (tests the whole round trip)
#
# How it works (user request 2026-11-25: OpenClaw does the analysis). OpenClaw runs on the laptop and never
# touches this machine, so it cannot train, evaluate, use the GPU or move the arm:
#   1. The results folder's text files (reports, matrices, logs; no models, frames or telemetry), the note, the
#      predictions, `tsp -l`, and the rules (the skill, the write-up guide and model) are copied to the laptop,
#      into ~/.openclaw/workspace/so-arm-101-analysis/exp-<NN>/in.
#   2. `openclaw agent` is given the instructions and writes three files into .../out: results.md (the results
#      section for the note), <NN>.json (the journal write-up) and patch.json (the record's outcome).
#   3. This machine copies them back, appends results.md to the note (it never rewrites the note), validates and
#      uploads the write-up, sets the outcome and commits the note and the write-up. Nothing is pushed.
# The job returns at once; steps 2 and 3 run detached, so the GPU queue is never held. The log is
# <dir>/analysis_<time>.log. ANALYSIS_HOST (default ubuntu-laptop), ANALYSIS_TIMEOUT in seconds (default 3000),
# ANALYSIS_MODEL (an OpenClaw model id) and ANALYSIS_THINKING (default high). The model is Gemini 4 Argon until
# Neon is available to us (user, 2026-11-25); change MODEL below then. OpenClaw's own default is a fast model.
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
[[ "$experiment" =~ ^[0-9]{2,3}$ ]] || { echo "--experiment is the journal's number, like 58" >&2; exit 2; }
[[ -d "$dir" ]] || { echo "No results folder $dir" >&2; exit 2; }
[[ -f "$note" ]] || { echo "No note $note" >&2; exit 2; }
[[ -z "$predictions" || -f "$predictions" ]] || { echo "No predictions file $predictions" >&2; exit 2; }

HOST="${ANALYSIS_HOST:-ubuntu-laptop}"
TIMEOUT="${ANALYSIS_TIMEOUT:-3000}"
MODEL="${ANALYSIS_MODEL:-google-vertex/gemini-4-argon}"
REMOTE=".openclaw/workspace/so-arm-101-analysis/exp-$experiment"   # under the laptop user's home
stamp="$(date +%Y%m%dT%H%M%S)"
log="$dir/analysis_$stamp.log"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST")

# ---- What OpenClaw is given.
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/in/results" "$stage/in/rules" "$stage/out"
( cd "$dir" && find . -type f \( -name '*.json' -o -name '*.txt' -o -name '*.md' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.log' \) \
    -size -2M -not -path '*/telemetry/*' -not -path '*/wandb/*' -not -path '*/model/*' -not -path '*/frames/*' \
    -not -name 'analysis_*.log' -print0 | rsync -a --from0 --files-from=- . "$stage/in/results/" )
cp "$note" "$stage/in/note.md"
[[ -n "$predictions" ]] && cp "$predictions" "$stage/in/predictions.json"
(tsp -l 2>/dev/null || echo "tsp -l could not be read") > "$stage/in/tsp.txt"
cp .claude/skills/experiment-analyze-explain/SKILL.md "$stage/in/rules/analysis-skill.md"
cp notes/journal-writeups/README.md "$stage/in/rules/writeup-guide.md"
cp notes/journal-writeups/53.json "$stage/in/rules/writeup-model-53.json"
python3 tools/journal_writeup.py record > "$stage/in/journal-record.txt" 2>/dev/null || echo "the journal did not answer" > "$stage/in/journal-record.txt"
size="$(du -sh "$stage/in" | cut -f1)"; count="$(find "$stage/in" -type f | wc -l)"

base="\$HOME/$REMOTE"
if [[ $smoke -eq 1 ]]; then
  cat > "$stage/prompt.txt" <<PROMPT
This is a test of an automatic job, not real work. In the folder $base there is a folder "in". Do exactly this and nothing else: count the files under $base/in/results, read the first line of $base/in/note.md, and write a file $base/out/results.md with two lines: the count, and that first line. Do not touch anything outside $base. Reply with one line saying you wrote it.
PROMPT
else
  cat > "$stage/prompt.txt" <<PROMPT
This is an automatic job with nobody watching, so do not ask questions: decide, do the work, and write down what you decided.

A machine-learning experiment queue has finished for the SO-ARM-101 robot-arm project (experiment $experiment in the Agentic Journal's record). Everything you need is in $base/in on this computer:
- in/results/: the report, matrix, summary, result and log files the jobs wrote.
- in/note.md: the experiment's note (its plan, what was run, the baselines).${predictions:+
- in/predictions.json: the predictions registered before the run.}
- in/tsp.txt: the job queue's listing. A job with a non-zero exit failed; "skipped" jobs did not run.${jobs:+ This experiment's jobs were $jobs.}
- in/journal-record.txt: the journal's experiments, so you can name the ones this builds on.
- in/rules/analysis-skill.md: how to analyse and explain a result. Follow it on the files you have; you cannot take new measurements, so if one is needed, say which under the follow-ups.
- in/rules/writeup-guide.md and in/rules/writeup-model-53.json: the fields, limits and rules of a journal write-up, and a finished one to copy the level of detail from.

Write exactly these three files into $base/out:
1. results.md: the results section for the note, starting with a "## Results" heading that carries today's date. Tables with their baseline rows, what held and what failed, what it means for the decision the experiment was for, the limits, and what to run next.
2. $experiment.json: the journal write-up, following the guide exactly (valid JSON, only the guide's fields, within its limits).
3. patch.json: {"status": "worked" | "failed" | "mixed" | "finding", "answer": "<one line>", "numbers": ["<up to 8 short strings>"]}. If the queue failed part-way, write up what exists, name the failed stage as "trouble" in run.stages, and use "status": "running".

Rules: every number comes from a file you read; copy it as the file gives it and never estimate one. If predictions were registered, mark every one of them held, failed, half or open, not only the headline ones, and say plainly which failed. Plain words for a new intern, every term defined on first use, no em dashes. Work only inside $base: do not use ssh, do not contact any other computer, do not start jobs. Finish by replying with five short lines: what the experiment found, what failed, what you could not determine, which files you wrote, and anything a person should check.
PROMPT
fi

if [[ $dry -eq 1 ]]; then
  echo "Would send $count files ($size) to $HOST:~/$REMOTE/in, run OpenClaw for up to ${TIMEOUT}s, and log to $log."
  echo "Files: $(cd "$stage/in" && find . -type f | sort | head -12 | tr '\n' ' ')..."
  echo; cat "$stage/prompt.txt"
  exit 0
fi

# ---- Failures a person must see are this job's exit code in `tsp -l`.
if ! "${SSH[@]}" 'command -v openclaw >/dev/null && systemctl --user is-active --quiet openclaw-gateway.service' 2>/dev/null; then
  echo "OpenClaw could not be reached on $HOST (no ssh, no openclaw command, or its gateway is stopped), so experiment $experiment was NOT analysed. Run this job again when it is back." >&2
  exit 4
fi
"${SSH[@]}" "rm -rf ~/$REMOTE && mkdir -p ~/$REMOTE/out"
rsync -a -e "ssh -o BatchMode=yes" "$stage/in" "$stage/prompt.txt" "$HOST:$REMOTE/"

# ---- The rest runs detached: OpenClaw's turn, then bringing its files home.
finish="$dir/.analysis_finish_$stamp.sh"
cat > "$finish" <<FINISH
#!/usr/bin/env bash
set -uo pipefail
cd "$ROOT"
say() { echo "[\$(date +%H:%M:%S)] \$*"; }
say "OpenClaw ($MODEL) is analysing experiment $experiment on $HOST (up to ${TIMEOUT}s)."
ssh -o BatchMode=yes -o ServerAliveInterval=30 "$HOST" 'openclaw agent --agent main --session-key so-arm-101-exp-$experiment-analysis-$stamp --message-file ~/$REMOTE/prompt.txt --timeout $TIMEOUT --thinking ${ANALYSIS_THINKING:-high} --model $MODEL' 2>&1 | tail -60
out="\$(mktemp -d)"
rsync -a -e "ssh -o BatchMode=yes" "$HOST:$REMOTE/out/" "\$out/" || { say "FAILED: could not copy OpenClaw's files back."; exit 1; }
say "OpenClaw wrote: \$(ls "\$out" | tr '\n' ' ')"
if [[ $smoke -eq 1 ]]; then
  say "Smoke test. results.md says:"; cat "\$out/results.md" 2>/dev/null || say "FAILED: no results.md"
  rm -rf "\$out" "$finish"; exit 0
fi
ok=1
if [[ -s "\$out/results.md" ]]; then
  { echo; echo "<!-- written by OpenClaw from the queue's files, $stamp; not reviewed -->"; cat "\$out/results.md"; } >> "$note"
  say "Appended the results section to $note."
else say "FAILED: no results.md, the note is unchanged."; ok=0; fi
if [[ -s "\$out/$experiment.json" ]]; then
  cp "\$out/$experiment.json" "notes/journal-writeups/$experiment.json"
  if python3 tools/journal_writeup.py put $experiment; then
    [[ -s "\$out/patch.json" ]] && { python3 tools/journal_writeup.py update $experiment "\$out/patch.json" || { say "FAILED: the record's outcome was refused."; ok=0; }; }
  else say "FAILED: the write-up is not valid or the journal refused it. The file is kept at notes/journal-writeups/$experiment.json."; ok=0; fi
else say "FAILED: no write-up file."; ok=0; fi
git add -- "$note" "notes/journal-writeups/$experiment.json" 2>/dev/null
git commit -q -m "Experiment $experiment: results section and journal write-up, written by OpenClaw at the end of the queue (not reviewed)" -- "$note" "notes/journal-writeups/$experiment.json" && say "Committed \$(git log --oneline -1 | cut -c1-70)"
[[ \$ok -eq 1 ]] && say "Done." || say "Done, with the failures above."
rm -rf "\$out" "$finish"
FINISH
{ echo "queue_analysis: experiment $experiment, started $(date -Is), $count files ($size) sent to $HOST:~/$REMOTE"; echo "note: $note"; echo "results: $dir"; echo; } > "$log"
setsid nohup bash "$finish" >> "$log" 2>&1 < /dev/null &
echo "Experiment $experiment was handed to OpenClaw on $HOST (pid $!). Its log: $log"
