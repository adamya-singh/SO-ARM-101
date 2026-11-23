# Experiment write-ups for the journal

One file per experiment, `NN.json`, where `NN` is the experiment's number in the Agentic Journal's record
(`python tools/journal_writeup.py record` lists them). The journal shows the file in that experiment's workspace,
in two tabs: **Understand it** and **Run and watch**. `53.json` is the model to copy.

    python tools/journal_writeup.py check 41     # validate
    python tools/journal_writeup.py put 41       # validate, then upload

The reader is a new intern: plain words, short sentences, every term defined on first use. Write it in the order
the `experiment-analyze-explain` skill explains a result.

## Rules

- Every number comes from a file in this repo (a note, a report, a research-log entry). Never estimate, round
  differently from the source, or fill a gap from memory. If the source does not have it, leave it out.
- A table needs its baseline in it: mark the comparison rows `baseline` and the rows the experiment is about `ours`.
- `predictions` are only what was written down before the run (a predictions file, a pre-registered bar in the
  note). If nothing was registered, leave the list empty. Do not invent predictions after the fact.
- `trust` lists what makes the comparison fair (`ok: true`) and the limits (`ok: false`): one seed, sim only,
  different scenes, small n.
- Leave out any field you cannot fill from the sources. Every field is optional.
- No em dashes. No project shorthand without a definition.

## Fields

Understand it:

| Field | Limit | What goes in it |
|---|---|---|
| `question` | 600 chars | What the experiment asks, in one or two sentences |
| `measure` | 600 | What is measured and how to read it |
| `outcome` | 600 | What came out, in one or two sentences |
| `setup` | 2500 | What a new intern must know first |
| `parts[]` | 8 parts | One per sub-question. `title` (200), `why` (800), `did` (800), `predictions[]` (`text` 300, `result`: `held`, `failed`, `half`, `open`), `table` (`head`: up to 8 cells; `rows[]`: up to 24, each `cells` as wide as `head` and `mark`: `baseline`, `ours` or `""`; a cell is at most 80 chars) or `null`, `conclusion` (800) |
| `trust[]` | 16 | `{text (400), ok}` |
| `meaning` | 1500 | What it means for the decision it informed |
| `lesson` | 600 | The one lesson |
| `next[]` | 6 | `{title (200), why (400)}`: follow-ups that were, or should be, run |
| `log` | 6000 | The research-log entry as markdown, only if one exists in `notes/research-log/` |

Run and watch (`run`):

| Field | Limit | What goes in it |
|---|---|---|
| `state` | 120 | `Finished`, or `Running: <stage>` |
| `line` | 300 | When, how many stages, what is left |
| `watch[]` | 8 | `{title (80), note (160), href}`: `href` is an http(s) address (a W&B run) or `""` |
| `command` | 600 | A command that replays or re-runs it, only if the source gives one |
| `stages[]` | 24 | `{name (80), note (300), state}`: `done`, `trouble`, `running` or `waiting` |
| `housekeeping[]` | 16 | `{text (400), ok}`: committed, written up, logged |
| `files[]` | 24 | `{path (300), note (160)}`: the note, the report, the queue script, commits |
