#!/usr/bin/env python3
"""Check and upload experiment write-ups to the Agentic Journal.

A write-up is notes/journal-writeups/<NN>.json (NN = the experiment's number in the journal's
record). It fills the two tabs of that experiment's workspace: "Understand it" and "Run and watch".
The fields and their limits are in notes/journal-writeups/README.md.

  python tools/journal_writeup.py check [NN ...]   validate files (all of them when none is named)
  python tools/journal_writeup.py put NN [NN ...]  validate, then upload
  python tools/journal_writeup.py put --all
  python tools/journal_writeup.py get NN           print what the journal has
  python tools/journal_writeup.py record           list the journal's experiments
  python tools/journal_writeup.py add FILE.json    create a record ({question, direction, when, status, ...})
  python tools/journal_writeup.py update NN FILE.json   patch a record's fields (status, answer, numbers, ...)

The journal's address comes from JOURNAL_URL (default: the laptop's tailnet address).
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = os.environ.get("JOURNAL_URL", "https://ubuntu-laptop.taile85e97.ts.net").rstrip("/")
DIR = Path(__file__).resolve().parent.parent / "notes" / "journal-writeups"
RESULTS = {"held", "failed", "half", "open"}
STAGES = {"done", "trouble", "running", "waiting"}
MARKS = {"", "baseline", "ours"}
TOP = {"question": 600, "measure": 600, "outcome": 600, "setup": 2500, "meaning": 1500, "lesson": 600, "log": 6000}


def problems(w):
    out = []

    def text(where, value, limit, need=False):
        if not isinstance(value, str):
            out.append(f"{where}: must be text")
        elif len(value.strip()) > limit:
            out.append(f"{where}: {len(value.strip())} characters, the limit is {limit}")
        elif need and not value.strip():
            out.append(f"{where}: is empty")

    def keys(where, obj, allowed):
        if not isinstance(obj, dict):
            out.append(f"{where}: must be an object")
            return False
        for key in set(obj) - set(allowed):
            out.append(f"{where}: unknown field '{key}'")
        return True

    def items(where, value, most):
        if not isinstance(value, list):
            out.append(f"{where}: must be a list")
            return []
        if len(value) > most:
            out.append(f"{where}: {len(value)} items, the limit is {most}")
        return value

    def checks(where, value):
        for i, c in enumerate(items(where, value, 16)):
            if keys(f"{where}[{i}]", c, ("text", "ok")):
                text(f"{where}[{i}].text", c.get("text", ""), 400, True)
                if not isinstance(c.get("ok"), bool):
                    out.append(f"{where}[{i}].ok: must be true or false")

    if not keys("writeup", w, (*TOP, "parts", "trust", "next", "run")):
        return out
    for key, limit in TOP.items():
        text(key, w.get(key, ""), limit)
    for i, p in enumerate(items("parts", w.get("parts", []), 8)):
        at = f"parts[{i}]"
        if not keys(at, p, ("title", "why", "did", "predictions", "table", "conclusion")):
            continue
        text(f"{at}.title", p.get("title", ""), 200, True)
        for key in ("why", "did", "conclusion"):
            text(f"{at}.{key}", p.get(key, ""), 800)
        for j, pr in enumerate(items(f"{at}.predictions", p.get("predictions", []), 8)):
            if keys(f"{at}.predictions[{j}]", pr, ("text", "result")):
                text(f"{at}.predictions[{j}].text", pr.get("text", ""), 300, True)
                if pr.get("result") not in RESULTS:
                    out.append(f"{at}.predictions[{j}].result: must be one of {sorted(RESULTS)}")
        table = p.get("table")
        if table is not None and keys(f"{at}.table", table, ("head", "rows")):
            head = items(f"{at}.table.head", table.get("head", []), 8)
            if not head:
                out.append(f"{at}.table.head: is empty")
            for cell in head:
                text(f"{at}.table.head", cell, 80)
            for j, row in enumerate(items(f"{at}.table.rows", table.get("rows", []), 24)):
                if not keys(f"{at}.table.rows[{j}]", row, ("cells", "mark")):
                    continue
                cells = row.get("cells", [])
                if not isinstance(cells, list) or len(cells) != len(head):
                    out.append(f"{at}.table.rows[{j}]: {len(cells) if isinstance(cells, list) else '?'} cells, the table has {len(head)} columns")
                    continue
                for cell in cells:
                    text(f"{at}.table.rows[{j}]", cell, 80)
                if row.get("mark", "") not in MARKS:
                    out.append(f"{at}.table.rows[{j}].mark: must be 'baseline', 'ours' or ''")
    checks("trust", w.get("trust", []))
    for i, n in enumerate(items("next", w.get("next", []), 6)):
        if keys(f"next[{i}]", n, ("title", "why")):
            text(f"next[{i}].title", n.get("title", ""), 200, True)
            text(f"next[{i}].why", n.get("why", ""), 400)
    run = w.get("run", {})
    if keys("run", run, ("state", "line", "watch", "command", "stages", "housekeeping", "files")):
        text("run.state", run.get("state", ""), 120)
        text("run.line", run.get("line", ""), 300)
        text("run.command", run.get("command", ""), 600)
        for i, t in enumerate(items("run.watch", run.get("watch", []), 8)):
            if keys(f"run.watch[{i}]", t, ("title", "note", "href")):
                text(f"run.watch[{i}].title", t.get("title", ""), 80, True)
                text(f"run.watch[{i}].note", t.get("note", ""), 160)
                href = t.get("href", "")
                if href and not href.startswith(("http://", "https://")):
                    out.append(f"run.watch[{i}].href: must be an http(s) address or empty")
        for i, s in enumerate(items("run.stages", run.get("stages", []), 24)):
            if keys(f"run.stages[{i}]", s, ("name", "note", "state")):
                text(f"run.stages[{i}].name", s.get("name", ""), 80, True)
                text(f"run.stages[{i}].note", s.get("note", ""), 300)
                if s.get("state") not in STAGES:
                    out.append(f"run.stages[{i}].state: must be one of {sorted(STAGES)}")
        checks("run.housekeeping", run.get("housekeeping", []))
        for i, f in enumerate(items("run.files", run.get("files", []), 24)):
            if keys(f"run.files[{i}]", f, ("path", "note")):
                text(f"run.files[{i}].path", f.get("path", ""), 300, True)
                text(f"run.files[{i}].note", f.get("note", ""), 160)
    return out


def call(method, path, body=None):
    request = urllib.request.Request(URL + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as error:
        answer = json.load(error)
    if not answer.get("success"):
        raise SystemExit(f"The journal refused: {answer.get('error')}")
    return answer


def load(number):
    path = DIR / f"{number}.json"
    if not path.exists():
        raise SystemExit(f"No write-up file {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: not valid JSON ({error})")


def numbers(args):
    return args or sorted(p.stem for p in DIR.glob("[0-9]*.json"))


def main(argv):
    if len(argv) < 1:
        raise SystemExit(__doc__)
    command, args = argv[0], argv[1:]
    if command in ("check", "put"):
        chosen = numbers([] if args == ["--all"] else args)
        if command == "put" and not args:
            raise SystemExit("Name the experiments to upload, or pass --all.")
        bad = 0
        for number in chosen:
            found = problems(load(number))
            for problem in found:
                print(f"{number}: {problem}")
            bad += bool(found)
        if bad:
            raise SystemExit(f"{bad} of {len(chosen)} write-ups have problems; nothing was uploaded.")
        if command == "check":
            print(f"{len(chosen)} write-ups are valid.")
            return
        for number in chosen:
            call("PUT", "/api/experiments/writeup", {"id": number, "writeup": load(number)})
            print(f"{number}: uploaded")
    elif command == "get":
        print(json.dumps(call("GET", f"/api/experiments/writeup?id={args[0]}")["data"], indent=2, ensure_ascii=False))
    elif command == "record":
        for e in call("GET", "/api/experiments")["data"]["experiments"]:
            print(e["id"], e["when"], e["direction"], e["status"], "|", e["question"])
    elif command == "add":
        created = call("POST", "/api/experiments", {"op": "create", "experiment": json.loads(Path(args[0]).read_text())})["experiment"]
        print(f"Created experiment {created['id']}: {created['question']}")
    elif command == "update":
        call("POST", "/api/experiments", {"op": "update", "id": args[0], "patch": json.loads(Path(args[1]).read_text())})
        print(f"{args[0]}: record updated")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
