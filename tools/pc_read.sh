#!/usr/bin/env bash
# Run one shell command on this machine with nothing writable, for a reader that must not change anything
# (OpenClaw analysing an experiment, tools/queue_analysis.sh). Usage: tools/pc_read.sh '<command line>'
#
# The command runs in a sandbox (bubblewrap) that it cannot leave:
#   - every file it can see is read-only; /tmp is an empty scratch that is thrown away when it exits;
#   - it sees this repository (code, notes, artifacts, git history), the project's memory notes, the lerobot fork
#     and the Python environments, and the system's own files. The rest of the home folder (ssh keys, logins, tokens, other
#     projects), the Windows drives under /mnt and /run are not there. PC_READ_EXTRA=dir1:dir2 adds folders;
#   - no devices: no GPU (/dev/dxg) and no serial port, so nothing in it can train on the GPU or move the arm;
#   - no network, no task spooler, no way to start Windows programs;
#   - it is stopped after PC_READ_TIMEOUT seconds (default 120) and its output is cut at 400 kB.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: tools/pc_read.sh '<command line>'" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM="$HOME/.claude/projects/-home-win10ubuntu-projects-SO-ARM-101/memory"
binds=(--ro-bind "$ROOT" "$ROOT")
[[ -d "$MEM" ]] && binds+=(--ro-bind "$MEM" "$MEM")
[[ -d "$HOME/miniforge3" ]] && binds+=(--ro-bind "$HOME/miniforge3" "$HOME/miniforge3")
[[ -d "$HOME/projects/lerobot-fork" ]] && binds+=(--ro-bind "$HOME/projects/lerobot-fork" "$HOME/projects/lerobot-fork")
IFS=: read -ra extra <<< "${PC_READ_EXTRA:-}"
for d in "${extra[@]}"; do [[ -n "$d" && -d "$d" ]] && binds+=(--ro-bind "$d" "$d"); done
# Files in the repository that hold secrets are shown empty.
while IFS= read -r -d '' secret; do binds+=(--ro-bind /dev/null "$secret"); done < <(
  find "$ROOT" -maxdepth 4 \( -name node_modules -o -name .git -o -name artifacts -o -name wandb \) -prune -o -type f \
    \( -name '.env' -o -name '.env.*' -o -name '*.pem' -o -name '*.key' -o -name '.netrc' -o -name 'credentials*.json' -o -name 'id_rsa*' -o -name 'id_ed25519*' \) -print0 2>/dev/null)
exec bwrap --ro-bind / / --tmpfs /mnt --tmpfs /run --tmpfs /tmp --tmpfs /var/tmp --tmpfs "$HOME" "${binds[@]}" \
  --dev /dev --proc /proc --unshare-all --die-with-parent --new-session --clearenv \
  --setenv HOME "$HOME" --setenv PATH "/usr/local/bin:/usr/bin:/bin" --setenv LANG C.UTF-8 --setenv GIT_OPTIONAL_LOCKS 0 \
  --chdir "$ROOT" -- /usr/bin/timeout -k 5 "${PC_READ_TIMEOUT:-120}" /bin/bash -c 'set -o pipefail; { eval "$1"; } 2>&1 | head -c 400000' pc_read "$*"
