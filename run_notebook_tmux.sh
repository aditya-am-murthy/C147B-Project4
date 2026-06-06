#!/usr/bin/env bash
# Execute rl.ipynb cell-by-cell in tmux (skip Colab cells), save checkpoint after each cell.
#
# Usage:
#   ./run_notebook_tmux.sh              # tmux: run notebook + log pane
#   ./run_notebook_tmux.sh run          # execute cells (foreground)
#   ./run_notebook_tmux.sh attach       # attach tmux session
#   ./run_notebook_tmux.sh kill         # stop tmux session
#
#   RUN_ONLY=38              animation only (loads checkpoint; no retrain)
#   SKIP_TRAIN=1             skip cells 24 & 35 (.train); use 36/38 to load weights
#   SKIP_WANDB=1             skip wandb cells 21-23 (default: run them)
#
# For DQN training, prefer:  ./run_dqn_train.sh
# Tmux detach: Ctrl+b  then  d  (NOT Ctrl+o). Foreground run blocks the terminal.
#
# IMPORTANT: This runner uses its own kernel (nbclient). Cursor/VS Code will NOT
# show rl.ipynb cells as [*] running. Progress is in:
#   tail -f .notebook_autosaves/run_log_latest.txt
#   cat .notebook_autosaves/live_status.json
# To see [*] in the notebook UI, run cell 24 inside Cursor (or use ./run_dqn_train.sh + tail).

set -eo pipefail

SESSION_NAME="${SESSION_NAME:-dqn-notebook}"
CONDA_ENV="${CONDA_ENV:-C147B-4}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK_DIR="${NOTEBOOK_DIR:-$PROJECT_DIR/DQN}"
NOTEBOOK="${NOTEBOOK:-rl.ipynb}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/.notebook_autosaves}"
SCRIPT="$PROJECT_DIR/run_notebook_tmux.sh"
EXECUTOR="$PROJECT_DIR/execute_notebook_cells.py"
RUN_LOG="${RUN_LOG:-$BACKUP_DIR/run_log_latest.txt}"

log() { printf '[notebook-runner] %s\n' "$*"; }

conda_activate() {
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh" >&2
    exit 1
  fi
  conda activate "$CONDA_ENV"
}

setup_project4_path() {
  local parent repo link
  repo="$PROJECT_DIR"
  parent="$(dirname "$repo")"
  link="$parent/Project4"
  [[ -e "$link" ]] || ln -sf "$repo" "$link"
  export PYTHONPATH="$parent${PYTHONPATH:+:$PYTHONPATH}"
}

ensure_executor_deps() {
  python3 -c "import nbclient, nbformat" 2>/dev/null || pip install -q nbclient nbformat
}

cmd_run() {
  set +e
  set +u
  conda_activate
  setup_project4_path
  cd "$NOTEBOOK_DIR"
  mkdir -p "$BACKUP_DIR"
  local log_ts="$BACKUP_DIR/run_log_$(date '+%Y%m%d_%H%M%S').txt"
  export BACKUP_DIR PROJECT_DIR NOTEBOOK_DIR LOG_FILE="$log_ts" PYTHONUNBUFFERED=1
  [[ -n "${START_CELL:-}" ]] && export START_CELL
  ensure_executor_deps
  log "Executing $NOTEBOOK cell-by-cell (skipping Colab cells)"
  log "Checkpoints -> $BACKUP_DIR/checkpoints/cell_NNN/"
  log "Output log    -> $log_ts (also copied to $RUN_LOG)"
  {
    echo "=== run started $(date) ==="
    python3 -u "$EXECUTOR"
    ec=$?
    echo "=== run ended $(date), exit code $ec ==="
    exit $ec
  } 2>&1 | tee "$RUN_LOG"
  cp -f "$log_ts" "$RUN_LOG" 2>/dev/null || true
}

cmd_kill() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
    echo "Killed tmux session '$SESSION_NAME'"
  else
    echo "No session '$SESSION_NAME'"
  fi
}

cmd_start() {
  command -v tmux >/dev/null || { echo "tmux not installed" >&2; exit 1; }
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Run: $0 attach  or  $0 kill"
    exit 1
  fi
  chmod +x "$SCRIPT"
  tmux new-session -d -s "$SESSION_NAME" -n run "bash '$SCRIPT' run"
  tmux new-window -t "$SESSION_NAME" -n log "tail -f '$RUN_LOG'"
  echo "Started tmux session '$SESSION_NAME' (windows: run, log)"
  echo "  attach:  tmux attach -t $SESSION_NAME   (Ctrl+b n switches run/log windows)"
  echo "  detach:  Ctrl+b  then  d"
  echo "  log:     tail -f $RUN_LOG"
  echo "  status:  cat $BACKUP_DIR/live_status.json"
  echo "  stop:    $0 kill"
  echo "  DQN train (recommended): ./run_dqn_train.sh"
  echo ""
  echo "  Cursor will NOT show cells as running — this is a background kernel, not the IDE."
  if [[ -t 1 ]]; then
    tmux attach -t "$SESSION_NAME"
  else
    echo "Attach from a terminal: tmux attach -t $SESSION_NAME"
  fi
}

cmd_attach() {
  tmux attach -t "$SESSION_NAME"
}

main() {
  case "${1:-start}" in
    run)    cmd_run ;;
    kill)   cmd_kill ;;
    attach) cmd_attach ;;
    start|"") cmd_start ;;
    *)
      echo "Usage: $0 [start|run|attach|kill]" >&2
      exit 1
      ;;
  esac
}

main "$@"
