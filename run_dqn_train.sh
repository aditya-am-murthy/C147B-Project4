#!/usr/bin/env bash
# Run DQN training outside the notebook (recommended for long training).
#
# Usage:
#   ./run_dqn_train.sh              # tmux session, detach with Ctrl+b then d
#   ./run_dqn_train.sh fg           # foreground (blocks terminal)
#   EPISODES=5 ./run_dqn_train.sh   # quick test
#
# Log: .notebook_autosaves/dqn_train.log

set -eo pipefail

SESSION_NAME="${SESSION_NAME:-dqn-train}"
CONDA_ENV="${CONDA_ENV:-C147B-4}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK_DIR="$PROJECT_DIR/DQN"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/.notebook_autosaves/dqn_train.log}"
EPISODES="${EPISODES:-200}"
VALIDATE_EVERY="${VALIDATE_EVERY:-50}"
SCRIPT="$PROJECT_DIR/run_dqn_train.sh"

conda_activate() {
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
}

setup_path() {
  local parent
  parent="$(dirname "$PROJECT_DIR")"
  [[ -e "$parent/Project4" ]] || ln -sf "$PROJECT_DIR" "$parent/Project4"
  export PYTHONPATH="$parent${PYTHONPATH:+:$PYTHONPATH}"
}

cmd_train() {
  set +u
  conda_activate
  setup_path
  mkdir -p "$(dirname "$LOG_FILE")"
  cd "$NOTEBOOK_DIR"

  {
    echo "=== DQN train started $(date) ==="
    echo "episodes=$EPISODES validate_every=$VALIDATE_EVERY device=cuda"
    python3 -u <<PY
import gymnasium as gym
import DQN, model, utils, env_wrapper

env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
trainer = DQN.DQN(
    env_wrapper.EnvWrapper(env),
    model.Nature_Paper_Conv,
    lr=0.00025,
    gamma=0.95,
    buffer_size=100000,
    batch_size=32,
    loss_fn="mse_loss",
    use_wandb=False,
    device="cuda",
    seed=42,
    epsilon_scheduler=utils.exponential_decay(1, 700, 0.1),
    save_path=utils.get_save_path("DQN", "./runs/"),
)
print("Training started...", flush=True)
trainer.train($EPISODES, $VALIDATE_EVERY, 30, 50, 50)
print("Training finished.", flush=True)
PY
    ec=$?
    echo "=== DQN train ended $(date), exit $ec ==="
    exit $ec
  } 2>&1 | tee -a "$LOG_FILE"
}

cmd_tmux() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' exists. Attach: tmux attach -t $SESSION_NAME"
    exit 1
  fi
  chmod +x "$SCRIPT"
  tmux new-session -d -s "$SESSION_NAME" "bash '$SCRIPT' fg"
  echo "Started training in tmux '$SESSION_NAME'"
  echo "  attach:  tmux attach -t $SESSION_NAME"
  echo "  detach:  Ctrl+b  then  d   (not Ctrl+o)"
  echo "  log:     tail -f $LOG_FILE"
  echo "  stop:    tmux kill-session -t $SESSION_NAME"
}

case "${1:-tmux}" in
  fg) cmd_train ;;
  tmux|"") cmd_tmux ;;
  *)
    echo "Usage: $0 [tmux|fg]" >&2
    exit 1
    ;;
esac
