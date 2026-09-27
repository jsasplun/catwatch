#!/usr/bin/env bash
set -euo pipefail
# Where: WSL
ssh "$PI" "cd ~/cat-bowl-monitor && source .venv/bin/activate && python -m catwatch.check_crop"
mkdir -p ~/git/cat-bowl-monitor/data
scp "$PI:~/cat-bowl-monitor/data/crop_check_*.jpg" ~/git/cat-bowl-monitor/data/
