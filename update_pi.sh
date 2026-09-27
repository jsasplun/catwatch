cd ~/git/cat-bowl-monitor
rsync -av \
  --exclude '.git' --exclude '.dvc' --exclude '.venv*' --exclude '__pycache__' \
  --exclude 'data/' --exclude 'runs/' --exclude 'models/' \
  --exclude 'delete_me/' --exclude '.env' \
  --exclude 'update_pi.sh' --exclude '.conda*' \
  ./ $PI:~/cat-bowl-monitor/