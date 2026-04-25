#!/bin/bash
# Deploy latest code to the IronClaw server.
# Usage: ./deploy.sh

KEY=~/Downloads/agent-private-key.pem
HOST=agent@baremetal0.agents.near.ai
PORT=23031
REMOTE=~/workspace/glassbox

SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no"
SCP="scp -i $KEY -P $PORT -o StrictHostKeyChecking=no"

set -e

echo "==> Packing..."
tar --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='.git'  --exclude='.claude' \
    -czf /tmp/glassbox_deploy.tar.gz .

echo "==> Uploading..."
$SCP /tmp/glassbox_deploy.tar.gz $HOST:~/workspace/

echo "==> Deploying on server..."
$SSH $HOST "
  cd ~/workspace
  rm -rf glassbox_new
  mkdir glassbox_new
  tar -xzf glassbox_deploy.tar.gz -C glassbox_new
  # carry the venv over from the existing install
  if [ -d glassbox/.venv ]; then
    mv glassbox/.venv glassbox_new/.venv
  fi
  # carry over any user-uploaded CSV files (not in the repo)
  # check current dir and previous backup, so CSVs survive across deploys
  for f in glassbox/*.csv glassbox_old/*.csv; do
    [ -f "$f" ] && cp "$f" glassbox_new/ 2>/dev/null || true
  done
  # swap atomically
  rm -rf glassbox_old
  mv glassbox glassbox_old 2>/dev/null || true
  mv glassbox_new glassbox
  # reinstall package in-place (keeps existing venv)
  cd glassbox
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  .venv/bin/pip install -e . --quiet
  echo 'Deploy done.'
"
