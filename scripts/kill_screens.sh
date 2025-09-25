#!/bin/bash
# kill_screens.sh - Kill all screen sessions with a given name (with confirmation)

if [ $# -ne 1 ]; then
  echo "Usage: $0 session_name"
  exit 1
fi

SESSION_NAME="$1"

# Find matching screen sessions
SESSIONS=$(screen -ls | grep "$SESSION_NAME" | awk '{print $1}')

if [ -z "$SESSIONS" ]; then
  echo "No screen sessions found with name: $SESSION_NAME"
  exit 0
fi

echo "Found the following screen sessions matching '$SESSION_NAME':"
echo "$SESSIONS"
echo

# Ask for confirmation (default = yes)
read -p "Do you want to kill these sessions? [Y/n]: " answer
answer=${answer:-Y}  # Default to Y if user just presses Enter

if [[ "$answer" =~ ^[Yy]$ ]]; then
  echo "Killing screen sessions..."
  for s in $SESSIONS; do
    echo "Killing $s"
    screen -S "$s" -X quit
  done
  echo "All matching screen sessions killed."
else
  echo "Aborted. No sessions were killed."
fi
