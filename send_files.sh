#!/bin/bash

# Define remote destination
REMOTE_USER="vishnu04"
REMOTE_HOST="xlogin0.comp.nus.edu.sg"
REMOTE_DIR="nlp"

# Use a loop to go through all files in the current directory
for file in *; do
    # Check if it's a file (not a directory)
    if [[ -f "$file" ]]; then
        scp "$file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
    fi
done
