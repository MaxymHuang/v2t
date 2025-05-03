#!/bin/bash

# Ask the user which server to run
echo "Select server to run:"
echo "1. Voice-to-Text Server"
echo "2. Simple Test Server"
read -p "Enter choice (1 or 2): " sel

if [ $sel -eq 1 ]; then
    echo "Starting Voice-to-Text Server..."
    python v2t_server.py
elif [ $sel -eq 2 ]; then
    echo "Starting Simple Test Server..."
    python test_server.py
else
    echo "Invalid selection!"
fi
