#!/bin/bash 

read -sp "Build (1) or Run (2) container?" sel

if [ $sel -eq 1 ]; then
    echo "Building Image"
    # Build the Docker image
    podman build -t whisper-v2t-server .
    echo "running container..."
    # Run the container
    podman run -d -p 5000:5000 --name whisper-server whisper-v2t-server
elif [ $sel -eq 2 ]; then
    echo "running container..."
    # Run the container
    podman run -d -p 5000:5000 --name whisper-server whisper-v2t-server
else
    echo "invalid choice!"
fi



