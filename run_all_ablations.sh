#!/bin/bash

echo "Starting Vision Ablations..."
uv run python run_vision_ablations.py
if [ $? -ne 0 ]; then
    echo "Vision ablations failed"
    exit 1
fi

echo "Starting Reasoning Ablations..."
uv run python run_reasoning_ablations.py
if [ $? -ne 0 ]; then
    echo "Reasoning ablations failed"
    exit 1
fi

echo "Starting Audio Ablations..."
uv run python run_audio_ablations.py
if [ $? -ne 0 ]; then
    echo "Audio ablations failed"
    exit 1
fi

echo "All ablations completed successfully!"
