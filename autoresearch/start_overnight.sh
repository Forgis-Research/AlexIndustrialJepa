#!/bin/bash
# Start overnight Many-to-1 transfer learning research
# Run this before starting Claude agent

set -e

echo "=============================================="
echo "Many-to-1 Transfer Learning Overnight Research"
echo "=============================================="

cd ~/IndustrialJEPA

# Pull latest
echo ""
echo "Pulling latest code..."
git pull

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Check memory
echo ""
echo "Memory Status:"
free -h

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q phmd ucimlrepo

# Run setup validation
echo ""
echo "=============================================="
echo "Validating Datasets..."
echo "=============================================="
python autoresearch/experiments/00_setup_datasets.py

# Show objectives
echo ""
echo "=============================================="
echo "Current Objectives:"
echo "=============================================="
cat autoresearch/OBJECTIVES_STATUS.md | head -50

echo ""
echo "=============================================="
echo "Ready for overnight research!"
echo "=============================================="
echo ""
echo "TRACK 1 (Bearings): Validate approach on clean data"
echo "  Sources: CWRU + PHM2012 + XJTU-SY"
echo "  Target:  Paderborn (zero-shot)"
echo "  Goal:    Accuracy >= 80%"
echo ""
echo "TRACK 2 (Robots): Novel contribution"
echo "  Sources: 4 robot datasets"
echo "  Target:  1 held-out robot"
echo "  Goal:    Avg AUC >= 0.60"
echo ""
echo "Next: Start Claude agent with:"
echo "  claude --dangerously-skip-permissions"
echo ""
echo "Then paste prompt from:"
echo "  autoresearch/MANY_TO_ONE_PROMPT.md"
echo ""
