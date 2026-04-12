#!/bin/bash
set -e
echo "=== input ==="
find /input -type f | head -20
echo "=== rodando inferência ==="
cd /app
python src/infer.py
echo "=== output ==="
ls -lh "${OUTPUT_DIR:-/output}/"