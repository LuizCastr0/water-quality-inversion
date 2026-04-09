# fase2/run.sh
#!/bin/bash
set -e
echo "=== input ==="
find /input -type f | head -20
echo "=== rodando inferência ==="
python src/infer.py
echo "=== output ==="
ls -lh /output/