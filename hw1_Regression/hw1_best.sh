#!/usr/bin/env bash
set -e
INPUT="$1"
OUTPUT="$2"
python3 src/predict.py --model ./models/best.npy --test "$INPUT" --out "$OUTPUT"
