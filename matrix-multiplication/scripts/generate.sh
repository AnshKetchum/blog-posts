#!/bin/bash
set -e

# Check for input arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <M> <N> <K>"
    echo "  M: rows of A (and rows of C)"
    echo "  N: cols of A (and rows of B) - inner dimension"
    echo "  K: cols of B (and cols of C)"
    exit 1
fi

M=$1
N=$2
K=$3

# Generate A.txt (M rows, N cols)
python3 generate_matrix.py $M $N "A.txt"

# Generate B.txt (N rows, K cols)
python3 generate_matrix.py $N $K "B.txt"

echo "Generated matrices for multiplication:"
echo "  A.txt: ${M}x${N}"
echo "  B.txt: ${N}x${K}"
echo "  C (output) will be: ${M}x${K}"