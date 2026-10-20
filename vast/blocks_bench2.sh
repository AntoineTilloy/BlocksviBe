#!/usr/bin/env bash
# Run ON the instance after vast/blocks_bench.sh: GPU recursion variants (CUDA-core kernels vs RNS
# GEMMs) at larger batches, then the blocks_3d stage split (vast/blocks_stages.sh).
# Appends to results/blocks_bench2.log.
cd ~/SDPviBe
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. ~/venv/bin/activate
LOG=results/blocks_bench2.log
echo "[$(date +%T)] GPU recursion variants" >> $LOG
for v in "auto cuda" "cuda cuda" "rns gemm"; do set -- $v
    CAUCHY=$1 POLE=$2 python tools/blocks/bench_recursion.py 8 8 60 30 29 24 64 >> $LOG 2>&1
done
for v in "auto cuda" "cuda cuda" "rns gemm"; do set -- $v
    CAUCHY=$1 POLE=$2 python tools/blocks/bench_recursion.py 8 8 120 60 29 2 6 >> $LOG 2>&1
done
CAUCHY=auto POLE=cuda python tools/blocks/bench_recursion.py 8 8 120 60 18 2 8 >> $LOG 2>&1
echo "[$(date +%T)] stages" >> $LOG
bash vast/blocks_stages.sh
echo "[$(date +%T)] DONE2" >> $LOG
