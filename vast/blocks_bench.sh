#!/usr/bin/env bash
# Run ON a rented GPU instance (detached) from ~/BlocksviBe, next to ~/SDPviBe (push both: SDPviBe's
# vast/sync.sh, and rsync this repository to ~/BlocksviBe):
#   ssh ... 'cd BlocksviBe && (setsid nohup bash vast/blocks_bench.sh < /dev/null > /dev/null 2>&1 &)'
# Benchmarks for docs/BLOCKS_PLAN.md:
#   1. GPU: multiprecision GEMM rates at the blocks-recursion shapes, batched GPU recursion
#      (synthetic operator of the shapes of real blocks_3d tasks);
#   2. CPU: the official blocks_3d image (musl) on the same tasks, default allocator vs jemalloc.
# Results are appended to results/blocks_bench.log as they come (pull them periodically); at the end
# results/DONE, then wait (<= 1 h) for the client's `touch results/PULLED`, then self-destroy.
# Hard deadline DEADLINE_H (derive it from the budget: budget / (dph + disk $/h), with margin).
cd ~/BlocksviBe
DEADLINE_H=${DEADLINE_H:-3}
mkdir -p results
LOG=results/blocks_bench.log
selfkill() {
    echo "$(date) shutting down ($1)" >> $LOG
    pip install -q --break-system-packages vastai > /dev/null 2>&1
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
WATCHDOG=$!
log() { echo "[$(date +%T)] $*" >> $LOG; }

log "setup"; bash ~/SDPviBe/vast/remote_setup.sh >> $LOG 2>&1
lscpu | grep -E "Model name|^CPU\(s\)" >> $LOG; free -g | head -2 >> $LOG
export PYTHONPATH=$PWD:$HOME/SDPviBe PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. ~/venv/bin/activate

log "GPU: GEMM shapes, 926-bit"
python tools/bench_shapes.py 29 >> $LOG 2>&1
log "GPU: batched recursion, synthetic operator (2j12=2j43=8)"
python tools/bench_recursion.py 8 8 60 30 18 6 24 64 >> $LOG 2>&1
python tools/bench_recursion.py 8 8 60 30 29 6 24 >> $LOG 2>&1
python tools/bench_recursion.py 8 8 120 60 18 1 4 >> $LOG 2>&1
python tools/bench_recursion.py 8 8 120 60 29 1 2 >> $LOG 2>&1

log "CPU: official blocks_3d image"
[ -x tools/b3d_root/usr/local/bin/blocks_3d ] || python3 tools/pull_image.py bootstrapcollaboration/blocks_3d master tools/b3d_root >> $LOG 2>&1
mkdir -p tools/apk && curl -s https://dl-cdn.alpinelinux.org/alpine/v3.18/main/x86_64/jemalloc-5.3.0-r2.apk -o tools/apk/j.apk \
    && tar -xzf tools/apk/j.apk -C tools/apk 2>/dev/null
R=$PWD/tools/b3d_root; JE=$PWD/tools/apk/usr/lib/libjemalloc.so.2
b3d() {   # b3d <label> <alloc: default|jemalloc> args...
    local lab=$1 al=$2; shift 2
    local P=""; [ $al = jemalloc ] && P="--preload $JE"
    local O=/tmp/b3d_$lab; rm -rf $O; mkdir -p $O
    local t0=$(date +%s.%N)
    timeout ${TMAX:-1800} $R/lib/ld-musl-x86_64.so.1 --library-path "$R/usr/local/lib:$R/usr/lib:$R/lib" $P \
        $R/usr/local/bin/blocks_3d "$@" -o $O/spin_{}.json > $O.log 2>&1
    local rc=$? t1=$(date +%s.%N)
    log "blocks_3d $lab alloc=$al rc=$rc wall $(python3 -c "print(round($t1-$t0,1))") s; profile: $(python3 tools/b3d_profile.py $O/spin_0.json.profile 2>&1)"
}
T=$(nproc)
A30="--j-external=2,2,2,2 --j-internal=0,2,4-30 --j-12=4 --j-43=4 --delta-12=0 --delta-43=0 --delta-1-plus-2=6 --four-pt-struct=0,0,2,2 --four-pt-sign=1 --order=30 --lambda=11 --coordinates=xt --kept-pole-order=30 --precision=512"
A60="--j-external=2,2,2,2 --j-internal=0,2,4-60 --j-12=4 --j-43=4 --delta-12=0 --delta-43=0 --delta-1-plus-2=6 --four-pt-struct=0,0,2,2 --four-pt-sign=1 --order=60 --lambda=19 --coordinates=xt_radial --kept-pole-order=60 --precision=512"
b3d o30_je_$T jemalloc $A30 --num-threads=$T
b3d o60_je_$T jemalloc $A60 --num-threads=$T
b3d o60_je_8 jemalloc $A60 --num-threads=8
TMAX=1500 b3d o30_musl_$T default $A30 --num-threads=$T

log "DONE"
touch results/DONE
for i in $(seq 1 60); do [ -f results/PULLED ] && break; sleep 60; done
kill $WATCHDOG 2>/dev/null
selfkill "finished"
