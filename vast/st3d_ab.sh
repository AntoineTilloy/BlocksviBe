#!/usr/bin/env bash
# End-to-end A/B validation of the drop-in on a real stress-tensor point (run ON a GPU instance, detached):
#   ssh ... 'cd BlocksviBe && (DEADLINE_H=<h> setsid nohup bash vast/st3d_ab.sh < /dev/null > /dev/null 2>&1 &)'
# Needs ~/SDPviBe (its vast/sync.sh push) and ~/BlocksviBe (tar of this repository) on the instance.
#   1. setup as SDPviBe/vast/st3d_point.sh: python env, official SDPB, stress-tensors-3d (hyperion) with its
#      sdpb call parked, official blocks_3d image + jemalloc; b3d_setup built from the blocks_3d sources
#   2. A: hyperion program $PROG with the official blocks_3d -> PMP A (a sample of block tables kept)
#   3. B: the same program, blocks_3d.sh -> BlocksviBe's drop-in (official binary as fallback) -> PMP B
#   4. block tables A vs B (sample), PMP A vs B (tools/compare_pmp.py)
#   5. pmp2sdp both; SDPviBe (GPU) on both; official SDPB (+ jemalloc, all cores) on B
# results/st3d_ab.log: one line per step.  At the end DONE, then wait (<= HOLD_H h) for results/PULLED,
# then self-destroy.  Hard deadline DEADLINE_H.
set -u
cd ~/BlocksviBe
DEADLINE_H=${DEADLINE_H:-6}; HOLD_H=${HOLD_H:-1}
PROG=${PROG:-TSigEps_test_nmax6}
ST3D_COMMIT=${ST3D_COMMIT:-21dfcc0}
NP=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)     # MPI ranks: physical cores
[ -n "${CONTAINER_API_KEY:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ | grep '^CONTAINER_\(ID\|API_KEY\)=' | sed 's/^/export /')"
mkdir -p results
LOG=$PWD/results/st3d_ab.log
log() { echo "$(date +%T) $*" >> $LOG; }
selfkill() {
    log "shutting down ($1)"
    pip install -q --break-system-packages vastai > /dev/null 2>&1
    vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" -y >> $LOG 2>&1 \
      || vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
}
( sleep $((DEADLINE_H * 3600)); selfkill "deadline ${DEADLINE_H}h" ) &
log "start: $NP cores ($(nproc) threads), $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -h ~ | awk 'NR==2{print $4}') free disk, $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ---------------------------------------------------------------- 1. setup
# resume: SKIP_SETUP=1 (all of section 1 done), SKIP_A=1 with OBJ_A, NSV_A (run A done)
t0=$(date +%s)
if [ "${SKIP_SETUP:-0}" != 1 ]; then
cd ~/SDPviBe
bash vast/remote_setup.sh > ~/BlocksviBe/results/setup_python.log 2>&1 || log "FAILED python setup: $(tail -2 ~/BlocksviBe/results/setup_python.log)"
bash vast/remote_sdpb_setup.sh > ~/BlocksviBe/results/setup_sdpb.log 2>&1 || log "FAILED sdpb setup: $(tail -2 ~/BlocksviBe/results/setup_sdpb.log)"
[ -d ext/stress-tensors-3d ] || { git clone -q https://gitlab.com/davidsd/stress-tensors-3d ext/stress-tensors-3d && git -C ext/stress-tensors-3d checkout -q $ST3D_COMMIT; }
SDPB_WRAPPER=block bash vast/st3d_setup.sh > ~/BlocksviBe/results/setup_st3d.log 2>&1 || log "FAILED st3d setup: $(tail -2 ~/BlocksviBe/results/setup_st3d.log)"
rm -f /lib/ld-musl-x86_64.so.1; rm -rf ~/.stack/programs/x86_64-linux/ghc-musl*     # stack must not see musl
( cd ext/stress-tensors-3d && time bash ./build.sh ) > ~/BlocksviBe/results/build.log 2>&1
ln -sf ~/SDPviBe/tools/sdpb_root/lib/ld-musl-x86_64.so.1 /lib/ld-musl-x86_64.so.1
EXE=$(find ~/SDPviBe/ext/stress-tensors-3d/.stack-work -type f -name stress-tensors-3d-exec -perm -u+x | head -1)
[ -x "$EXE" ] || { log "FAILED build: $(tail -5 ~/BlocksviBe/results/build.log | tr '\n' ' ')"; selfkill "build failed"; exit 1; }
bash vast/fake_slurm.sh > ~/BlocksviBe/results/fake_slurm.log 2>&1 || log "FAILED fake slurm"
cd ~/BlocksviBe
# b3d_setup from the blocks_3d sources (the image's binary is a static musl build without them)
apt-get install -y -qq libboost-dev libgmp-dev libeigen3-dev > results/apt_b3d.log 2>&1
[ -d ext/blocks_3d ] || git clone -q https://gitlab.com/bootstrapcollaboration/blocks_3d ext/blocks_3d
log "blocks_3d sources at $(git -C ext/blocks_3d log --oneline -1)"
make -C csrc -j$(nproc) B3D=$PWD/ext/blocks_3d ENV=/usr > results/build_b3d_setup.log 2>&1 || log "FAILED b3d_setup: $(grep -m3 error results/build_b3d_setup.log | tr '\n' ' ')"
fi
EXE=$(find ~/SDPviBe/ext/stress-tensors-3d/.stack-work -type f -name stress-tensors-3d-exec -perm -u+x | head -1)
cd ~/BlocksviBe
. ~/venv/bin/activate
python3 -c "import gmpy2" 2> /dev/null || python3 -m pip install -q gmpy2 > /dev/null 2>&1
PY=$(command -v python3)
log "setup done in $(( ($(date +%s) - t0) / 60 )) min (exe $EXE, python $PY)"

ST=~/SDPviBe/ext/stress-tensors-3d
C=$ST/hyperion-config/vast
B3=~/SDPviBe/tools/b3d_root/usr/local/bin/blocks_3d
JE=~/SDPviBe/tools/apk/usr/lib/libjemalloc.so.2
R=~/SDPviBe/tools/sdpb_root
export OPAL_PREFIX=$R/usr OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPI="$R/usr/bin/mpirun --allow-run-as-root"
cp $C/blocks_3d.sh results/blocks_3d_official.sh
# hyperion's pmp2sdp call: OpenMPI aborts under the fake Slurm environment (SLURM_TASKS_PER_NODE missing);
# not needed here (the PMP is converted below) but kept working
for s in pmp2sdp srun_pmp2sdp; do grep -q '^for v in' $C/$s.sh || sed -i '2i for v in $(env | grep -o "^SLURM[A-Z_]*"); do unset $v; done' $C/$s.sh; done

stop_hyperion() {
    for p in $(pgrep -f stress-tensors-3d-exec); do kill $p 2>/dev/null; done
    pkill -f '^sleep infinity$' 2>/dev/null
    for p in $(pgrep -f 'b3d_root/usr/local/bin/blocks_3d') $(pgrep -f 'python3 -m blocksvibe') $(pgrep -x pmp2sdp); do kill $p 2>/dev/null; done
    sleep 5
}

# one hyperion run of $PROG until its PMP is written (hyperion's own pmp2sdp / sdpb calls are not needed:
# they are run below); sets OBJ, NSV
run_point() {   # label
    local L=$1
    local before=$(ls /root/hyp/data/*/ 2>/dev/null | sort)
    local t0=$(date +%s) id="" ok=0 logs="" last=$(date +%s) n n_prev=""
    ( cd $ST && setsid nohup $EXE master -p $PROG > ~/BlocksviBe/results/${L}_master.out 2>&1 < /dev/null & )
    while true; do
        sleep 20
        if [ -z "$id" ]; then
            id=$(comm -13 <(echo "$before") <(ls /root/hyp/data/*/ 2>/dev/null | sort) | grep -v '^$' | head -1)
            logs=$id
        fi
        # "SDP Part's", not "SDP Part": "Finished computing SDP PartChunk's" comes before the PMP is written
        [ -n "$id" ] && grep -qs "Finished computing SDP Part's" -r $(for g in $logs; do echo /root/hyp/logs/*/$g; done) && { ok=1; break; }
        pgrep -f stress-tensors-3d-exec > /dev/null || break
        # hyperion can lose a worker (deployed, connected, never reports) and then waits forever: after
        # STALL_MIN quiet minutes, restart the master from the run's database (finished tasks are memoized;
        # it writes into the same data directory)
        n=$(cat $(for g in $logs; do echo /root/hyp/logs/*/$g/*.log; done) 2>/dev/null | wc -c)
        [ "$n" != "$n_prev" ] && { n_prev=$n; last=$(date +%s); }
        if [ -n "$id" ] && [ $(( $(date +%s) - last )) -gt $(( ${STALL_MIN:-15} * 60 )) ]; then
            stop_hyperion
            log "$L: hyperion idle for ${STALL_MIN:-15} min (lost worker?): master restarted from its database"
            ( cd $ST && setsid nohup $EXE master -p $PROG -d $(ls /root/hyp/databases/*/$id.sqlite | head -1) \
                >> ~/BlocksviBe/results/${L}_master.out 2>&1 < /dev/null & )
            sleep 10
            logs="$logs $(grep -o 'Logging to: "[^"]*' ~/BlocksviBe/results/${L}_master.out | tail -1 | sed 's#.*/##; s#\.log$##')"
            last=$(date +%s)
        fi
    done
    local t1=$(date +%s)
    stop_hyperion
    [ $ok = 1 ] || { log "FAILED $L: hyperion exited before the PMP: $(tail -5 results/${L}_master.out | tr '\n' ' ')"; return 1; }
    OBJ=$(ls -d /root/hyp/data/*/$id/Object_* | head -1); NSV=$(ls $OBJ/json/obj/*.nsv 2>/dev/null | head -1)
    local BD=$OBJ/blocks
    log "$L: PMP built in $(( (t1 - t0) / 60 )) min ($((t1 - t0)) s) incl. blocks + composites + PMP; $(ls -d $BD/blocks_3d_tables_* | wc -l) block tables, $(du -sh $BD | cut -f1); obj $OBJ"
    python3 - $BD >> $LOG <<'EOF'
import glob, os, sys
ds = glob.glob(sys.argv[1] + "/blocks_3d_tables_*")
fs = [f for d in ds for f in glob.glob(d + "/*.json")]
if fs:
    t = [os.path.getmtime(f) for f in fs]
    print(f"   block tables written between +0 and +{max(t) - min(t):.0f} s (first to last file)")
EOF
    cp -r /root/hyp/logs results/${L}_hyp_logs 2>/dev/null
    rm -rf /root/hyp/logs
    return 0
}

# ---------------------------------------------------------------- 2. A: official blocks_3d
if [ "${SKIP_A:-0}" = 1 ]; then
    log "A: resumed (obj $OBJ_A)"
else
log "== A: $PROG with the official blocks_3d (+ jemalloc)"
if run_point A; then
    OBJ_A=$OBJ; NSV_A=$NSV
    mkdir -p /root/sampleA
    for d in $(ls -d $OBJ_A/blocks/blocks_3d_tables_* | sort | awk 'NR % 50 == 1'); do cp -r $d /root/sampleA/; done
    log "A: kept $(ls /root/sampleA | wc -l) block tables as a sample ($(du -sh /root/sampleA | cut -f1))"
    rm -rf $OBJ_A/blocks                                      # disk: the B run writes as much again
fi
fi

# ---------------------------------------------------------------- 3. B: BlocksviBe drop-in
# hyperion keeps the point-independent TTTT results in a central store: move A's aside, so that B
# recomputes every blocks_3d task (550 TTTT + 94 per-point at nmax=6) with the drop-in
[ -d /root/hyp/data/tttt_blocks ] && mv /root/hyp/data/tttt_blocks /root/hyp/data/tttt_blocks_A
cat > $C/blocks_3d.sh <<EOF
#!/bin/bash
export BLOCKSVIBE_FALLBACK=$B3 BLOCKSVIBE_JEMALLOC=$JE BLOCKSVIBE_PYTHON=$PY BLOCKSVIBE_VERBOSE=1
export BLOCKSVIBE_CACHE=/root/bvcache PYTHONPATH=$HOME/SDPviBe PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "\$(date +%s) \$@" >> /root/hyp/blocks_calls.log
exec $HOME/BlocksviBe/bin/blocks_3d "\$@" 2> >(tee -a /root/hyp/blocksvibe_stderr.log >&2)
EOF
chmod +x $C/blocks_3d.sh
( nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 30 > results/B_gpu_util.log 2>&1 & )
log "== B: $PROG with BlocksviBe (GPU)"
if run_point B; then
    OBJ_B=$OBJ; NSV_B=$NSV
    log "B: drop-in calls $(wc -l < /root/hyp/blocks_calls.log); spool batches: $(grep -c 'spool leader' /root/hyp/blocksvibe_stderr.log), sizes $(grep -o 'batch of [0-9]*' /root/hyp/blocksvibe_stderr.log | awk '{print $3}' | sort -n | uniq -c | tr '\n' ' '); fallbacks: $(grep -c 'fallback to' /root/hyp/blocksvibe_stderr.log); GPU failures: $(grep -c 'GPU run failed' /root/hyp/blocksvibe_stderr.log)"
    cp /root/hyp/blocksvibe_stderr.log /root/hyp/blocks_calls.log results/ 2>/dev/null
fi
pkill -f 'nvidia-smi --query-gpu' 2>/dev/null
cp results/blocks_3d_official.sh $C/blocks_3d.sh

# ---------------------------------------------------------------- 4. comparisons
if [ -n "${OBJ_A:-}" ] && [ -n "${OBJ_B:-}" ]; then
    n=0
    for d in /root/sampleA/*; do
        b=$OBJ_B/blocks/$(basename $d)
        [ -d $b ] || { log "sample $(basename $d) missing in B"; continue; }
        log "blocks $(basename $d | cut -c1-40): $(python3 tools/compare_json.py $d $b --prec 768 | head -1)"
        n=$((n+1))
    done
    log "PMP A vs B: $(python3 tools/compare_pmp.py $NSV_A $NSV_B --prec 768 2>&1 | head -3 | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- 5. solves
PARAMS="--precision 1024 --findPrimalFeasible --findDualFeasible --detectPrimalFeasibleJump --detectDualFeasibleJump"
for L in A B; do
    eval NSVL=\${NSV_$L:-}; [ -n "$NSVL" ] || continue
    W=/root/solve_$L; mkdir -p $W
    ( time $MPI -n $NP $R/usr/local/bin/pmp2sdp --precision 1024 --input $NSVL --output $W/sdp_bin ) > results/${L}_pmp2sdp_bin.log 2>&1
    ( time $MPI -n $NP $R/usr/local/bin/pmp2sdp --precision 1024 --input $NSVL --output $W/sdp_json --outputFormat json ) > results/${L}_pmp2sdp_json.log 2>&1
    log "$L pmp2sdp: $(grep -h Processed results/${L}_pmp2sdp_bin.log | head -1)"
    ( cd ~/SDPviBe && time python3 -m sdpvibe -s $W/sdp_json $PARAMS -o $W/gpu_out ) > results/${L}_sdpvibe.log 2>&1
    log "$L SDPviBe: $(grep -E 'read SDP|-----|iterations|^real' results/${L}_sdpvibe.log | tr '\n' ' ')"
    cp $W/gpu_out/iterations.json results/${L}_sdpvibe_iterations.json 2>/dev/null
done
if [ -n "${NSV_B:-}" ]; then
    W=/root/solve_B
    ( time $MPI --bind-to core -n $NP -x LD_PRELOAD=$JE $R/usr/local/bin/sdpb -s $W/sdp_bin -o $W/sdpb_out $PARAMS --noFinalCheckpoint --verbosity 1 ) > results/B_sdpb.log 2>&1
    log "B SDPB ($NP ranks, jemalloc): $(grep -hE 'terminateReason|^real' results/B_sdpb.log $W/sdpb_out/out.txt | tr '\n' ' ')"
    cp $W/sdpb_out/iterations.json results/B_sdpb_iterations.json 2>/dev/null
    python3 ~/SDPviBe/tools/compare_iterations.py results/B_sdpb_iterations.json results/B_sdpvibe_iterations.json 2>&1 | tail -3 >> $LOG
fi
[ -f results/A_sdpvibe_iterations.json ] && [ -f results/B_sdpvibe_iterations.json ] && \
    python3 ~/SDPviBe/tools/compare_iterations.py results/A_sdpvibe_iterations.json results/B_sdpvibe_iterations.json 2>&1 | tail -3 >> $LOG
log DONE
touch results/DONE
for i in $(seq $((HOLD_H * 120))); do [ -f results/PULLED ] && break; sleep 30; done
selfkill "finished"
