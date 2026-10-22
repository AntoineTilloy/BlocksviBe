#!/usr/bin/env bash
# Run ON a rented GPU instance (detached) from ~/BlocksviBe, next to ~/SDPviBe (push both: SDPviBe's
# vast/sync.sh, and rsync this repository to ~/BlocksviBe):
#   ssh ... 'cd BlocksviBe && (DEADLINE_H=<h> setsid nohup bash vast/blocks_chain.sh < /dev/null > /dev/null 2>&1 &)'
# Full GPU block chain (docs/BLOCKS_PLAN.md section 6): plan builder, recursion + derivative moments +
# numerators per lambda order, write_output, on synthetic operators of real task shapes:
#   nmax=10 (order 60, j_max 60, Lambda 19, 926 bits): whole task, numerators staged on disk, write_output;
#   nmax=22 (order 130, j_max 90, Lambda 43, 832 bits): lambda orders 0, 22, 42 and write_output separately.
# Results are appended to results/blocks_chain.log as they come (pull them periodically); at the end
# results/DONE, then wait (<= 1 h) for the client's `touch results/PULLED`, then self-destroy.
# Hard deadline DEADLINE_H (derive it from the budget: budget / (dph + disk $/h), with margin).
cd ~/BlocksviBe
DEADLINE_H=${DEADLINE_H:-2}
mkdir -p results
LOG=results/blocks_chain.log
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
lscpu | grep -E "Model name|^CPU\(s\)" >> $LOG; free -g | head -2 >> $LOG; df -h ~ | tail -1 >> $LOG
export PYTHONPATH=$PWD:$HOME/SDPviBe PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. ~/venv/bin/activate
STAGE=/root/blocks_stage

log "plan builder and derivative constants"
python tools/bench_plan.py 8 8 60 60 19 29 >> $LOG 2>&1
python tools/bench_plan.py 8 8 90 130 43 27 >> $LOG 2>&1
python tools/bench_plan.py 8 8 100 152 51 28 >> $LOG 2>&1

log "chain, nmax=10 shape (order 60, Lambda 19), 926 bits, whole task"
python tools/bench_chain.py 8 8 60 60 19 29 8 $STAGE >> $LOG 2>&1
log "chain, order 52 (the laptop's largest), 926 bits, whole task"
python tools/bench_chain.py 8 8 52 52 19 29 8 $STAGE >> $LOG 2>&1

log "chain, nmax=22 shape (order 130, j_max 90, Lambda 43), 832 bits, lambda orders 0, 22, 42"
ORDERS=0,22,42 python tools/bench_chain.py 8 8 90 130 43 27 4 $STAGE >> $LOG 2>&1 \
  || { log "retry with OFFLOAD=1 TASK_CHUNK=2048"; ORDERS=0,22,42 OFFLOAD=1 TASK_CHUNK=2048 python tools/bench_chain.py 8 8 90 130 43 27 4 $STAGE >> $LOG 2>&1; }
log "write_output, Lambda 43 / 19"
python tools/bench_output.py 8 8 90 130 43 27 2 >> $LOG 2>&1
python tools/bench_output.py 8 8 60 60 19 29 4 >> $LOG 2>&1

log "DONE"
touch results/DONE
for i in $(seq 1 60); do [ -f results/PULLED ] && break; sleep 60; done
kill $WATCHDOG 2>/dev/null
selfkill "finished"
