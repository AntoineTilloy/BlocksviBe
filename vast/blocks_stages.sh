#!/usr/bin/env bash
# Run ON the instance after vast/blocks_bench.sh (reuses its blocks_3d image + jemalloc): stage split of
# blocks_3d (recursion / derivatives / shift / output) at larger orders, to know which stages a GPU
# port must cover.  Appends to results/blocks_stages.log; touches results/STAGES_DONE.
cd ~/BlocksviBe
LOG=results/blocks_stages.log
R=$PWD/tools/b3d_root; JE=$PWD/tools/apk/usr/lib/libjemalloc.so.2; T=$(nproc)
run() {   # run <label> args...
    local lab=$1; shift; local O=/tmp/st_$lab; rm -rf $O; mkdir -p $O
    local t0=$(date +%s)
    timeout ${TMAX:-3000} $R/lib/ld-musl-x86_64.so.1 --library-path "$R/usr/local/lib:$R/usr/lib:$R/lib" --preload $JE \
        $R/usr/local/bin/blocks_3d "$@" --num-threads=$T -o $O/spin_{}.json > $O.log 2>&1
    echo "[$(date +%T)] $lab rc=$? wall $(( $(date +%s) - t0 )) s threads $T maxrss? out $(du -sh $O | cut -f1); $(python3 tools/b3d_profile.py $O/spin_0.json.profile 2>&1)" >> $LOG
}
C="--j-external=2,2,2,2 --delta-12=0 --delta-43=0 --delta-1-plus-2=6 --four-pt-sign=1 --coordinates=xt"
# nmax=14-like (Lambda=27), full spin list, 2j12=2j43=8 and 4
run n14_44 $C --j-12=4 --j-43=4 --four-pt-struct=0,0,2,2 --j-internal=0,2,4-50 --order=90 --lambda=27 --kept-pole-order=90 --precision=832
run n14_22 $C --j-12=2 --j-43=2 --four-pt-struct=0,0,2,2 --j-internal=0-50 --order=90 --lambda=27 --kept-pole-order=90 --precision=832
# nmax=22-like (Lambda=43, order 130) with 3 internal spins (recursion as in production, output 3/~90 spins)
run n22_22_3spins $C --j-12=2 --j-43=2 --four-pt-struct=0,0,2,2 --j-internal=0,1,90 --order=130 --lambda=43 --kept-pole-order=130 --precision=832
touch results/STAGES_DONE
