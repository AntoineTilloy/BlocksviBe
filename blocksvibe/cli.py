"""Drop-in blocks_3d command line on the GPU.

    blocksvibe [blocks_3d options]            same options and param file (-p) as blocks_3d, same per-spin
                                              JSON files and .profile
    blocksvibe batch [--lines FILE] [PARAM_FILE ...]
                                              many blocks_3d tasks (param files, or one blocks_3d command
                                              line per line of FILE): tasks of one (j_external, j12, j43,
                                              Delta_12, Delta_43, order, Lambda, j_max, precision) group
                                              share one setup run and one GPU recursion operator, and their
                                              right-hand sides (q4, lambda order) run as one batch

Supported on the GPU: coordinates xt and/or xt_radial, with or without pole shifting (kept_pole_order).
Anything else (or no GPU, or no b3d_setup helper, or a GPU failure) runs the official blocks_3d instead:
$BLOCKSVIBE_FALLBACK (path to the binary or a wrapper script) with $BLOCKSVIBE_JEMALLOC preloaded if set
(the official images are musl builds: 20-40x slower without it).

Environment: BLOCKSVIBE_B3D_SETUP (helper, default <repo>/build/b3d_setup), BLOCKSVIBE_LIMBS (GPU limbs,
default ceil((precision + 62) / 32) with the precision rounded up to 64-bit limbs as blocks_3d does),
BLOCKSVIBE_MAX_RHS (cap on right-hand sides per GPU batch), BLOCKSVIBE_CACHE (write_output maps and M
matrices, default ~/.cache/blocksvibe), BLOCKSVIBE_LOCKDIR (GPU lock files, default /tmp),
BLOCKSVIBE_SPOOL / _SPOOL_QUIET / _SPOOL_MAX / _SPOOL_OFF=1 (joining concurrent invocations of one group
into one batch, spool_run), BLOCKSVIBE_VERBOSE=1.
GPUs: one task (group) at a time per GPU, through a lock file per GPU; the CPU setup stage runs before
the lock is taken, so concurrent tasks overlap their CPU work with each other's GPU work.
"""
import fcntl
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

OPTS = {  # blocks_3d option -> (attribute, default); None = required
    "j-external": ("j_external", None), "j-internal": ("j_internal", None), "j-12": ("j12", None),
    "j-43": ("j43", None), "delta-12": ("delta_12", None), "delta-43": ("delta_43", None),
    "delta-1-plus-2": ("delta_1_plus_2", "0"), "four-pt-struct": ("four_pt_struct", None),
    "four-pt-sign": ("four_pt_sign", None), "order": ("order", None), "lambda": ("lambda", None),
    "coordinates": ("coordinates", "zzb"), "kept-pole-order": ("kept_pole_order", None),
    "num-threads": ("num_threads", None), "precision": ("precision", None),
    "out-template": ("out_template", None), "debug": ("debug", "false"), "profile": ("profile", "true"),
    "param-file": ("param_file", ""), "help": ("help", None),
}
SHORT = {"o": "out-template", "p": "param-file", "h": "help"}
GPU_COORDS = {"xt", "xt_radial"}


class UsageError(Exception):
    pass


def log(*a):
    if os.environ.get("BLOCKSVIBE_VERBOSE") == "1":
        print("[blocksvibe]", *a, file=sys.stderr, flush=True)


def _long(name):
    """boost::program_options long-name guessing: exact match or unique prefix"""
    if name in OPTS:
        return name
    cand = [k for k in OPTS if k.startswith(name)]
    if len(cand) != 1:
        raise UsageError(f"unrecognised option '--{name}'")
    return cand[0]


def parse_args(argv):
    """blocks_3d's handle_input: command line, then the param file (key=value) for options not given"""
    vals, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            name, eq, v = a[2:].partition("=")
            name = _long(name)
        elif a.startswith("-") and len(a) >= 2 and a[1] in SHORT:
            name, v, eq = SHORT[a[1]], a[2:], a[2:] != ""
        else:
            raise UsageError(f"unexpected argument '{a}'")
        if name == "help":
            vals["help"] = "1"; i += 1; continue
        if not eq:
            if i + 1 >= len(argv):
                raise UsageError(f"option '--{name}' requires an argument")
            v = argv[i + 1]; i += 1
        if name in vals:
            raise UsageError(f"option '--{name}' cannot be specified more than once")
        vals[name] = v
        i += 1
    if vals.get("param-file"):
        try:
            lines = open(vals["param-file"]).read().splitlines()
        except OSError:
            raise UsageError(f"Could not open '{vals['param-file']}'")
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("["):
                continue
            k, eq, v = line.partition("=")
            k = k.strip()
            if not eq or k not in OPTS:
                raise UsageError(f"unrecognised line in param file: '{line}'")
            vals.setdefault(k, v.strip())
    return vals


def _two(s):
    x = 2 * float(s)
    if x != int(x):
        raise UsageError(f"not a whole or half integer: {s}")
    return int(x)


def _bool(s):
    s = s.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise UsageError(f"not a boolean: {s}")


def spins(spec, two_j12):
    """j-internal ranges '0.5-3.5,5.5,8.5-16.5-2' -> sorted two_j values"""
    out = set()
    for s in spec.split(","):
        parts = s.split("-")
        if not s or len(parts) > 3:
            raise UsageError(f"Found invalid element in j-internal: '{s}'")
        first = _two(parts[0]); last = _two(parts[1]) if len(parts) > 1 else first
        if first % 2 != two_j12 % 2:
            raise UsageError("j_12 and j_internal must all be either whole or half integers")
        step = 2 * int(parts[2]) if len(parts) > 2 else 2
        if (last - first) % step:
            raise UsageError(f"The step of a spin range in j_internal must exactly divide the difference: {s}")
        out.update(range(first, last + 1, step))
    return sorted(out)


class Task:
    """one blocks_3d invocation (normalised options)"""

    def __init__(self, argv, prog="blocks_3d"):
        self.argv = [prog] + list(argv)
        v = parse_args(argv)
        self.help = "help" in v
        if self.help:
            return
        missing = [k for k, (_, d) in OPTS.items() if d is None and k != "help" and k not in v]
        if missing:
            raise UsageError(f"the option '--{missing[0]}' is required but missing")
        g = lambda k: v.get(k, OPTS[k][1])
        self.two_j12, self.two_j43 = _two(g("j-12")), _two(g("j-43"))
        if self.two_j12 % 2 != self.two_j43 % 2:
            raise UsageError("j_12 and j_43 must both be either integers or half integers")
        self.two_js = [_two(x) for x in g("j-external").split(",")]
        self.q4 = [_two(x) for x in g("four-pt-struct").split(",")]
        if len(self.two_js) != 4 or len(self.q4) != 4:
            raise UsageError("j-external and four-pt-struct take 4 values")
        self.sign = int(g("four-pt-sign"))
        if self.sign not in (1, -1):
            raise UsageError(f"Four-pt-sign must be 1 or -1, but got: {self.sign}")
        for i in range(4):
            if (self.q4[i] - self.two_js[i]) % 2 or not -self.two_js[i] <= self.q4[i] <= self.two_js[i]:
                raise UsageError(f"invalid four-pt-struct entry {i}")
        self.two_js_internal = spins(g("j-internal"), self.two_j12)
        self.delta_12, self.delta_43, self.delta_1_plus_2 = g("delta-12"), g("delta-43"), g("delta-1-plus-2")
        self.order, self.Lambda = int(g("order")), int(g("lambda"))
        self.kept_pole_order, self.num_threads = int(g("kept-pole-order")), int(g("num-threads"))
        self.precision = int(g("precision"))
        self.out_template = g("out-template")
        self.coordinates = set(g("coordinates").split(","))
        bad = self.coordinates - {"zzb", "yyb", "xt", "ws", "xt_radial", "ws_radial"}
        if bad:
            raise UsageError(f"Invalid entry for coordinates: '{bad.pop()}'")
        self.radial = all(c.endswith("_radial") for c in self.coordinates)
        self.debug, self.profile = _bool(g("debug")), _bool(g("profile"))
        self.p0 = (1 - self.sign) // 2
        self.tjmax = self.two_js_internal[-1]

    def unsupported(self):
        if not self.coordinates <= GPU_COORDS:
            return f"coordinates {sorted(self.coordinates - GPU_COORDS)} (GPU: xt, xt_radial)"
        return None

    def group_key(self):
        """tasks sharing the recursion operator and the M matrices"""
        return (tuple(self.two_js), self.two_j12, self.two_j43, self.delta_12, self.delta_43, self.order,
                self.Lambda, self.tjmax, self.precision, min(self.kept_pole_order, self.order))

    def lambda_orders(self):
        return [self.p0] if self.radial else list(range(self.p0, self.Lambda + 1, 2))

    def path(self, x):
        return fmt_template(self.out_template, x)


def fmt_template(tmpl, x):
    """fmt::format(template, double) for the '{}' / '{:spec}' fields blocks_3d's templates use"""
    def rep(m):
        spec = m.group(1)
        if spec:
            return format(x, spec)
        s = repr(float(x))
        return s[:-2] if s.endswith(".0") else s
    return re.sub(r"\{(?::([^}]*))?\}", rep, tmpl)


# ---------------------------------------------------------------- fallback

def fallback(argv, reason):
    exe = os.environ.get("BLOCKSVIBE_FALLBACK")
    if not exe:
        print(f"blocksvibe: {reason}, and no fallback (set BLOCKSVIBE_FALLBACK to the official blocks_3d)",
              file=sys.stderr)
        sys.exit(1)
    env = dict(os.environ)
    je = os.environ.get("BLOCKSVIBE_JEMALLOC")
    if je:
        env["LD_PRELOAD"] = (je + " " + env.get("LD_PRELOAD", "")).strip()
    log(f"fallback to {exe}: {reason}")
    sys.stdout.flush()
    os.execve(exe, [exe] + list(argv), env)


def gpu_count():
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        ids = [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip() and x.strip() != "-1"]
        return ids
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [str(i) for i, l in enumerate(r.stdout.splitlines()) if l.startswith("GPU ")]


class GpuLock:
    """exclusive use of one GPU: lock file per GPU id; sets CUDA_VISIBLE_DEVICES before torch is imported"""

    def __init__(self, ids):
        d = os.environ.get("BLOCKSVIBE_LOCKDIR", tempfile.gettempdir())
        self.f = None
        for gid in ids:                                       # a free GPU if any
            f = open(os.path.join(d, f"blocksvibe-gpu{gid}.lock"), "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.f, self.gid = f, gid
                break
            except OSError:
                f.close()
        if self.f is None:                                    # else wait for one
            gid = ids[os.getpid() % len(ids)]
            self.f = open(os.path.join(d, f"blocksvibe-gpu{gid}.lock"), "w")
            t = time.time()
            fcntl.flock(self.f, fcntl.LOCK_EX)
            self.gid = gid
            log(f"waited {time.time() - t:.1f} s for GPU {gid}")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gid

    def release(self):
        if self.f:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.f = None


# ---------------------------------------------------------------- GPU run of a group of tasks

def limbs(precision):
    if os.environ.get("BLOCKSVIBE_LIMBS"):
        return int(os.environ["BLOCKSVIBE_LIMBS"])
    rounded = ((precision - 1) // 64 + 1) * 64
    return -(-(rounded + 62) // 32)


def cache_dir():
    return os.environ.get("BLOCKSVIBE_CACHE", os.path.expanduser("~/.cache/blocksvibe"))


def cached_output(coord, Lam, p0, d12, L):
    """output.Output with its write_output map cached on disk (mpmath construction: ~2 s at Lambda 19,
    ~1 min at Lambda 43)"""
    import torch
    from . import output as OUT
    from sdpvibe import mpf
    from sdpvibe.rns import dev
    d = cache_dir()
    key = hashlib.sha1(f"{coord}|{Lam}|{p0}|{d12}|{L}|v1".encode()).hexdigest()[:20]
    path = os.path.join(d, f"wmap_{coord}_L{Lam}_p{p0}_{key}.pt")
    if os.path.exists(path):
        try:
            st = torch.load(path)
            o = object.__new__(OUT.Output)
            o.__dict__.update(st["attrs"])
            o.W = mpf.MPF(st["m"].to(dev), st["e"].to(dev))
            return o
        except Exception as e:                                # unreadable cache: rebuild
            log(f"cache {path}: {e}")
    o = OUT.Output(coord, Lam, p0, d12, L)
    try:
        os.makedirs(d, exist_ok=True)
        tmp = path + f".{os.getpid()}"
        torch.save({"attrs": {k: v for k, v in o.__dict__.items() if k != "W"}, "m": o.W.m.cpu(),
                    "e": o.W.e.cpu()}, tmp)
        os.replace(tmp, path)
    except OSError as e:
        log(f"cannot write cache {path}: {e}")
    return o


def _max_rhs(plan, der, L):
    """right-hand sides per batch from the free GPU memory (residue slots + derivative moments, x3 for
    temporaries), capped by BLOCKSVIBE_MAX_RHS"""
    import torch
    free, _ = torch.cuda.mem_get_info()
    S2, nb = plan.Smax ** 2, 4 * L + 8
    per = S2 * nb * ((plan.nphys + 1) + 2 * der.Jo * der.NP * (der.R + 1) + 2 * (plan.order + 1) * plan.Jh)
    n = max(1, int(0.6 * free / (3 * per)))
    cap = os.environ.get("BLOCKSVIBE_MAX_RHS")
    return min(n, int(cap)) if cap else n


def run_group(tasks, timers):
    """tasks: list of Task with one group_key.  Writes all their outputs."""
    t0 = time.time()
    from . import setup as SU
    T = tasks[0]
    q4s, seen = [], set()
    for t in tasks:
        k = (tuple(t.q4), t.sign)
        if k not in seen:
            seen.add(k); q4s.append(k)
    radial = all(t.radial for t in tasks)
    threads = max(t.num_threads for t in tasks)
    # M matrices: cached on disk per group and four-point parities (the 22 q4 tasks of a group share them)
    parities = sorted({SU.four_point_parity(T.two_js, q) for q, _ in q4s})
    mkey = hashlib.sha1(repr((T.group_key(), parities, "m1")).encode()).hexdigest()[:24]
    mcache = os.path.join(cache_dir(), f"M_{mkey}.bin")
    have_m = os.path.exists(mcache)
    with tempfile.TemporaryDirectory(prefix="blocksvibe_") as tmp:
        sfile = os.path.join(tmp, "setup.bin")
        SU.run_helper(sfile, T.two_js, T.two_j12, T.two_j43, T.delta_12, T.delta_43, T.order, T.Lambda,
                      T.tjmax, radial, q4s, T.precision, threads, skip_m=have_m)
        if not have_m:
            try:
                os.makedirs(cache_dir(), exist_ok=True)
                SU.save_m(sfile, mcache)
            except OSError as e:
                log(f"cannot cache M matrices: {e}")
        t1 = time.time()
        timers.append((".main.setup", t1 - t0, 1))
        log(f"setup (h_inf{', M cached' if have_m else ', M'}) for {len(q4s)} structure(s): {t1 - t0:.2f} s")
        ids = gpu_count()
        if not ids:
            raise RuntimeError("no GPU")
        lock = GpuLock(ids)
        try:
            t2 = time.time()
            if t2 - t1 > 0.5:
                timers.append((".main.gpu_wait", t2 - t1, 1))
            results = _gpu(tasks, q4s, sfile, mcache if have_m else None, timers)
        finally:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            lock.release()
    t3 = time.time()
    _write(tasks, results)
    timers.append((".main.write_output", time.time() - t3, 1))


def _gpu(tasks, q4s, sfile, mfile, timers):
    """-> {(q4, sign): (spins, pp, {coord: numpy object array (Jo, 2, S2, outputs, Dout)}, Output by coord,
    Smax)}"""
    import torch
    from . import setup as SU, recursion_gpu as RG, derivatives as DV, output as OUT
    from sdpvibe import mpf
    T = tasks[0]
    L = limbs(T.precision)
    t0 = time.time()
    M, stasks = SU.read(sfile, L, m_path=mfile)
    by_q4 = {(st.q4, st.sign): st for st in stasks}
    # subgroups: same pseudo-parity types and sign -> same Plan and lambda orders
    sub = {}
    for k in q4s:
        st = by_q4[(tuple(k[0]), k[1])]
        sub.setdefault((st.pp, st.sign), []).append(st)
    torch.cuda.synchronize()
    timers.append((".main.gpu.read_setup", time.time() - t0, 1))
    results = {}
    for (pp, sign), sts in sub.items():
        ts = [t for t in tasks if t.sign == sign and (tuple(t.q4), t.sign) in {(s.q4, s.sign) for s in sts}]
        p0 = (1 - sign) // 2
        radial = all(t.radial for t in ts)
        ns = [p0] if radial else list(range(p0, T.Lambda + 1, 2))
        tjs = sorted(set().union(*[t.two_js_internal for t in ts]))
        coords = [c for c in ("xt", "xt_radial") if any(c in t.coordinates for t in ts)]
        d12s = sorted({t.delta_1_plus_2 for t in ts})
        t1 = time.time()
        plan = RG.Plan(T.two_j12, T.two_j43, T.tjmax, T.order, pp, reuse=True)
        rec = RG.Recursion(plan, M, L)
        der = DV.Derivs(plan, T.Lambda - p0, L, two_js=tjs, kept_pole_order=T.kept_pole_order)
        outs = {(c, d): cached_output(c, T.Lambda, p0, d, L) for c in coords for d in d12s}
        torch.cuda.synchronize()
        t2 = time.time()
        timers.append((".main.gpu.constants", t2 - t1, 1))
        Sm, Jh = plan.Smax, plan.Jh
        bmax = _max_rhs(plan, der, L)
        nums_all = []                     # per chunk of structures: list over ns of numerators
        chunks = [sts[i:i + bmax] for i in range(0, len(sts), bmax)]
        for ch in chunks:
            nums = []
            for n in ns:
                dn = der.for_order(T.Lambda - n)
                hs = [SU.h_inf_tensor(st, [n], T.order, Jh, Sm, L) for st in ch]
                h = mpf.MPF(torch.cat([x.m for x in hs], -1), torch.cat([x.e for x in hs], -1)) if len(hs) > 1 else hs[0]
                dn.start(len(ch))
                rec.run(h, hook=dn.accumulate)
                nums.append(dn.finish(h))
                del h, hs
            nums_all.append(nums)
        torch.cuda.synchronize()
        t3 = time.time()
        timers.append((".main.gpu.chain", t3 - t2, len(ns) * len(chunks)))
        log(f"pp {pp} sign {sign}: {len(sts)} structure(s) x {len(ns)} lambda order(s), batch <= {bmax}: "
            f"constants {t2 - t1:.2f} s, chain {t3 - t2:.2f} s")
        outs_host = {k: _OutputInfo(o) for k, o in outs.items()}
        for ch, nums in zip(chunks, nums_all):
            vals = {}
            for (c, d), o in outs.items():
                r = OUT.apply_per_order(o, nums)                   # (Jo, 2, S2, nq, outputs, Dmax)
                vals[(c, d)] = (r.m.cpu().numpy(), r.e.cpu().numpy())       # host limbs (formatted later)
            for q, st in enumerate(ch):
                results[(st.q4, st.sign)] = (tjs, pp, {k: (m[:, :, :, :, q], e[:, :, :, q]) for k, (m, e) in vals.items()},
                                             outs_host, Sm, 32 * L - 2)
        timers.append((".main.gpu.write_output", time.time() - t3, 1))
    return results


class _OutputInfo:
    """what the writer needs of an output.Output (host only, picklable)"""

    def __init__(self, o):
        self.outputs = list(o.outputs)


_JOBS = {}


def _write(tasks, results):
    """all spin files of all tasks, formatted in parallel (fork: the workers only use host data)"""
    import multiprocessing as mpr
    jobs = [(i, tj) for i, t in enumerate(tasks) for tj in t.two_js_internal]
    _JOBS.update(tasks=tasks, results=results)
    nproc = max(1, min(len(jobs), max(t.num_threads for t in tasks), os.cpu_count() or 1))
    try:
        if nproc == 1:
            for j in jobs:
                _write_one(j)
        else:
            with mpr.get_context("fork").Pool(nproc) as pool:
                for _ in pool.imap_unordered(_write_one, jobs):
                    pass
    finally:
        _JOBS.clear()


def _write_one(job):
    import numpy as np
    from . import b3djson as BJ, structure as S
    i, tj = job
    t = _JOBS["tasks"][i]
    tjs, pp, vals, outs, Sm, prec = _JOBS["results"][(tuple(t.q4), t.sign)]
    jo = tjs.index(tj)
    structs = [(len(S.so3(pp[k][0], t.two_j12, tj)), len(S.so3(pp[k][1], t.two_j43, tj))) for k in range(2)]
    coord_rows = []
    for c in [c for c in BJ.COORD_ORDER if c in t.coordinates]:
        o = outs[(c, t.delta_1_plus_2)]
        m, e = vals[(c, t.delta_1_plus_2)]                         # (L, Jo, 2, S2, outputs, Dmax)
        mj, ej = m[:, jo], e[jo]
        v = np.array(BJ.limbs_to_mpfr(mj.reshape(mj.shape[0], -1), ej.reshape(-1), prec),
                     dtype=object).reshape(ej.shape)               # (2, S2, outputs, Dmax)
        coord_rows.append([[[_rows(o, v[k, a * Sm + b]) for b in range(structs[k][1])]
                            for a in range(structs[k][0])] for k in range(2)])
    path = t.path(tj / 2.0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    BJ.write_spin(path, t, tj, pp, structs, coord_rows, gmp=True)


def _rows(o, v):
    """v: (outputs, Dout) object array -> rows[deriv_0][deriv_1] = polynomial coefficients"""
    rows = {}
    for k, (n, b) in enumerate(o.outputs):
        rows.setdefault(n, []).append(list(v[k]))
    return [rows[n] for n in sorted(rows)]


# ---------------------------------------------------------------- entry points

def run_tasks(tasks, argvs):
    """GPU for the supported tasks (grouped), official blocks_3d for the others; -> exit status per task"""
    groups, status = {}, [0] * len(tasks)
    for i, t in enumerate(tasks):
        why = t.unsupported()
        if why:
            status[i] = _run_fallback(argvs[i], why)
        else:
            groups.setdefault(t.group_key(), []).append(i)
    for key, idx in groups.items():
        ts = [tasks[i] for i in idx]
        timers = []
        t0 = time.time()
        try:
            run_group(ts, timers)
        except Exception as e:
            import traceback
            log(traceback.format_exc())
            for i in idx:
                status[i] = _run_fallback(argvs[i], f"GPU run failed: {type(e).__name__}: {e}")
            continue
        total = time.time() - t0
        for t in ts:
            if t.profile:
                write_profile(t, total, timers)
    return status


def write_profile(t, total, timers):
    from . import b3djson as BJ
    path = fmt_template(t.out_template, 0.0) + ".profile"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    BJ.write_profile(path, t, t.num_threads, [(".main", total, 1)] + timers)


def _run_fallback(argv, reason):
    exe = os.environ.get("BLOCKSVIBE_FALLBACK")
    if not exe:
        print(f"blocksvibe: {reason}, and no fallback (set BLOCKSVIBE_FALLBACK)", file=sys.stderr)
        return 1
    env = dict(os.environ)
    if os.environ.get("BLOCKSVIBE_JEMALLOC"):
        env["LD_PRELOAD"] = (os.environ["BLOCKSVIBE_JEMALLOC"] + " " + env.get("LD_PRELOAD", "")).strip()
    log(f"fallback to {exe}: {reason}")
    return subprocess.run([exe] + list(argv), env=env).returncode


# ---------------------------------------------------------------- spool: concurrent invocations -> one batch

def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _atomic_write(path, text):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def spool_run(task, argv):
    """Join concurrent invocations of the same group into one GPU batch, transparently (hyperion calls
    blocks_3d once per four-point structure).  Every invocation drops its arguments in a spool directory
    per group; whoever holds the group's leader lock waits until no new task arrived for
    BLOCKSVIBE_SPOOL_QUIET s (default 3, at most BLOCKSVIBE_SPOOL_MAX = 30 s), claims all pending tasks,
    runs them as one batch and writes each task's exit status; the others wait for theirs.  A task
    claimed by a leader that died is put back."""
    import json
    import uuid
    root = os.environ.get("BLOCKSVIBE_SPOOL", os.path.join(tempfile.gettempdir(), "blocksvibe-spool"))
    d = os.path.join(root, hashlib.sha1(repr(task.group_key()).encode()).hexdigest()[:20])
    os.makedirs(d, exist_ok=True)
    quiet = float(os.environ.get("BLOCKSVIBE_SPOOL_QUIET", "3"))
    tmax = float(os.environ.get("BLOCKSVIBE_SPOOL_MAX", "30"))
    me = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _atomic_write(os.path.join(d, me + ".task"), json.dumps({"argv0": task.argv[0], "argv": argv}))
    done = os.path.join(d, me + ".done")
    lockf = open(os.path.join(d, "leader.lock"), "w")
    while True:
        if os.path.exists(done):
            st = int(open(done).read().strip() or 1)
            os.remove(done)
            return st
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            time.sleep(0.5)
            if not os.path.exists(os.path.join(d, me + ".task")):
                claimed = [f for f in os.listdir(d) if f.startswith(me + ".claimed.")]
                for f in claimed:                                  # leader died: put the task back
                    pid = int(f.rsplit(".", 1)[1])
                    if not _alive(pid) and not os.path.exists(done):
                        try:
                            os.rename(os.path.join(d, f), os.path.join(d, me + ".task"))
                        except OSError:
                            pass
            continue
        try:                                                       # leader
            t0, last, n = time.time(), time.time(), -1
            while True:
                pending = sorted(f for f in os.listdir(d) if f.endswith(".task"))
                if len(pending) != n:
                    n, last = len(pending), time.time()
                if time.time() - last >= quiet or time.time() - t0 >= tmax:
                    break
                time.sleep(0.25)
            mine = []
            for f in pending:
                tid = f[:-5]
                try:
                    os.rename(os.path.join(d, f), os.path.join(d, f"{tid}.claimed.{os.getpid()}"))
                    mine.append(tid)
                except OSError:
                    pass
            if mine:
                specs = [json.load(open(os.path.join(d, f"{tid}.claimed.{os.getpid()}"))) for tid in mine]
                argvs = [sp["argv"] for sp in specs]
                log(f"spool leader: batch of {len(mine)} task(s) after {time.time() - t0:.1f} s")
                try:
                    tasks = [Task(sp["argv"], sp["argv0"]) for sp in specs]
                    status = run_tasks(tasks, argvs)
                except Exception as e:
                    log(f"spool batch failed: {e}")
                    status = [_run_fallback(a, f"batch failed: {e}") for a in argvs]
                for tid, st in zip(mine, status):
                    _atomic_write(os.path.join(d, tid + ".done"), str(st))
                    os.remove(os.path.join(d, f"{tid}.claimed.{os.getpid()}"))
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    prog = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "blocks_3d"
    if argv[:1] == ["batch"]:
        rest, argvs = argv[1:], []
        while rest:
            if rest[0] == "--lines":
                for line in open(rest[1]):
                    if line.strip() and not line.lstrip().startswith("#"):
                        argvs.append(shlex.split(line))
                rest = rest[2:]
            else:
                argvs.append(["-p", rest[0]]); rest = rest[1:]
        try:
            tasks = [Task(a, prog) for a in argvs]
        except UsageError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return max(run_tasks(tasks, argvs), default=0)
    try:
        task = Task(argv, sys.argv[0] if sys.argv and sys.argv[0] else prog)
    except UsageError as e:
        if os.environ.get("BLOCKSVIBE_FALLBACK"):
            fallback(argv, f"options not understood ({e})")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if task.help:
        if os.environ.get("BLOCKSVIBE_FALLBACK"):
            fallback(argv, "--help")
        print(__doc__)
        return 0
    why = task.unsupported()
    if why:
        fallback(argv, why)
    if os.environ.get("BLOCKSVIBE_SPOOL_OFF") != "1":
        return spool_run(task, argv)
    return run_tasks([task], [argv])[0]


if __name__ == "__main__":
    sys.exit(main())
