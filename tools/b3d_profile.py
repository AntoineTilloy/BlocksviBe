"""Summarise a blocks_3d .profile file by stage (summed over lambda orders / threads).
   python tools/b3d_profile.py <spin_0.json.profile>"""
import re, sys, collections
agg = collections.defaultdict(float)
for line in open(sys.argv[1]):
    m = re.match(r'\s*\{"(\.[^"]+)", ([0-9.e+-]+), (\d+) \}', line)
    if m:
        agg[re.sub(r'lambda_log_r_deriv\.\d+', 'n', m.group(1))] += float(m.group(2))
stages = {"main": ".main", "init": ".main.lambda_log_r_tables.cache_init",
          "h_inf": ".n.conformal_block_derivatives.h_infinity", "recursion": ".n.conformal_block_derivatives.recursion",
          "derivs": ".n.conformal_block_derivatives.residues_to_derivatives", "shift": ".n.shift",
          "write": ".main.write_output"}
print(" ".join(f"{k} {sum(v for n, v in agg.items() if n.endswith(s)):.1f}" for k, s in stages.items()))
