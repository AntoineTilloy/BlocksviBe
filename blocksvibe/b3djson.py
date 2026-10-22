"""blocks_3d's output files, byte layout included: the per-spin JSON (write_output.cxx, write_header.cxx,
write_derivs.cxx) and the .profile file (timers/Timers.hxx), so that consumers that parse them as text
(hyperion's streaming block-table reader) see the same structure.

Numbers follow boost::multiprecision's mpf_float output at blocks_3d's stream precision p = digits10 of
the limb-rounded precision (general format: p significant digits, trailing zeros removed, fixed notation
for decimal exponents -4 <= e < p, else d.ddde+XX); header doubles use C++'s %.{p}g.
"""
import math
import mpmath as mp
from mpmath import libmp

from . import structure as S

COORD_ORDER = ["zzb", "yyb", "xt", "ws", "xt_radial", "ws_radial"]      # std::set<Coordinate> order


def digits10(precision_base_2):
    """boost digits2_2_10 of blocks_3d's limb-rounded precision (mpf_float::default_precision)"""
    rounded = ((precision_base_2 - 1) // 64 + 1) * 64
    return rounded * 301 // 1000


def precision_actual(p):
    return int((p + 1) * math.log(10.0) / math.log(2.0))


def fmt_big(x, p):
    """boost gmp_float::str(p, default flags) of an mpmath mpf"""
    if not x:
        return "0"
    sign, digs, e = libmp.to_digits_exp(x._mpf_, p + 3)        # value = d1.d2d3... * 10^e, digits floored
    if len(digs) > p and digs[p] in "56789":                    # round to p digits (as mpmath's to_str)
        i = p - 1
        while i >= 0 and digs[i] == "9":
            i -= 1
        digs = digs[:i] + str(int(digs[i]) + 1) if i >= 0 else "1"
        if i < 0:
            e += 1
    else:
        digs = digs[:p]
    digs = digs.rstrip("0") or "0"
    if e < -4 or e >= p:
        s = digs[0] + ("." + digs[1:] if len(digs) > 1 else "")
        s += "e" + ("-" if e < 0 else "+") + f"{abs(e):02d}"
    elif e >= 0:
        if len(digs) > e + 1:
            s = digs[:e + 1] + "." + digs[e + 1:]
        else:
            s = digs + "0" * (e + 1 - len(digs))
    else:
        s = "0." + "0" * (-1 - e) + digs
    return ("-" if sign else "") + s


def fmt_double(x, p):
    """std::ostream << double at precision p (general format)"""
    s = f"{float(x):.{p}g}"
    if "e" in s:
        m, ex = s.split("e")
        s = m + "e" + ex[0] + ex[1:].lstrip("0").rjust(2, "0")
    return s


def _list(xs):
    return "[" + ", ".join(xs) + "]"


def header(o, two_j, pp, p):
    """write_header.cxx.  o: options namespace of cli.Task"""
    tjs, tj12, tj43 = o.two_js, o.two_j12, o.two_j43
    dbl = lambda x: fmt_double(x, p)
    ub = two_j / 2.0 + 0.5 if two_j <= 1 else two_j / 2.0 + 1
    mp.mp.prec = max(mp.mp.prec, 64)
    poles = sorted({mp.mpf(S.delta_2(P, two_j)) / 2 - mp.mpf(ub) for P in S.poles(two_j, tj12, tj43, o.kept_pole_order)})
    coords = [c for c in COORD_ORDER if c in o.coordinates]
    out = ["{\n",
           f"  \"j_external\": {_list([dbl(x / 2.0) for x in tjs])},\n",
           f"  \"j_internal\": {dbl(two_j / 2.0)},\n",
           f"  \"delta_minus_x\": {dbl(ub)},\n",
           f"  \"j_12\": {dbl(tj12 / 2.0)},\n",
           f"  \"j_43\": {dbl(tj43 / 2.0)},\n",
           f"  \"delta_12\": \"{o.delta_12}\",\n",
           f"  \"delta_43\": \"{o.delta_43}\",\n",
           f"  \"delta_1_plus_2\": \"{o.delta_1_plus_2}\",\n",
           f"  \"four_pt_struct\": {_list([dbl(x / 2.0) for x in o.q4])},\n",
           f"  \"four_pt_sign\": {o.sign},\n",
           f"  \"order\": {o.order},\n",
           f"  \"lambda\": {o.Lambda},\n",
           f"  \"kept_pole_order\": {o.kept_pole_order},\n",
           "  \"coordinates\": [" + ",".join(f"\"{c}\"" for c in coords) + "],\n",
           "  \"pole_list_x\": [" + ", ".join(fmt_big(x, p) for x in poles) + "],\n",
           f"  \"precision_input\": {o.precision},\n",
           f"  \"precision_actual\": {precision_actual(p)},\n",
           "  \"j_internal_input\": [" + ", ".join(dbl(x / 2.0) for x in o.two_js_internal) + "],\n",
           f"  \"output_template\": \"{o.out_template}\",\n",
           f"  \"args\": \"{' '.join(o.argv)}\",\n",
           "  \"index_values\": [\n"]
    for k in range(2):
        if k:
            out.append(",\n")
        par120 = abs((tj12 - tjs[0] + tjs[1]) // 2 + pp[k][0]) % 2
        par430 = abs((tj43 - tjs[3] + tjs[2]) // 2 + pp[k][1]) % 2
        out += ["    {\n",
                f"     \"three_pt_parity_120\": {par120},\n",
                f"     \"three_pt_parity_430\": {par430},\n",
                f"     \"j_120\": {_list([dbl(x / 2.0) for x in S.so3(pp[k][0], tj12, two_j)])},\n",
                f"     \"j_430\": {_list([dbl(x / 2.0) for x in S.so3(pp[k][1], tj43, two_j)])},\n",
                "     \"coordinates\": [" + ", ".join(f"\"{c}\"" for c in coords) + "]\n",
                "    }"]
    out.append("\n  ],\n")
    out.append("  \"index_names\": [\"three_pt_parity_index\",\n"
               "                  \"j_120_index\",\n"
               "                  \"j_430_index\",\n"
               "                  \"coordinate_index\",\n"
               "                  \"deriv_0\",\n"
               "                  \"deriv_1\",\n"
               "                  \"monomial_degree\"],\n")
    return "".join(out)


def derivs_text(rows, p, eps):
    """write_derivs.cxx: rows[deriv_0][deriv_1] = list of mpmath coefficients (one polynomial)"""
    out = ["          [\n"]
    for d0, row in enumerate(rows):
        out.append("            [\n")
        for d1, poly in enumerate(row):
            out.append("              [\n")
            if all(not c for c in poly):
                out.append("\"0\"")
            else:
                size, mx = len(poly), mp.mpf(0)          # blocks_3d's truncation of negligible top coefficients
                for d in range(len(poly) - 1, -1, -1):
                    c = abs(poly[d])
                    if mx < eps * c:
                        size = d + 1
                    mx = max(mx, c)
                out.append(",\n".join(f"\"{fmt_big(c, p)}\"" for c in poly[:size]))
            out.append("\n              ]")
            if d1 + 1 != len(row):
                out.append(",\n")
        out.append("\n            ]")
        if d0 + 1 != len(rows):
            out.append(",\n")
    out.append("\n          ]")
    return "".join(out)


def write_spin(path, o, two_j, pp, structs, coord_rows):
    """one spin file.  structs[k] = (n_left, n_right); coord_rows[c][k][a][b] = rows for derivs_text
    (c over the requested coordinates in blocks_3d's order)"""
    p = digits10(o.precision)
    eps = mp.mpf(0.5) * mp.mpf(2) ** -int(0.9 * math.log2(10) * p)
    out = [header(o, two_j, pp, p), "  \"derivs\":\n  [\n"]
    for k in range(2):
        out.append("    [\n")
        nl, nr = structs[k]
        for a in range(nl):
            out.append("      [\n")
            for b in range(nr):
                out.append("        [\n")
                out.append(",\n".join(derivs_text(c[k][a][b], p, eps) for c in coord_rows))
                out.append("\n        ]")
                if b + 1 != nr:
                    out.append(",\n")
            out.append("\n      ]")
            if a + 1 != nl:
                out.append(",\n")
        out.append("\n    ]")
        if k + 1 != 2:
            out.append(",\n")
    out.append("\n  ]\n}\n")
    with open(path, "w") as f:
        f.write("".join(out))


def write_profile(path, o, num_threads, timers):
    """timers: list of (name, seconds, runs), names like ".main.setup" (blocks_3d's Timers::write_profile)"""
    dbl = lambda x: f"{x:.10g}"
    coords = [c for c in COORD_ORDER if c in o.coordinates]
    out = ["{{\n",
           f"  {{\"num_threads\", {num_threads}}},\n",
           f"  {{\"j_external\", {_list([dbl(x / 2.0) for x in o.two_js])}}},\n",
           f"  {{\"j_12\", {dbl(o.two_j12 / 2.0)}}},\n",
           f"  {{\"j_43\", {dbl(o.two_j43 / 2.0)}}},\n",
           f"  {{\"four_pt_struct\", {_list([dbl(x / 2.0) for x in o.q4])}}},\n",
           f"  {{\"order\", {o.order}}},\n",
           f"  {{\"lambda\", {o.Lambda}}},\n",
           f"  {{\"kept_pole_order\", {o.kept_pole_order}}},\n",
           "  {\"coordinates\", {" + ",".join(f"\"{c}\"" for c in coords) + "}},\n",
           f"  {{\"precision_input\", {o.precision}}},\n",
           "  {\"j_internal_input\", {" + ", ".join(dbl(x / 2.0) for x in o.two_js_internal) + "}}\n",
           "},{\n"]
    out.append(",\n".join(f"    {{\"{n}\", {dbl(t)}, {r} }}" for n, t, r in timers) + "\n}}\n")
    with open(path, "w") as f:
        f.write("".join(out))
