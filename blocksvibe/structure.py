"""Index bookkeeping of blocks_3d's radial recursion (Erramilli, Iliesiu, Kravchuk, Landry, Poland,
Simmons-Duffin, arXiv:2011.01959; https://gitlab.com/bootstrapcollaboration/blocks_3d), ported 1:1
from Pole.hxx, Pole_Types.cxx, two_j_values.cxx, so3_tensor_parity.cxx and fill_residues.cxx.

Spins are passed doubled (two_j), as in blocks_3d.  A pole is (type, k) with type 0..3 = I..IV.
"""
from fractions import Fraction
from functools import lru_cache

I, II, III, IV = range(4)


def bosonic(two_j): return two_j % 2 == 0


def pole_range_k(t, two_j, two_j12, two_j43, order):
    if t == I:
        return order
    if t == II:
        return min(two_j // 2, order)
    if t == III:
        return order // 2 if bosonic(two_j) else min(two_j // 2, two_j12 // 2, two_j43 // 2, order // 2)
    return (min(two_j // 2, two_j12 // 2, two_j43 // 2, (order + 1) // 2) if bosonic(two_j)
            else (order + 1) // 2)


@lru_cache(maxsize=None)
def poles(two_j, two_j12, two_j43, order):
    return tuple((t, k) for t in range(4) for k in range(1, pole_range_k(t, two_j, two_j12, two_j43, order) + 1))


def position(two_j, two_j12, two_j43, order, pole):
    t, k = pole
    return k - 1 + sum(pole_range_k(u, two_j, two_j12, two_j43, order) for u in range(t))


def shift(pole):
    t, k = pole
    return k if t < III else (2 * k if t == III else 2 * k - 1)


def pole_two_j(pole, two_j):
    t, k = pole
    return two_j + 2 * k if t == I else (two_j - 2 * k if t == II else two_j)


def delta_2(pole, two_j):
    """twice the pole position in Delta"""
    t, k = pole
    return (2 - two_j - 2 * k, 4 + two_j - 2 * k, 3 - 2 * k, 4 - 2 * k)[t]


def parity(pole): return 1 if pole[0] == IV else 0


def q_value(pole, two_j):
    """Q(pole, two_j) of fill_residues/Q.hxx, exact"""
    t, k = pole
    d = Fraction(4) ** shift(pole)
    if t in (I, II):
        return -d
    if t == III:
        return -d * Fraction(k * (1 + two_j - 2 * k), 1 + two_j + 2 * k)
    return d * Fraction((2 + two_j - 2 * k) * (2 * k - 1), 2 * two_j + 4 * k)


def two_j_values(m, order, two_j12, two_j_max):
    return list(range(two_j12 % 2, two_j_max + 2 * (order - m - 1) + 1, 2))


@lru_cache(maxsize=None)
def so3(pseudo_parity, two_j1, two_j2):
    return tuple(c for c in range(abs(two_j1 - two_j2), two_j1 + two_j2 + 1, 2)
                 if ((two_j1 + two_j2 - c) - 2 * pseudo_parity) % 4 == 0)


def j_excess(two_j, two_j_max): return (two_j - two_j_max) // 2 if two_j > two_j_max else 0


def cauchy_coeff(dprime_2, pole_delta_2):
    """+-2/|diff| as in fill_residues (0 if the difference vanishes)"""
    diff = dprime_2 - pole_delta_2
    return Fraction(0) if diff == 0 else Fraction(2, diff)


# ---------------------------------------------------------------- vectorised versions (numpy int64 arrays)
import numpy as _np


def counts_v(tj, tj12, tj43, order):
    """number of poles of each type: int64 array (4, *shape) (pole_range_k, clipped at 0)"""
    tj, order = _np.broadcast_arrays(_np.asarray(tj, _np.int64), _np.asarray(order, _np.int64))
    bos = tj % 2 == 0
    h, o2, o21 = tj // 2, order // 2, (order + 1) // 2
    small = _np.minimum(h, min(tj12 // 2, tj43 // 2))
    c = _np.stack([order, _np.minimum(h, order), _np.where(bos, o2, _np.minimum(small, o2)),
                   _np.where(bos, _np.minimum(small, o21), o21)])
    return _np.maximum(c, 0)


def pole_at_v(tj, tj12, tj43, order, idx):
    """(type, k) of the idx-th pole of poles(tj, ., ., order)"""
    tj, order, idx = _np.broadcast_arrays(*[_np.asarray(a, _np.int64) for a in (tj, order, idx)])
    c = _np.cumsum(counts_v(tj, tj12, tj43, order), 0)
    t = (idx >= c[0]).astype(_np.int64) + (idx >= c[1]) + (idx >= c[2])
    start = _np.where(t == 0, 0, _np.take_along_axis(c, _np.maximum(t - 1, 0)[None], 0)[0])
    return t, idx - start + 1


def position_v(tj, tj12, tj43, order, t, k):
    tj, order, t, k = _np.broadcast_arrays(*[_np.asarray(a, _np.int64) for a in (tj, order, t, k)])
    c = _np.cumsum(counts_v(tj, tj12, tj43, order), 0)
    start = _np.where(t == 0, 0, _np.take_along_axis(c, _np.maximum(t - 1, 0)[None], 0)[0])
    return start + k - 1


def shift_v(t, k): return _np.where(t < 2, k, _np.where(t == 2, 2 * k, 2 * k - 1))


def pole_two_j_v(t, k, tj): return _np.where(t == 0, tj + 2 * k, _np.where(t == 1, tj - 2 * k, tj))


def delta_2_v(t, k, tj):
    return _np.choose(t, [2 - tj - 2 * k, 4 + tj - 2 * k, 3 - 2 * k, 4 - 2 * k])


def j_excess_v(tj, two_j_max): return _np.where(tj > two_j_max, (tj - two_j_max) // 2, 0)
