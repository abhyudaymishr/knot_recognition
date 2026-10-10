#!/usr/bin/env python3
"""
enumerate_knots.py - from-scratch enumeration of prime knots by crossing number (HTW/DT style).

Pipeline for each crossing number n
  1. DT-style Gauss words: crossing j visits position 2j and an odd position  (n! words)
  2. Keep words that are reduced (no nugatory crossing), prime (no connected-sum cut),
     and planar.  Planarity + embedding come from a mod-2 rule on the interlace graph
     (validated against brute-force face tracing for all n <= 7).
  3. Canonicalise under cyclic shift / reversal  ->  one representative per knot SHADOW
  4. Put over/under on every crossing (2^(n-1) per shadow, mirror fixed)
  5. Fingerprint each diagram by (Alexander polynomial, Jones polynomial), identifying mirrors.
  6. A fingerprint is a NEW prime knot iff it is not trivial, not seen at lower n,
     and not the (Alexander, Jones) product of two lower knots (connected sum).
     NB: testing Jones alone is wrong - 8_9 has the same Jones polynomial as 4_1#4_1.

Limits (be aware):
  * Distinct knots with equal (Alexander, Jones) would be merged.  Compare counts with
    1,1,2,3,7,21,49,165 for n=3..10 - any shortfall points at such a collision.
  * Cost grows roughly 6x per crossing.  n<=8 is quick; n=9,10 want --workers and patience.
Usage:  python enumerate_knots.py --max 8 --workers 4 --out knots.json
"""
import argparse, json, sys, time
from itertools import permutations
from fractions import Fraction
from multiprocessing import Pool

EXPECTED = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165}


# ------------------------------------------------------------------ shadows
def _info(w, n):
    first, second = [-1] * n, [-1] * n
    for k, s in enumerate(w):
        if first[s] < 0:
            first[s] = k
        else:
            second[s] = k
    adj = [0] * n
    for u in range(n):
        fu, su = first[u], second[u]
        for v in range(u + 1, n):
            fv, sv = first[v], second[v]
            if fu < fv < su < sv or fv < fu < sv < su:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return first, second, adj


def _is_prime_word(w, N):
    for i in range(N):
        cnt, open_ = {}, 0
        for L in range(1, N - 1):
            s = w[(i + L - 1) % N]
            c = cnt.get(s, 0) + 1
            cnt[s] = c
            open_ += 1 if c == 1 else -1
            if open_ == 0:
                return False          # proper closed cyclic segment => composite or kink
    return True


def _eps(w, n):
    """Handedness of every crossing (relative to its first-visited strand), or None if not planar."""
    first, second, adj = _info(w, n)
    if 0 in adj:
        return None                   # nugatory crossing
    ep, st = [0] * n, [0]
    ep[0] = 1
    while st:
        u = st.pop()
        for v in range(n):
            if (adj[u] >> v) & 1:
                rel = -1 if bin(adj[u] & adj[v]).count("1") % 2 == 0 else 1
                want = ep[u] * rel
                if ep[v] == 0:
                    ep[v] = want
                    st.append(v)
                elif ep[v] != want:
                    return None
    if not all(ep):
        return None
    # non-interlaced pairs must share an even number of interlaced neighbours
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1 and bin(adj[u] & adj[v]).count("1") % 2:
                return None
    eps = [ep[v] * (1 if first[v] % 2 == 0 else -1) for v in range(n)]
    return eps if _planar(w, n, eps, first, second) else None


def _planar(w, n, eps, first, second):
    N = 2 * n
    nxt = [0] * (2 * N)
    for v in range(n):
        Ai, Ao, Bi, Bo = 2 * first[v], 2 * first[v] + 1, 2 * second[v], 2 * second[v] + 1
        cyc = [Ai, Bi, Ao, Bo] if eps[v] == 1 else [Ai, Bo, Ao, Bi]
        for i in range(4):
            nxt[cyc[i]] = cyc[(i + 1) % 4]
    opp = [(2 * ((h // 2 + 1) % N) if h & 1 else 2 * ((h // 2 - 1) % N) + 1) for h in range(2 * N)]
    seen, F = [False] * (2 * N), 0
    for h in range(2 * N):
        if not seen[h]:
            F += 1
            x = h
            while not seen[x]:
                seen[x] = True
                x = nxt[opp[x]]
    return F == n + 2


def _canon(w, N):
    best = None
    for d in (1, -1):
        for s in range(N):
            seen, out = {}, []
            for i in range(N):
                x = w[(s + d * i) % N]
                if x not in seen:
                    seen[x] = len(seen)
                out.append(seen[x])
            tup = tuple(out)
            if best is None or tup < best:
                best = tup
    return best


def shadows(n):
    """All prime, reduced knot shadows with n crossings (up to isotopy/reflection) as (word, eps)."""
    N, found = 2 * n, {}
    for perm in permutations(range(n)):
        w = [0] * N
        for j, pj in enumerate(perm):
            w[2 * j] = j
            w[2 * pj + 1] = j
        _, _, adj = _info(w, n)
        if 0 in adj:
            continue
        if not _is_prime_word(w, N):
            continue
        if _eps(w, n) is None:
            continue
        c = _canon(w, N)
        if c not in found:
            found[c] = None
    out = []
    for c in found:
        eps = _eps(list(c), n)
        assert eps is not None
        out.append((list(c), eps))
    return out


# ------------------------------------------------------------------ diagrams -> PD
def make_pd(word, eps, over):
    """over[v]=1 means the first-visited strand is over. Edge e_k (between positions k,k+1) has label k+1."""
    n, N = len(eps), len(word)
    first, second = [-1] * n, [-1] * n
    for k, s in enumerate(word):
        if first[s] < 0:
            first[s] = k
        else:
            second[s] = k
    X = []
    for v in range(n):
        p1, p2 = first[v], second[v]
        Ain, Aout = (p1 - 1) % N + 1, p1 + 1
        Bin, Bout = (p2 - 1) % N + 1, p2 + 1
        cyc = [Ain, Bin, Aout, Bout] if eps[v] == 1 else [Ain, Bout, Aout, Bin]
        if over[v]:
            start, oi, oo, ui, uo = cyc.index(Bin), Ain, Aout, Bin, Bout
        else:
            start, oi, oo, ui, uo = 0, Bin, Bout, Ain, Aout
        pd = tuple(cyc[(start + q) % 4] for q in range(4))
        sign = eps[v] * (1 if over[v] else -1)
        X.append(dict(pd=pd, sign=sign, o_in=oi, o_out=oo, u_in=ui, u_out=uo))
    return X


# ------------------------------------------------------------------ fast invariants
def _bareiss(M):
    n = len(M)
    M = [row[:] for row in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    M[k], M[r] = M[r], M[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def alexander_fast(X):
    n = len(X)
    par = list(range(2 * n + 1))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for c in X:
        par[find(c["o_in"])] = find(c["o_out"])
    arcs = sorted({find(l) for l in range(1, 2 * n + 1)})
    if len(arcs) != n:
        return None
    idx = {a: i for i, a in enumerate(arcs)}
    rows = []
    for c in X:                                  # entries as (c0, c1): c0 + c1*t
        r = [[0, 0] for _ in range(n)]
        k, i, o = idx[find(c["u_out"])], idx[find(c["u_in"])], idx[find(c["o_in"])]
        if c["sign"] == 1:                       # a_k - t a_i + (t-1) a_o
            r[k][0] += 1; r[i][1] -= 1; r[o][1] += 1; r[o][0] -= 1
        else:                                    # t a_k - a_i - (t-1) a_o  (row scaled by t)
            r[k][1] += 1; r[i][0] -= 1; r[o][1] -= 1; r[o][0] += 1
        rows.append(r)
    m = n - 1
    pts = list(range(n))
    ys = [_bareiss([[rows[a][b][0] + rows[a][b][1] * x for b in range(m)] for a in range(m)]) for x in pts]
    # Lagrange interpolation with exact fractions -> coefficient list (low degree first)
    coeffs = [Fraction(0)] * n
    for i, xi in enumerate(pts):
        num, denom = [Fraction(1)], Fraction(1)
        for j, xj in enumerate(pts):
            if j == i:
                continue
            new = [Fraction(0)] * (len(num) + 1)
            for d, a in enumerate(num):
                new[d + 1] += a
                new[d] -= a * xj
            num = new
            denom *= xi - xj
        for d, a in enumerate(num):
            coeffs[d] += ys[i] * a / denom
    co = [int(c) for c in coeffs]
    while co and co[-1] == 0:
        co.pop()
    k = 0
    while k < len(co) and co[k] == 0:
        k += 1
    co = co[k:]
    if sum(co) < 0:
        co = [-c for c in co]
    return tuple(min(co, co[::-1])), rows


def _rank_mod(rows, n, p, a):
    M = [[(rows[i][j][0] + rows[i][j][1] * a) % p for j in range(n)] for i in range(n)]
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, n) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], -1, p)
        M[r] = [(x * inv) % p for x in M[r]]
        for i in range(n):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(x - f * y) % p for x, y in zip(M[i], M[r])]
        r += 1
    return r


_ORBITS = []
for _p in (3, 5, 7, 11, 13):
    _seen = set()
    for _a in range(2, _p):
        if _a in _seen:
            continue
        _b = pow(_a, -1, _p)
        _seen.update((_a, _b))
        _ORBITS.append((_p, _a, _b))


def module_fingerprint(rows, n):
    """Coranks of the Alexander matrix at t=a (mod p); a and 1/a paired so mirror/reversal-safe."""
    return tuple(tuple(sorted((n - _rank_mod(rows, n, p, a), n - _rank_mod(rows, n, p, b))))
                 for p, a, b in _ORBITS)


_DPOW = {}


def _dpow(L):
    """d^L as {A-exponent: coeff}, d = -A^2 - A^-2."""
    if L in _DPOW:
        return _DPOW[L]
    if L == 0:
        r = {0: 1}
    else:
        p, r = _dpow(L - 1), {}
        for e, c in p.items():
            r[e + 2] = r.get(e + 2, 0) - c
            r[e - 2] = r.get(e - 2, 0) - c
    _DPOW[L] = r
    return r


def jones_fast(X):
    n = len(X)
    pairs = [((c["pd"][0], c["pd"][1], c["pd"][2], c["pd"][3])) for c in X]
    counts = {}
    for state in range(1 << n):
        par = list(range(2 * n + 1))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        na = 0
        for k in range(n):
            a, b, c, d = pairs[k]
            if (state >> k) & 1:
                par[find(a)] = find(b); par[find(c)] = find(d); na += 1
            else:
                par[find(a)] = find(d); par[find(b)] = find(c)
        loops = sum(1 for l in range(1, 2 * n + 1) if find(l) == l)
        key = (2 * na - n, loops)
        counts[key] = counts.get(key, 0) + 1
    w = sum(c["sign"] for c in X)
    br = {}
    for (e, L), m in counts.items():
        for e2, c2 in _dpow(L - 1).items():
            br[e + e2] = br.get(e + e2, 0) + m * c2
    out = {}
    sgn = -1 if w % 2 else 1
    for e, c in br.items():
        if c == 0:
            continue
        e2 = e - 3 * w
        assert e2 % 4 == 0
        out[-e2 // 4] = out.get(-e2 // 4, 0) + sgn * c
    return {e: c for e, c in out.items() if c}


def _mirror(J):
    return {-e: c for e, c in J.items()}


def _mul(J1, J2):
    r = {}
    for e1, c1 in J1.items():
        for e2, c2 in J2.items():
            r[e1 + e2] = r.get(e1 + e2, 0) + c1 * c2
    return {e: c for e, c in r.items() if c}


def _jkey(J):
    a, b = tuple(sorted(J.items())), tuple(sorted(_mirror(J).items()))
    return min(a, b)


def _amul(A1, A2):
    r = [0] * (len(A1) + len(A2) - 1)
    for i, a in enumerate(A1):
        for j, b in enumerate(A2):
            r[i + j] += a * b
    if sum(r) < 0:
        r = [-c for c in r]
    return tuple(min(r, r[::-1]))


TRIVIAL = ((1,), ((0, 1),))


# ------------------------------------------------------------------ driver
def _is_alternating(word, over):
    """Visits along the word must alternate over/under (first visit over iff over[v])."""
    seen, st = set(), []
    for v in word:
        first = v not in seen
        seen.add(v)
        st.append(over[v] if first else 1 - over[v])
    N = len(word)
    return all(st[i] != st[(i + 1) % N] for i in range(N))


def _process_shadow(args):
    word, eps = args
    n = len(eps)
    seen = {}
    for mask in range(1 << (n - 1)):
        over = [1] + [(mask >> (v - 1)) & 1 for v in range(1, n)]
        X = make_pd(word, eps, over)
        res = alexander_fast(X)
        if res is None:
            continue
        A, rows = res
        J = jones_fast(X)
        key = (A, _jkey(J), module_fingerprint(rows, len(X)))
        alt = _is_alternating(word, over)
        if key not in seen:
            seen[key] = (over, J, alt)
        elif alt and not seen[key][2]:
            seen[key] = (seen[key][0], seen[key][1], True)
    return seen


def enumerate_knots(max_n, workers=1, log=print):
    known, jlist, composite, result = set(), [], set(), {}
    trivial_ab = TRIVIAL
    for n in range(3, max_n + 1):
        t0 = time.time()
        sh = shadows(n)
        log(f"n={n}: {len(sh)} prime reduced shadows  ({time.time() - t0:.1f}s)")
        t0 = time.time()
        if workers > 1:
            with Pool(workers) as p:
                parts = p.map(_process_shadow, [(w, e) for w, e in sh], chunksize=1)
        else:
            parts = [_process_shadow((w, e)) for w, e in sh]
        cand = {}
        for si, part in enumerate(parts):
            for key, (over, J, alt) in part.items():
                if key not in cand:
                    cand[key] = (si, over, J, alt)
                elif alt and not cand[key][3]:
                    cand[key] = (si, over, J, True)
        new = []
        for key, (si, over, J, alt) in sorted(cand.items(), key=lambda kv: repr(kv[0])):
            if key[:2] == trivial_ab or key in known:
                continue
            # composite test uses BOTH polynomials; alternating prime diagrams are never composite (Menasco)
            if key[:2] in composite and not alt:
                continue
            new.append((key, si, over, J))
        result[n] = []
        for key, si, over, J in new:
            known.add(key)
            word, eps = sh[si]
            X = make_pd(word, eps, over)
            result[n].append(dict(crossings=n, pd=[list(c["pd"]) for c in X],
                                  alexander=list(key[0]), det=abs(sum(c * (-1) ** i for i, c in enumerate(key[0]))),
                                  jones=[list(p) for p in key[1]], word=word, eps=eps, over=over))
            for A2, J2 in jlist + [(key[0], J)]:
                composite.add((_amul(key[0], A2), _jkey(_mul(J, J2))))
                composite.add((_amul(key[0], A2), _jkey(_mul(J, _mirror(J2)))))
            jlist.append((key[0], J))
        exp = EXPECTED.get(n)
        flag = "" if exp is None else ("  OK" if exp == len(new) else f"  MISMATCH (expected {exp})")
        log(f"n={n}: {len(new)} prime knots{flag}  ({time.time() - t0:.1f}s)")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=7)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="knots.json")
    a = ap.parse_args()
    res = enumerate_knots(a.max, a.workers)
    json.dump(res, open(a.out, "w"))
    print("total prime knots:", sum(len(v) for v in res.values()), "->", a.out)
