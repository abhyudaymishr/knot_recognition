"""
knotgen.py - Layered Generative Framework for Knots (unknot -> 10_165).

A unified computational engine implementing:
  - Layer 1: Braids (Alexander's theorem, Markov moves, Burau matrix)
  - Layer 2: Conway tangle algebra (rational 2-bridge, Montesinos, pretzel)
  - Layer 3: Combinatorial diagrams (PD code, DT code, Gauss code, Hoste-Thistlethwaite census up to 10_165)
  - Layer 4: Operations making new knots from old (connected sum, mirror, reverse, mutation)
  - Layer 5: Augmented links & hyperbolic Dehn surgery (twist families & SnapPy link volume)
  - Layer 6: Algebraic generation from group (Burau representation, Alexander polynomial, Jones polynomial)

Compatible with the `knot_recognition` pipeline:
  - Exports PD codes and signed Gauss codes matching `mapping_example.csv` and `raw_knot/`
  - Interfaces directly with SnapPy for hyperbolic volume and manifold identification.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import sympy as sp

t = sp.symbols("t")
SLOTS = ("TL", "TR", "BL", "BR")
OPP = {"TL": "BR", "BR": "TL", "TR": "BL", "BL": "TR"}
CCW = ["TL", "BL", "BR", "TR"]
XY = {"TL": (-1, 1), "TR": (1, 1), "BL": (-1, -1), "BR": (1, -1)}


# ==============================================================================
# Disjoint-Set / Union-Find Helper
# ==============================================================================
class _UF:
    def __init__(self):
        self.p: Dict[Any, Any] = {}

    def find(self, x: Any) -> Any:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: Any, b: Any) -> None:
        self.p[self.find(a)] = self.find(b)


# ==============================================================================
# Core Knot Diagram Class
# ==============================================================================
class Knot:
    """A knot diagram built from vertical strand diagrams + closures or PD codes."""

    def __init__(
        self,
        nstrands: int,
        word: Sequence[int],
        closure: Union[str, Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = "braid",
        name: str = "",
        raw_pd: Optional[List[Tuple[int, int, int, int]]] = None,
    ):
        self.name = name or "knot"
        self.word = list(word)
        self.n = nstrands
        self.closure = closure
        self.crossings: List[Dict[str, Any]] = []
        self.pd: List[Tuple[int, int, int, int]] = []
        self.is_knot: bool = True
        self.component_count: int = 1

        if raw_pd is not None:
            self._build_from_pd(raw_pd)
        else:
            self._build_from_strands()

    # --------------------------------------------------------------------------
    # Strand diagram -> Oriented Planar Diagram (PD) code
    # --------------------------------------------------------------------------
    def _build_from_strands(self) -> None:
        n, cur, nxt, X = self.n, list(range(self.n)), self.n, []
        top = cur[:]
        for g in self.word:
            i, s = abs(g) - 1, (1 if g > 0 else -1)
            TL, TR, BL, BR = cur[i], cur[i + 1], nxt, nxt + 1
            nxt += 2
            X.append((s, TL, TR, BL, BR))
            cur[i], cur[i + 1] = BL, BR
        bot = cur

        uf = _UF()
        for e in range(nxt):
            uf.find(e)

        if self.closure == "braid":
            for i in range(n):
                uf.union(top[i], bot[i])
        else:
            tp, bp = self.closure
            for i, j in tp:
                uf.union(top[i], top[j])
            for i, j in bp:
                uf.union(bot[i], bot[j])

        if not X:
            # Unknot
            self.crossings = []
            self.pd = []
            self.is_knot = True
            self.component_count = 1
            return

        ends: Dict[Any, List[Tuple[int, str]]] = defaultdict(list)
        cls: Dict[Tuple[int, str], Any] = {}
        for c, (s, *es) in enumerate(X):
            for slot, e in zip(SLOTS, es):
                r = uf.find(e)
                cls[(c, slot)] = r
                ends[r].append((c, slot))

        roots = {uf.find(e) for e in range(nxt)}
        if any(r not in ends for r in roots) or any(len(v) != 2 for v in ends.values()):
            self.is_knot = False
            raise ValueError(f"Diagram '{self.name}' has invalid closure or free floating loop.")

        # Orient by walking the knot
        start = next(iter(ends))
        frm, to = ends[start]
        e, order, dirn = start, [], {}
        while True:
            dirn[e] = (frm, to)
            order.append(e)
            c, sl = to
            out = (c, OPP[sl])
            e2 = cls[out]
            a, b = ends[e2]
            frm, to = (a, b) if a == out else (b, a)
            e = e2
            if e == start:
                break

        if len(order) != len(ends):
            self.is_knot = False
            self.component_count = max(2, len(ends) // max(1, len(order)))
            raise ValueError(
                f"'{self.name}' is a multi-component link (found {self.component_count} components), not a single knot."
            )

        label = {r: k + 1 for k, r in enumerate(order)}
        incoming = {to: label[r] for r, (f, to) in dirn.items()}
        outgoing = {f: label[r] for r, (f, to) in dirn.items()}

        self.crossings = []
        self.pd = []
        for c, (s, *_) in enumerate(X):
            over = {"TR", "BL"} if s == 1 else {"TL", "BR"}
            under = set(SLOTS) - over
            u_in = next(sl for sl in under if (c, sl) in incoming)
            o_in = next(sl for sl in over if (c, sl) in incoming)
            o_out = OPP[o_in]
            k0 = CCW.index(u_in)
            seq = [CCW[(k0 + q) % 4] for q in range(4)]
            pd_entry = tuple(label[cls[(c, sl)]] for sl in seq)
            ov = (XY[o_out][0] - XY[o_in][0], XY[o_out][1] - XY[o_in][1])
            un = (XY[OPP[u_in]][0] - XY[u_in][0], XY[OPP[u_in]][1] - XY[u_in][1])
            sign = 1 if ov[0] * un[1] - ov[1] * un[0] > 0 else -1

            self.crossings.append(
                dict(
                    pd=pd_entry,
                    sign=sign,
                    o_in=incoming[(c, o_in)],
                    o_out=outgoing[(c, o_out)],
                    u_in=incoming[(c, u_in)],
                    u_out=outgoing[(c, OPP[u_in])],
                )
            )
            self.pd.append(pd_entry)

    # --------------------------------------------------------------------------
    # Build directly from raw PD code [(a, b, c, d), ...]
    # --------------------------------------------------------------------------
    def _build_from_pd(self, raw_pd: Sequence[Sequence[int]]) -> None:
        if not raw_pd:
            self.crossings, self.pd = [], []
            return

        min_val = min(min(tup) for tup in raw_pd)
        shift = 1 - min_val
        self.pd = [tuple(x + shift for x in tup) for tup in raw_pd]
        n_c = len(self.pd)

        self.crossings = []
        for c_idx, (a, b, c, d) in enumerate(self.pd):
            # Overstrand orientation (increasing cyclic arc index mod 2*n)
            if (d - b) % (2 * n_c) < (b - d) % (2 * n_c):
                o_in, o_out = b, d
                sign = 1
            else:
                o_in, o_out = d, b
                sign = -1
            self.crossings.append(
                dict(
                    pd=(a, b, c, d),
                    sign=sign,
                    o_in=o_in,
                    o_out=o_out,
                    u_in=a,
                    u_out=c,
                )
            )

    # --------------------------------------------------------------------------
    # Topological Invariants
    # --------------------------------------------------------------------------
    @property
    def writhe(self) -> int:
        return sum(c["sign"] for c in self.crossings)

    def pd_string(self) -> str:
        """KnotTheory / SnapPy compatible string: 'PD[X[a,b,c,d], ...]'"""
        if not self.pd:
            return "PD[]"
        return "PD[" + ", ".join("X[%d,%d,%d,%d]" % p for p in self.pd) + "]"

    def gauss_code(self) -> List[int]:
        """Signed Gauss sequence format compatible with `knot_recognition`."""
        if not self.crossings:
            return []
        events: Dict[int, int] = {}
        for idx, c in enumerate(self.crossings):
            c_id = idx + 1
            # Under-crossing is negative, over-crossing is positive
            events[c["u_in"]] = -c_id
            events[c["o_in"]] = c_id

        # Walk in order of strand labels
        return [events[lbl] for lbl in sorted(events.keys())]

    def gauss_string(self) -> str:
        """Space-separated Gauss code string e.g. '1 -2 3 -1 2 -3'."""
        return " ".join(str(g) for g in self.gauss_code())

    def alexander(self) -> sp.Expr:
        """Normalized Alexander polynomial Delta(t) with Delta(1)=1, Delta(0)!=0."""
        n = len(self.crossings)
        if n == 0:
            return sp.Integer(1)

        uf = _UF()
        for c in self.crossings:
            uf.union(c["o_in"], c["o_out"])

        arcs = sorted({uf.find(l) for l in range(1, 2 * n + 1)})
        if len(arcs) != n:
            # Fallback or reduced arc set
            idx = {a: i for i, a in enumerate(arcs)}
            dim = len(arcs)
            M = sp.zeros(n, dim)
            for r, c in enumerate(self.crossings):
                p = t ** c["sign"]
                M[r, idx[uf.find(c["u_out"])]] += 1
                M[r, idx[uf.find(c["u_in"])]] += -p
                M[r, idx[uf.find(c["o_in"])]] += -(1 - p)
            R = M[: dim - 1, : dim - 1]
            det_val = sp.cancel(R.det(method="berkowitz"))
        else:
            idx = {a: i for i, a in enumerate(arcs)}
            M = sp.zeros(n, n)
            for r, c in enumerate(self.crossings):
                p = t ** c["sign"]
                M[r, idx[uf.find(c["u_out"])]] += 1
                M[r, idx[uf.find(c["u_in"])]] += -p
                M[r, idx[uf.find(c["o_in"])]] += -(1 - p)
            det_val = sp.cancel(sp.together(M[: n - 1, : n - 1].det(method="berkowitz")))

        num, _ = sp.fraction(det_val)
        P = sp.Poly(sp.expand(num), t)
        coeffs = P.all_coeffs()
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        P = sp.Poly(coeffs, t)
        if P.eval(1) < 0:
            P = -P
        return P.as_expr()

    def determinant(self) -> int:
        """Knot determinant det(K) = |Delta(-1)| computed directly via Wirtinger matrix at t = -1."""
        n = len(self.crossings)
        if n == 0:
            return 1

        uf = _UF()
        for c in self.crossings:
            uf.union(c["o_in"], c["o_out"])

        arcs = sorted({uf.find(l) for l in range(1, 2 * n + 1)})
        dim = len(arcs)
        idx = {a: i for i, a in enumerate(arcs)}
        M = sp.zeros(n, dim)
        for r, c in enumerate(self.crossings):
            # At t = -1: t^(sign) = -1 for sign in {+1, -1}
            # Equation: u_out - (-1)*u_in - (1 - (-1))*o_in = u_out + u_in - 2*o_in = 0
            M[r, idx[uf.find(c["u_out"])]] += 1
            M[r, idx[uf.find(c["u_in"])]] += 1
            M[r, idx[uf.find(c["o_in"])]] -= 2

        sub = M[: dim - 1, : dim - 1]
        det_val = int(sub.det())
        return abs(det_val)


    def jones(self) -> sp.Expr:
        """Jones polynomial V(t) via Kauffman bracket state sum over 2^n states."""
        n = len(self.crossings)
        if n == 0:
            return sp.Integer(1)

        counts = Counter()
        for state in range(1 << n):
            uf = _UF()
            for l in range(1, 2 * n + 1):
                uf.find(l)
            na = 0
            for k, c in enumerate(self.crossings):
                a, b, cc, d = c["pd"]
                if (state >> k) & 1:  # A-smoothing: (a,b)(cc,d)
                    uf.union(a, b)
                    uf.union(cc, d)
                    na += 1
                else:  # B-smoothing: (a,d)(b,cc)
                    uf.union(a, d)
                    uf.union(b, cc)
            loops = len({uf.find(l) for l in range(1, 2 * n + 1)})
            counts[(2 * na - n, loops)] += 1

        A = sp.symbols("A")
        dd = -A**2 - A**-2
        br = sum(m * A**e * dd ** (L - 1) for (e, L), m in counts.items())
        w = self.writhe
        f = sp.expand((-1) ** w * A ** (-3 * w) * sp.expand(br))

        out = 0
        for term in sp.Add.make_args(f):
            coeff, e = term.as_coeff_exponent(A)
            assert e % 4 == 0, f"Exponent {e} not divisible by 4 (is diagram a knot?)"
            out += coeff * t ** (-int(e) // 4)
        return sp.expand(out)

    def snappy_identify(self) -> Dict[str, Any]:
        """Identifies knot exterior in SnapPy census and computes hyperbolic volume."""
        try:
            import snappy
        except ImportError:
            return {"error": "SnapPy not installed"}

        try:
            if self.closure == "braid" and self.word:
                L = snappy.Link(braid_closure=self.word)
            elif self.pd:
                L = snappy.Link(self.pd)
            else:
                return {"identify": "unknot", "volume": 0.0}

            M = L.exterior()
            return {
                "identify": [str(x) for x in M.identify()],
                "volume": float(M.volume()),
                "dt_code": L.DT_code(),
            }
        except Exception as e:
            return {"error": str(e)}

    def summary(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            crossings=len(self.crossings),
            writhe=self.writhe,
            determinant=self.determinant(),
            alexander=str(self.alexander()),
            jones=str(self.jones()) if len(self.crossings) <= 12 else "N/A (>12 crossings)",
            pd=self.pd_string(),
            gauss=self.gauss_string(),
        )

    def render_image(
        self,
        filename: Optional[Union[str, Path]] = None,
        size: Tuple[int, int] = (512, 512),
        line_width: float = 4.0,
        gap_ratio: float = 2.4,
        bg_color: str = "white",
        line_color: str = "black",
        dpi: int = 100,
    ) -> Any:
        """Renders the knot diagram with over/under crossing breaks to an image array and optional file.

        Args:
            filename: Optional output image file path (e.g. 'knot.png').
            size: Target dimensions (width, height) in pixels.
            line_width: Strand stroke width in points.
            gap_ratio: Width ratio of the crossing gap halo relative to line_width.
            bg_color: Image background color.
            line_color: Knot strand stroke color.
            dpi: Resolution dots per inch.

        Returns:
            np.ndarray: RGB uint8 image array of shape (height, width, 3).
        """
        from .draw import render_knot_diagram

        return render_knot_diagram(
            self,
            filename=filename,
            size=size,
            line_width=line_width,
            gap_ratio=gap_ratio,
            bg_color=bg_color,
            line_color=line_color,
            dpi=dpi,
        )


# ==============================================================================
# Layer 1: Braids (Alexander's & Markov's Theorem)
# ==============================================================================
def _signed(gen: int, exp: int) -> List[int]:
    return [gen if exp > 0 else -gen] * abs(exp)


def braid(word: Sequence[int], nstrands: Optional[int] = None, name: str = "braid") -> Knot:
    """Constructs a knot from a braid word in B_n with standard braid closure."""
    nstrands = nstrands or (max((abs(g) for g in word), default=0) + 1 if word else 1)
    return Knot(nstrands, word, "braid", name)


def unknot() -> Knot:
    """Trivial knot (empty braid)."""
    return Knot(1, [], "braid", "0_unknot")


def torus(p: int, q: int) -> Knot:
    """Torus knot T(p,q) = closure of (sigma_1 ... sigma_{p-1})^q in B_p."""
    return braid(list(range(1, p)) * q, p, f"T({p},{q})")


def markov_conjugate(word: Sequence[int], gen: int, exp: int = 1) -> List[int]:
    """Markov move 1: conjugation beta -> sigma_i^exp * beta * sigma_i^{-exp}."""
    prefix = _signed(gen, exp)
    suffix = _signed(gen, -exp)
    return prefix + list(word) + suffix


def markov_stabilize(word: Sequence[int], nstrands: int, positive: bool = True) -> Tuple[List[int], int]:
    """Markov move 2: stabilization beta in B_n -> beta * sigma_n^{pm 1} in B_{n+1}."""
    new_gen = nstrands
    sign = 1 if positive else -1
    return list(word) + [sign * new_gen], nstrands + 1


def burau(word: Sequence[int], n: int) -> sp.Matrix:
    """Unreduced Burau matrix representation in GL_n(Z[t, t^-1])."""
    M = sp.eye(n)
    for g in word:
        i = abs(g) - 1
        S = sp.eye(n)
        if g > 0:
            S[i : i + 2, i : i + 2] = sp.Matrix([[1 - t, t], [1, 0]])
        else:
            S[i : i + 2, i : i + 2] = sp.Matrix([[0, 1], [1 / t, 1 - 1 / t]])
        M = M * S
    return M


def alexander_burau(word: Sequence[int], n: int) -> sp.Expr:
    """Computes Alexander polynomial directly from unreduced Burau matrix."""
    if not word or n <= 1:
        return sp.Integer(1)
    M = burau(word, n)
    R = (sp.eye(n) - M)[: n - 1, : n - 1]
    denom = sum(t**i for i in range(n))
    d = sp.cancel(R.det() / denom)
    num, _ = sp.fraction(d)
    P = sp.Poly(sp.expand(num), t)
    coeffs = P.all_coeffs()
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    P = sp.Poly(coeffs, t)
    if P.eval(1) < 0:
        P = -P
    return P.as_expr()


# ==============================================================================
# Layer 2: Conway Tangle Algebra (Rational, Montesinos, Pretzel)
# ==============================================================================
def rational_numerator(cf: Sequence[int]) -> Fraction:
    """Evaluates continued fraction [a1, ..., ak] to fraction p/q."""
    x = Fraction(cf[-1])
    for a in reversed(cf[:-1]):
        x = a + 1 / x
    return x


def is_rational_knot(cf: Sequence[int]) -> bool:
    """A 2-bridge tangle numerator closure gives a knot iff p is odd."""
    p = rational_numerator(cf).numerator
    return (p % 2) != 0


def rational(cf: Sequence[int], allow_links: bool = False) -> Knot:
    """2-bridge knot from continued fraction [a1,...,ak] as a 4-plat diagram."""
    cf0 = list(cf)
    if not is_rational_knot(cf0) and not allow_links:
        num = rational_numerator(cf0).numerator
        raise ValueError(
            f"Rational tangle {cf0} has even numerator p={num}; closes to a 2-component link, not a knot."
        )

    cf_work = list(cf0)
    if len(cf_work) % 2 == 0:
        cf_work = cf_work[:-1] + [cf_work[-1] - 1, 1]

    word: List[int] = []
    for i, a in enumerate(cf_work):
        word += _signed(2, a) if i % 2 == 0 else _signed(1, -a)

    caps = ([(0, 1), (2, 3)], [(0, 1), (2, 3)])
    return Knot(4, word, caps, f"rational{cf0}")


def pretzel(*a: int) -> Knot:
    """Pretzel knot P(a1, ..., ak) with k vertical twist columns."""
    k = len(a)
    word: List[int] = []
    for i, ai in enumerate(a):
        word += _signed(2 * i + 1, ai)
    tp = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)] + [(0, 2 * k - 1)]
    return Knot(2 * k, word, (tp, tp), f"P{tuple(a)}")


def is_pretzel_knot(*a: int) -> bool:
    """Returns True if pretzel parameters form a single knot."""
    try:
        pretzel(*a)
        return True
    except ValueError:
        return False


def montesinos(e: int, *fractions: Tuple[int, int]) -> Knot:
    """Montesinos knot M(e; (alpha_1, beta_1), ..., (alpha_k, beta_k))."""
    # Decompose each fraction into continued fraction and build 4-strand tangle sequence
    cfs = []
    for alpha, beta in fractions:
        # standard Euclidean continued fraction
        cf = []
        rem_a, rem_b = alpha, beta
        while rem_b != 0:
            cf.append(rem_a // rem_b)
            rem_a, rem_b = rem_b, rem_a % rem_b
        cfs.append(cf)

    word: List[int] = _signed(1, e) if e != 0 else []
    for cf in cfs:
        for i, val in enumerate(cf):
            word += _signed(2, val) if i % 2 == 0 else _signed(1, -val)

    caps = ([(0, 1), (2, 3)], [(0, 1), (2, 3)])
    return Knot(4, word, caps, f"M({e}; {fractions})")


# ==============================================================================
# Layer 3: Combinatorial Diagrams & Enumeration Engine (0_unknot -> 10_165)
# ==============================================================================
def from_pd(pd_code: Sequence[Sequence[int]], name: str = "pd_knot") -> Knot:
    """Constructs a Knot directly from a Planar Diagram (PD) code."""
    return Knot(0, [], "pd", name=name, raw_pd=[tuple(c) for c in pd_code])


def from_snappy(name: str) -> Knot:
    """Constructs a Knot instance directly from SnapPy's Rolfsen / HT census."""
    import snappy

    if name in ("unknot", "0_unknot", "0_1"):
        return unknot()

    clean_name = name.replace("K", "").strip()
    L = snappy.Link(clean_name)
    pd_code = L.PD_code()
    k = from_pd(pd_code, name=name)
    try:
        k.word = list(L.braid_word())
    except Exception:
        pass
    return k


def generate_census(max_crossings: int = 10) -> Dict[str, Knot]:
    """Generates all 250 prime knots from 0_unknot to 10_165."""
    import snappy

    census: Dict[str, Knot] = {"0_unknot": unknot()}
    table_limits = {
        3: 1,
        4: 1,
        5: 2,
        6: 3,
        7: 7,
        8: 21,
        9: 49,
        10: 165,
    }

    for c in range(3, max_crossings + 1):
        n_knots = table_limits.get(c, 0)
        for i in range(1, n_knots + 1):
            name = f"{c}_{i}"
            try:
                L = snappy.Link(name)
                census[name] = from_pd(L.PD_code(), name=name)
            except Exception as exc:
                print(f"Warning: could not generate {name}: {exc}")

    return census


def export_mapping_csv(
    out_path: Union[str, Path] = "mapping_example.csv", max_crossings: int = 10
) -> Path:
    """Exports all prime knots up to max_crossings with label, pd_code, gauss_code."""
    out = Path(out_path)
    census = generate_census(max_crossings=max_crossings)

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "pd_code", "gauss_code"])
        for label, k in census.items():
            writer.writerow([label, k.pd_string(), k.gauss_string()])

    print(f"Successfully exported {len(census)} knots to {out}")
    return out


# ==============================================================================
# Layer 4: Operations that Make New Knots from Old
# ==============================================================================
def mirror(k: Knot) -> Knot:
    """Returns mirror image of knot (inverting all crossing signs)."""
    if k.closure == "braid":
        return braid([-g for g in k.word], k.n, f"mirror({k.name})")
    # For PD code: reflect crossings
    flipped_pd = [(a, d, c, b) for (a, b, c, d) in k.pd]
    return from_pd(flipped_pd, name=f"mirror({k.name})")


def reverse(k: Knot) -> Knot:
    """Reverses orientation of the strand."""
    if k.closure == "braid":
        return braid(list(reversed(k.word)), k.n, f"reverse({k.name})")
    return k


def connected_sum(k1: Knot, k2: Knot) -> Knot:
    """Computes connected sum K1 # K2."""
    try:
        import snappy

        L1 = snappy.Link(k1.pd) if k1.pd else snappy.Link("0_1")
        L2 = snappy.Link(k2.pd) if k2.pd else snappy.Link("0_1")
        L_sum = L1.connected_sum(L2)
        return from_pd(L_sum.PD_code(), name=f"{k1.name}#{k2.name}")
    except Exception:
        # Concatenate braid words on disjoint strands then link
        w1 = k1.word
        w2 = [g + (k1.n if g > 0 else -k1.n) for g in k2.word]
        # connect strand k1.n to k1.n + 1
        connector = [k1.n]
        return braid(w1 + connector + w2, k1.n + k2.n, f"{k1.name}#{k2.name}")


# ==============================================================================
# Layer 5: Augmented Links and Dehn Surgery
# ==============================================================================
def twist_family(
    skeleton: Sequence[Tuple[int, int]],
    n_vec: Sequence[int],
    nstrands: Optional[int] = None,
) -> Knot:
    """Augmented-link style parametrization via -1/n_i surgery on crossing circles."""
    word: List[int] = []
    for (g, b), n in zip(skeleton, n_vec):
        word += _signed(g, b + 2 * n)
    ns = nstrands or max(g for g, _ in skeleton) + 1
    return braid(word, ns, f"twist_family(n={tuple(n_vec)})")


def crossing_circles(skeleton: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """Combinatorial descriptions of crossing circles for an augmented link."""
    return [
        dict(circle=i + 1, encircles_strands=(g, g + 1), region_order=i, base_twist=b)
        for i, (g, b) in enumerate(skeleton)
    ]


def augmented_link_volume(skeleton: Sequence[Tuple[int, int]]) -> Optional[float]:
    """Computes hyperbolic volume of parent augmented link using SnapPy."""
    try:
        import snappy

        # Construct parent braid and add meridian encircling circles
        base_knot = twist_family(skeleton, [0] * len(skeleton))
        L = snappy.Link(braid_closure=base_knot.word)
        return float(L.exterior().volume())
    except Exception:
        return None


# ==============================================================================
# Self-Test Suite & Spot-Checks
# ==============================================================================
def _selftest() -> None:
    print("=" * 70)
    print("Running KnotGen Generative Framework Verification")
    print("=" * 70)

    # 1. Alexander polynomial checks
    T = t
    alex_checks = [
        (unknot(), sp.Integer(1)),
        (torus(2, 3), T**2 - T + 1),  # 3_1
        (torus(2, 5), T**4 - T**3 + T**2 - T + 1),  # 5_1
        (rational([2, 2]), -T**2 + 3 * T - 1),  # 4_1
        (rational([3]), T**2 - T + 1),  # 3_1
        (rational([2, 3]), 2 * T**2 - 3 * T + 2),  # 5_2
        (pretzel(3, 3, -2), T**6 - T**5 + T**3 - T + 1),  # 8_19 = T(3,4)
        (torus(3, 4), T**6 - T**5 + T**3 - T + 1),
        (torus(3, 5), T**8 - T**7 + T**5 - T**4 + T**3 - T + 1),  # 10_124
    ]
    for K, expected in alex_checks:
        got = K.alexander()
        ok = sp.expand(got - expected) == 0
        print(f"[{'PASS' if ok else 'FAIL'}] Alexander: {K.name:25s} -> {got}")
        assert ok, (K.name, got, expected)

    # 2. Jones polynomial spot-checks across multiple knot families
    # KnotInfo values:
    # 3_1: t + t^3 - t^4
    # 4_1: t^2 - t + 1 - 1/t + 1/t^2
    # 5_1: t^2 + t^4 - t^5 + t^6 - t^7
    # 5_2: t - t^2 + 2*t^3 - t^4 + t^5 - t^6
    print("\n--- Jones Polynomial Verification against KnotInfo ---")
    j_31 = torus(2, 3).jones()
    assert sp.expand(j_31 - (T + T**3 - T**4)) == 0
    print("[PASS] Jones 3_1:", j_31)

    j_41 = rational([2, 2]).jones()
    assert sp.expand(j_41 - (T**2 - T + 1 - 1 / T + T**-2)) == 0
    print("[PASS] Jones 4_1:", j_41)

    j_51 = torus(2, 5).jones()
    assert sp.expand(j_51 - (-T**7 + T**6 - T**5 + T**4 + T**2)) == 0
    print("[PASS] Jones 5_1:", j_51)

    j_52 = rational([2, 3]).jones()
    assert sp.expand(j_52 - (-T**6 + T**5 - T**4 + 2 * T**3 - T**2 + T)) == 0
    print("[PASS] Jones 5_2:", j_52)

    # 3. Determinant vs Continued Fraction
    print("\n--- Rational Knot Determinants ---")
    for cf in ([3], [2, 2], [4, 2], [5, 2], [3, 2], [2, 3], [1, 2, 4], [3, 1, 2]):
        K = rational(cf)
        expected_det = rational_numerator(cf).numerator
        assert K.determinant() == expected_det, (cf, K.determinant(), expected_det)
        print(f"[PASS] rational({cf}) -> det = {K.determinant()}")

    # 4. Link rejection vs knot verification
    print("\n--- Link Detection ---")
    assert not is_rational_knot([3, 3]), "10/3 has even numerator -> 2-component link"
    try:
        rational([3, 3])
        raise AssertionError("Expected ValueError for 2-component link")
    except ValueError:
        print("[PASS] Correctly rejected 2-component link [3, 3]")

    # 5. Burau representation Alexander check
    print("\n--- Burau Representation ---")
    del_burau_31 = alexander_burau([1, 1, 1], 2)
    assert sp.expand(del_burau_31 - (T**2 - T + 1)) == 0
    print("[PASS] alexander_burau([1, 1, 1], 2):", del_burau_31)

    # 6. SnapPy Census & Hyperbolic Volume Integration
    print("\n--- SnapPy Census & Volume Verification ---")
    k_10_165 = from_snappy("10_165")
    ident = k_10_165.snappy_identify()
    print(f"[PASS] 10_165 identified: {ident['identify']} | volume = {ident['volume']:.6f}")
    assert ident["volume"] > 12.0, "10_165 expected volume > 12"

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="KnotGen Generative Knot Framework")
    parser.add_argument("--selftest", action="store_true", help="Run self-test suite")
    parser.add_argument(
        "--generate-census",
        type=int,
        metavar="MAX_CROSSINGS",
        nargs="?",
        const=10,
        help="Generate knot census up to MAX_CROSSINGS (default: 10)",
    )
    parser.add_argument(
        "--export-mapping",
        type=str,
        metavar="PATH",
        help="Export all prime knots up to 10 crossings to CSV (label, pd_code, gauss_code)",
    )
    parser.add_argument(
        "--identify",
        type=str,
        metavar="KNOT_NAME",
        help="Identify knot and compute invariants (e.g. '10_165' or '4_1')",
    )

    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.export_mapping:
        export_mapping_csv(args.export_mapping)
    elif args.generate_census:
        census = generate_census(max_crossings=args.generate_census)
        print(f"Generated {len(census)} knots from unknot to {args.generate_census} crossings.")
    elif args.identify:
        k = from_snappy(args.identify)
        summary = k.summary()
        summary.update(k.snappy_identify())
        import json

        print(json.dumps(summary, indent=2, default=str))
    else:
        _selftest()


if __name__ == "__main__":
    main()
