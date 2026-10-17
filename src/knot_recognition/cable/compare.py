"""compare.py - Comparing reported types with labels."""


def _parse_link(key: str):
    import ast
    body = key[len("link["):-1]
    types, split, lk = body.split("|")
    return types.split(","), split, ast.literal_eval(lk.split("=", 1)[1])


def same_link(a: str, b: str) -> bool:
    """Links are equal up to renumbering their components (the order of components in an image
    is arbitrary): same component types, same split verdict, and a permutation of components
    that maps one linking matrix onto the other."""
    import itertools
    if not (a.startswith("link") and b.startswith("link")):
        return False
    ta, sa, la = _parse_link(a)
    tb, sb, lb = _parse_link(b)
    if sa != sb or len(ta) != len(tb):
        return False
    n = len(ta)
    for p in itertools.permutations(range(n)):
        if all(ta[p[i]] == tb[i] for i in range(n)) and all(la[p[i]][p[j]] == lb[i][j] for i in range(n) for j in range(n)):
            return True
    return False


def same_answer(answer, gt) -> bool:
    """Exact type match (chirality included); links up to renumbering of components."""
    if answer is None:
        return False
    if answer.startswith("link") and gt.startswith("link"):
        return same_link(answer, gt)
    return answer == gt
