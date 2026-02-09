
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import networkx as nx


DEFAULT_CFG = {
    "connectivity": 8,
    "spur_length": 5,
    "junction_degree": 3,
}


def _neighbors(connectivity=8):
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    raise ValueError("connectivity must be 4 or 8")


def skeleton_to_graph(skel, connectivity=8):
    coords = list(map(tuple, np.argwhere(skel)))
    pos2id = {p: i for i, p in enumerate(coords)}
    G = nx.Graph()
    for p, i in pos2id.items():
        G.add_node(i, pos=p)
    offsets = _neighbors(connectivity)
    for (r, c), i in pos2id.items():
        for dr, dc in offsets:
            nb = (r + dr, c + dc)
            j = pos2id.get(nb)
            if j is not None:
                G.add_edge(i, j)
    return G


def prune_spurs(G, min_length=5):
    if min_length <= 0:
        return G
    G = G.copy()
    changed = True
    while changed:
        changed = False
        endpoints = [n for n, d in G.degree() if d == 1]
        for ep in endpoints:
            if ep not in G:
                continue
            path = [ep]
            prev = None
            cur = ep
            while True:
                nbrs = [n for n in G.neighbors(cur) if n != prev]
                if not nbrs:
                    break
                if len(nbrs) > 1:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if G.degree(cur) != 2:
                    break
                if len(path) > min_length:
                    break
            if len(path) <= min_length and G.degree(cur) != 2:
                # remove short spur (keep junction if present)
                to_remove = path[:-1] if G.degree(cur) >= 3 else path
                for n in to_remove:
                    if n in G:
                        G.remove_node(n)
                        changed = True
    return G


def cluster_junctions(G, deg_thresh=3):
    junction_nodes = [n for n, d in G.degree() if d >= deg_thresh]
    if not junction_nodes:
        return {}, {}
    sub = G.subgraph(junction_nodes)
    comps = list(nx.connected_components(sub))
    junctions = {}
    junction_map = {}
    for jid, comp in enumerate(comps):
        coords = np.array([G.nodes[n]["pos"] for n in comp], dtype=float)
        centroid = tuple(coords.mean(axis=0))
        junctions[jid] = {"nodes": set(comp), "pos": centroid}
        for n in comp:
            junction_map[n] = jid
    return junctions, junction_map


def simplify_graph(G, junctions, junction_map):
    SG = nx.MultiGraph()

    next_id = 0
    junction_id_map = {}
    terminal_for_pixel = {}

    for jid, data in junctions.items():
        nid = next_id
        next_id += 1
        junction_id_map[jid] = nid
        SG.add_node(nid, type="junction", pos=data["pos"], pixels=set(data["nodes"]))
        for n in data["nodes"]:
            terminal_for_pixel[n] = nid

    endpoint_nodes = {n for n, d in G.degree() if d == 1 and n not in junction_map}
    for n in endpoint_nodes:
        nid = next_id
        next_id += 1
        SG.add_node(nid, type="endpoint", pos=G.nodes[n]["pos"], pixels={n})
        terminal_for_pixel[n] = nid

    def is_terminal(n):
        return n in junction_map or n in endpoint_nodes

    visited_edges = set()
    for start_pixel, start_tid in terminal_for_pixel.items():
        for nb in G.neighbors(start_pixel):
            if start_pixel in junction_map and nb in junction_map:
                if junction_map[nb] == junction_map[start_pixel]:
                    continue
            ekey = (min(start_pixel, nb), max(start_pixel, nb))
            if ekey in visited_edges:
                continue
            path = [start_pixel, nb]
            visited_edges.add(ekey)
            prev = start_pixel
            cur = nb
            while True:
                if is_terminal(cur):
                    break
                nbrs = [x for x in G.neighbors(cur) if x != prev]
                if not nbrs:
                    break
                if len(nbrs) > 1:
                    # choose a degree-2 continuation if possible
                    cand = [x for x in nbrs if G.degree(x) == 2 and x not in junction_map]
                    nxt = cand[0] if cand else nbrs[0]
                else:
                    nxt = nbrs[0]
                ekey = (min(cur, nxt), max(cur, nxt))
                if ekey in visited_edges:
                    break
                visited_edges.add(ekey)
                path.append(nxt)
                prev, cur = cur, nxt

            end_pixel = cur
            if end_pixel in junction_map:
                end_tid = junction_id_map[junction_map[end_pixel]]
            elif end_pixel in endpoint_nodes:
                end_tid = terminal_for_pixel[end_pixel]
            else:
                continue

            if start_tid == end_tid:
                continue
            coords = [G.nodes[n]["pos"] for n in path]
            SG.add_edge(start_tid, end_tid, path=coords)

    return SG


def _unit(vec):
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([1.0, 0.0])
    return vec / norm


def _edge_direction_at_node(path, at_start=True):
    if len(path) < 2:
        return np.array([1.0, 0.0])
    if at_start:
        p0 = path[0]
        p1 = path[1]
    else:
        p0 = path[-1]
        p1 = path[-2]
    dy = p1[0] - p0[0]
    dx = p1[1] - p0[1]
    return _unit([dx, dy])


def pair_edges_at_crossings(SG):
    pairing = {}
    for node, ndata in SG.nodes(data=True):
        if ndata.get("type") != "junction":
            continue
        incident = []
        for u, v, k, edata in SG.edges(node, keys=True, data=True):
            at_start = (node == u)
            vec = _edge_direction_at_node(edata["path"], at_start=at_start)
            incident.append(((u, v, k), vec))
        if len(incident) < 2:
            continue
        unused = set(range(len(incident)))
        while len(unused) >= 2:
            i = unused.pop()
            j = min(unused, key=lambda idx: np.dot(incident[i][1], incident[idx][1]))
            unused.remove(j)
            e1 = incident[i][0]
            e2 = incident[j][0]
            pairing[(node, _canon_edge(*e1))] = _canon_edge(*e2)
            pairing[(node, _canon_edge(*e2))] = _canon_edge(*e1)
    return pairing


def _canon_edge(u, v, k):
    return (u, v, k) if u <= v else (v, u, k)


def trace_curve(SG, pairing):
    traversals = []
    used_edges = set()
    edges = [(_canon_edge(u, v, k)) for u, v, k in SG.edges(keys=True)]

    for edge in edges:
        if edge in used_edges:
            continue
        u, v, k = edge
        start_edge = edge
        start_node = u
        current_edge = edge
        current_node = start_node
        crossings = []
        edge_order = []
        guard = 0
        while True:
            if current_edge in used_edges and (current_edge != start_edge or current_node != start_node):
                break
            used_edges.add(current_edge)
            edge_order.append(current_edge)
            u, v, k = current_edge
            next_node = v if current_node == u else u
            if SG.nodes[next_node]["type"] == "junction":
                crossings.append(next_node)
                partner = pairing.get((next_node, current_edge))
                if partner is None:
                    break
                current_node = next_node
                current_edge = partner
            else:
                break
            guard += 1
            if current_edge == start_edge and current_node == start_node:
                break
            if guard > SG.number_of_edges() * 4:
                break
        traversals.append({"crossings": crossings, "edges": edge_order})
    return traversals


def _edge_angle_at_node(path, at_start=True):
    if len(path) < 2:
        return 0.0
    if at_start:
        p0 = path[0]
        p1 = path[1]
    else:
        p0 = path[-1]
        p1 = path[-2]
    dy = p1[0] - p0[0]
    dx = p1[1] - p0[1]
    return math.atan2(dy, dx)


def build_pd(SG, edge_labels, crossing_id_map):
    pd = []
    for node, ndata in SG.nodes(data=True):
        if ndata.get("type") != "junction":
            continue
        incident = []
        for u, v, k, edata in SG.edges(node, keys=True, data=True):
            ekey = _canon_edge(u, v, k)
            label = edge_labels.get(ekey)
            if label is None:
                continue
            angle = _edge_angle_at_node(edata["path"], at_start=(node == u))
            incident.append((angle, label))
        incident.sort(key=lambda t: t[0])
        arcs = tuple(label for _, label in incident)
        pd.append((crossing_id_map[node], arcs))
    return pd


def extract_gauss_code(skel, img_gray=None, *, cfg=None, return_debug=False):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    G = skeleton_to_graph(skel, connectivity=cfg["connectivity"])
    if G.number_of_nodes() == 0:
        empty_pd = {"crossings": [], "edge_labels": {}}
        return ([], empty_pd) if not return_debug else ([], empty_pd, {"graph": G})
    G = prune_spurs(G, min_length=cfg["spur_length"])
    junctions, junction_map = cluster_junctions(G, deg_thresh=cfg["junction_degree"])

    if not junctions:
        empty_pd = {"crossings": [], "edge_labels": {}}
        if return_debug:
            return [], empty_pd, {"graph": G}
        return [], empty_pd

    SG = simplify_graph(G, junctions, junction_map)
    pairing = pair_edges_at_crossings(SG)
    traversals = trace_curve(SG, pairing)

    # assign crossing IDs
    junction_nodes = [n for n, d in SG.nodes(data=True) if d.get("type") == "junction"]
    crossing_id_map = {n: i + 1 for i, n in enumerate(junction_nodes)}

    # label all edges for PD construction
    all_edges = [_canon_edge(u, v, k) for u, v, k in SG.edges(keys=True)]
    edge_labels = {edge: i + 1 for i, edge in enumerate(all_edges)}

    # build gauss from first traversal (single component expected)
    gauss = []
    if traversals:
        t0 = traversals[0]
        for node in t0["crossings"]:
            gauss.append(crossing_id_map[node])

    pd = {
        "crossings": build_pd(SG, edge_labels, crossing_id_map),
        "edge_labels": edge_labels,
    }

    if return_debug:
        debug = {
            "graph": G,
            "simplified": SG,
            "pairing": pairing,
            "traversals": traversals,
            "crossing_id_map": crossing_id_map,
        }
        return gauss, pd, debug
    return gauss, pd
