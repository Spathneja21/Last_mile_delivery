#!/usr/bin/env python3
# Road-graph routing library used by map_app.py: parses/caches a local OSM
# extract into a graph and computes shortest, densified road paths between
# lat/lon points for the delivery robot to follow.
"""
Lightweight local road-network router built from a saved .osm.pbf extract.

Uses pyosmium (already installed) + networkx (already installed) rather than
osmnx/pyrosm — the extract this session is working with is tiny (~800 nodes,
38 ways), so a hand-rolled graph is simpler and avoids pulling in
geopandas/pandas/shapely just to route ~26 road segments.

Only nodes that belong to a way tagged `highway=*` are added to the graph —
the raw .pbf also contains unrelated nodes (building corners, POIs) that
aren't valid snap targets for road routing.

The parsed graph is pickled to a cache file next to the source .pbf so
subsequent runs load instantly without re-parsing — this is the "saved map"
mechanism: parse once, reuse indefinitely.
"""
import math
import pickle
import os

import osmium
import networkx as nx

EARTH_RADIUS_M = 6371000.0  # mean Earth radius, used by the haversine distance formula


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class _RoadGraphHandler(osmium.SimpleHandler):
    """Builds a networkx.Graph from highway-tagged ways only."""

    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()

    def way(self, w):
        if 'highway' not in w.tags:
            return
        prev = None
        for n in w.nodes:
            if not n.location.valid():
                continue
            node_id = n.ref
            lat, lon = n.location.lat, n.location.lon
            self.graph.add_node(node_id, lat=lat, lon=lon)
            if prev is not None:
                dist = haversine_m(prev[1], prev[2], lat, lon)
                self.graph.add_edge(prev[0], node_id, weight=dist)
            prev = (node_id, lat, lon)  # (id, lat, lon) of the previous node in this way, for the next edge


def build_graph(pbf_path):
    handler = _RoadGraphHandler()
    handler.apply_file(pbf_path, locations=True)
    return handler.graph


def load_graph(pbf_path, cache_path=None):
    """Load from cache_path if present (and newer than the source .pbf),
    otherwise parse the .pbf and write the cache for next time."""
    if cache_path is None:
        cache_path = pbf_path + '.graph.pkl'

    if (os.path.exists(cache_path)
            and os.path.getmtime(cache_path) >= os.path.getmtime(pbf_path)):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    graph = build_graph(pbf_path)
    with open(cache_path, 'wb') as f:
        pickle.dump(graph, f)
    return graph


def nearest_node(graph, lat, lon):
    """Brute-force nearest node search — fine at this graph size (hundreds
    of nodes); would need a spatial index (e.g. a k-d tree) for a
    city-scale extract."""
    best_id, best_dist = None, float('inf')
    for node_id, data in graph.nodes(data=True):
        d = haversine_m(lat, lon, data['lat'], data['lon'])
        if d < best_dist:
            best_id, best_dist = node_id, d
    return best_id, best_dist


def route_latlon(graph, start_latlon, end_latlon):
    """Shortest path between two lat/lon points, snapped to the graph.
    Returns an ordered list of (lat, lon) tuples along real roads, or None
    if no path exists (disconnected graph components)."""
    start_id, _ = nearest_node(graph, *start_latlon)
    end_id, _ = nearest_node(graph, *end_latlon)
    try:
        node_path = nx.shortest_path(graph, start_id, end_id, weight='weight')
    except nx.NetworkXNoPath:
        return None
    return [(graph.nodes[n]['lat'], graph.nodes[n]['lon']) for n in node_path]


def densify_latlon(path_latlon, spacing_m=1.5):
    """Subdivide an ordered list of (lat, lon) points so consecutive points are
    at most ~spacing_m apart, linearly interpolating along each leg.

    Dijkstra road nodes in this extract can be ~15 m apart; the fusion
    navigator needs a fine reference line so cross-track deviation (how far the
    robot has been pushed off the planned path by an obstacle detour) is
    meaningful. Interpolation is linear in lat/lon — fine at these robot-scale
    (tens of metres) leg lengths, same flat-earth assumption the navigators'
    latlon_to_local_xy() already makes.

    Endpoints of every leg are always preserved; only intermediate points are
    added, so a densified path still passes exactly through every original
    node/waypoint."""
    if len(path_latlon) < 2 or spacing_m <= 0:
        return list(path_latlon)

    dense = [tuple(path_latlon[0])]
    for (lat1, lon1), (lat2, lon2) in zip(path_latlon, path_latlon[1:]):
        leg_m = haversine_m(lat1, lon1, lat2, lon2)
        n_seg = max(1, int(math.ceil(leg_m / spacing_m)))  # number of sub-segments this leg is split into
        for k in range(1, n_seg + 1):        # skip k=0 (already the last point added)
            t = k / n_seg
            dense.append((lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t))
    return dense


def snap_path(graph, waypoints):
    """Take an ordered list of raw (lat, lon) clicks and return a dense,
    road-following list by routing between each consecutive pair. Falls
    back to passing a leg through unsnapped if no road path connects it
    (e.g. click was outside the mapped area) rather than dropping it."""
    if len(waypoints) < 2:
        return list(waypoints)

    dense_path = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        leg = route_latlon(graph, a, b)
        if leg is None:
            dense_path.append(b)          # no road connection found — straight fallback
        else:
            dense_path.extend(leg[1:])    # skip leg[0], already the last point added
    return dense_path
