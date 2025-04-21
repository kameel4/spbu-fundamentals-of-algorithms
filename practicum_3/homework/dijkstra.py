from typing import Any, Protocol
from itertools import combinations
import heapq
from collections import defaultdict

import numpy as np
import networkx as nx

from src.plotting.graphs import plot_graph, plot_network_via_plotly
from src.common import AnyNxGraph

def dijkstra(graph: AnyNxGraph, source: Any) -> dict[Any, float]:
    distance = {node: float('inf') for node in graph.nodes}
    distance[source] = 0

    # Priority queue to select node with smallest distance
    queue = [(0, source)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # If a shorter path to the current node has been found, continue
        if current_distance > distance[current_node]:
            continue
        for neighbor, attributes in graph[current_node].items():
            weight = attributes.get("weight", 1)
            distance_to_neighbor = current_distance + weight

            # Update if a shorter path is found
            if distance_to_neighbor < distance[neighbor]:
                distance[neighbor] = distance_to_neighbor
                heapq.heappush(queue, (distance_to_neighbor, neighbor))

    return distance

def dijkstra_all_paths(graph, start):
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0
    predecessors = defaultdict(list)
    heap = [(0, start)]
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > distances[u]:
            continue
        for v in graph.neighbors(u):
            new_dist = current_dist + 1
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = [u]
                heapq.heappush(heap, (new_dist, v))
            elif new_dist == distances[v]:
                predecessors[v].append(u)
    return distances, predecessors

def get_all_paths(predecessors, start, end):
    paths = []
    def dfs(node, path):
        if node == start:
            paths.append(list(reversed(path)))
            return
        for pred in predecessors[node]:
            dfs(pred, path + [pred])
    dfs(end, [end])
    return paths


G_karate = nx.karate_club_graph()

# paths_from_zero_node = dijkstra(G_karate, 0)
# for i in range(34):
#     print(f"shortest path to {i} node: {paths_from_zero_node[i]}")
# print("\n\n\n")

# print(list(nx.all_pairs_dijkstra(G_karate))[0])
# print(list(nx.all_shortest_paths(G_karate, 0, 33)))

all_paths = dict()
for start_node in range(0, 34):
    distances, predcessors = dijkstra_all_paths(G_karate, start_node)
    all_paths_from_node = [get_all_paths(predcessors, start_node, end_node) for end_node in range(34)]
    all_paths[start_node] = all_paths_from_node

print(all_paths[0][33])
