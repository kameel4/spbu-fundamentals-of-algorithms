from typing import Any, Protocol
from itertools import combinations
import heapq

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


G_karate = nx.karate_club_graph()

# paths_from_zero_node = dijkstra(G_karate, 0)
# for i in range(34):
#     print(f"shortest path to {i} node: {paths_from_zero_node[i]}")
# print("\n\n\n")

# print(list(nx.all_pairs_dijkstra(G_karate))[0])

# print(dijkstra(G_karate, 0))