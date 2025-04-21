from typing import Any, Protocol
from itertools import combinations

import numpy as np
import networkx as nx

from src.plotting.graphs import plot_graph, plot_network_via_plotly
from src.common import AnyNxGraph

from dijkstra import dijkstra, dijkstra_all_paths, get_all_paths


class CentralityMeasure(Protocol):
    def __call__(self, G: AnyNxGraph) -> dict[Any, float]:
        ...


def closeness_centrality(G: AnyNxGraph) -> dict[Any, float]:
    node_closeness_centrality = dict()
    n = len(G)
    for node in range(n):
        dist_array = dijkstra(G, node)
        node_closeness_centrality[node] = (1 / sum([dist_array[i] for i in range(n)]))
    return node_closeness_centrality



def betweenness_centrality(G: AnyNxGraph) -> dict[Any, float]: 
    node_betweenness_centrality = dict()
    all_paths = dict()
    n = len(G)
    for start_node in range(n):
        distances, predcessors = dijkstra_all_paths(G, start_node)
        all_paths_from_node = [get_all_paths(predcessors, start_node, end_node) for end_node in range(n)]
        all_paths[start_node] = all_paths_from_node
    
    for bridge_node in range(n):
        sigma = 0
        for s in range(n):
            for t in range(s+1, n):
                if (bridge_node != s and bridge_node != t):
                    shortest_paths, shortest_paths_with_bridge = 0, 0
                    for path in all_paths[s][t]:
                        if bridge_node in path:
                            shortest_paths_with_bridge +=1
                        shortest_paths +=1
                    sigma += shortest_paths_with_bridge / shortest_paths
        node_betweenness_centrality[bridge_node] = sigma
    return node_betweenness_centrality
    

def eigenvector_centrality(G: AnyNxGraph) -> dict[Any, float]: 

    ##########################
    ### PUT YOUR CODE HERE ###
    #########################

    pass


def plot_centrality_measure(G: AnyNxGraph, measure: CentralityMeasure) -> None:
    values = measure(G)
    if values is not None:
        plot_graph(G, node_weights=values, figsize=(14, 8), name=measure.__name__)
    else:
        print(f"Implement {measure.__name__}")


if __name__ == "__main__":
    G = nx.karate_club_graph()

    plot_centrality_measure(G, closeness_centrality)
    plot_centrality_measure(G, betweenness_centrality)
    # plot_centrality_measure(G, eigenvector_centrality)

