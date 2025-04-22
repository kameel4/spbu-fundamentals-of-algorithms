from typing import Any, Protocol
from itertools import combinations
import heapq
from collections import defaultdict

import numpy as np
import networkx as nx

from src.plotting.graphs import plot_graph, plot_network_via_plotly
from src.common import AnyNxGraph
from src.linalg import get_numpy_eigenvalues, get_scipy_solution

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

def find_dominant_eigenvector(A, max_iter=100, tol=1e-6):
    eigenvalues = get_numpy_eigenvalues(A)
    lambda_max = np.max(eigenvalues.real)
    
    sigma = lambda_max + 1e-3  
    
    n = A.shape[0]
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    
    A_shifted = A - sigma * np.eye(n)
    
    for _ in range(max_iter):
        # Решаем систему (A - σI)v_new = v_old
        v_new = get_scipy_solution(A_shifted, v)
        
        # Нормируем вектор
        v_new /= np.linalg.norm(v_new)
        
        # Проверяем сходимость
        if np.linalg.norm(v_new - v) < tol:
            break
        
        v = v_new
    
    return v

G_karate = nx.karate_club_graph()

# paths_from_zero_node = dijkstra(G_karate, 0)
# for i in range(34):
#     print(f"shortest path to {i} node: {paths_from_zero_node[i]}")
# print("\n\n\n")

# print(list(nx.all_pairs_dijkstra(G_karate))[0])
# print(list(nx.all_shortest_paths(G_karate, 0, 33)))

# all_paths = dict()
# for start_node in range(0, 34):
#     distances, predcessors = dijkstra_all_paths(G_karate, start_node)
#     all_paths_from_node = [get_all_paths(predcessors, start_node, end_node) for end_node in range(34)]
#     all_paths[start_node] = all_paths_from_node

# print(all_paths[0][33])




# adj_matrix = nx.adjacency_matrix(G_karate).todense()
# max_lambda = max([1/i.real for i in get_numpy_eigenvalues(adj_matrix)])

# matrix = adj_matrix - max_lambda * np.eye(adj_matrix.shape[0])

# A_reduced = matrix[:-1, :-1]
# b_reduced = -matrix[:-1, -1]  # - (последний столбец без последней строки)
        
# v_part = get_scipy_solution(A_reduced, b_reduced)
# print(v_part)