import os
import sys

import networkx as nx
from utils.lesson2 import degree
from utils.priorityq import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt


def shapley_degree(G, C=None):
    """
    Shapley value for the characteristic function
        value(C) = |C| + |N(C)|,
    where N(C) is the set of nodes outside C with at least one neighbor in C

    :param G: Unweighted and undirected Networkx graph
    :param C: A coalition of players, it should be iterable
    :return: Shapley value for the characteristic function for all nodes of the coalition
    """

    if C is None:
        return 0

    deg = degree(G)

    # Shapley values
    sv = {}

    for v in C:
        sv[v] = 1 / (1 + deg[v])
        for u in G.neigbors(v):
            sv[v] += 1 / (1 + deg[u])

    return sv


def shapley_threshold(G, k, C=None):
    """
    Shapley value for the characteristic function
        value(C) = |C| + |N(C,k)|,
    where N(C,k) is the set of nodes outside C with at least k neighbors in C

    :param G: Unweighted and undirected Networkx graph
    :param C: A coalition of players, it should be iterable
    :param k: Threshold
    :return: Shapley value for the characteristic function for all nodes of the coalition
    """

    if C is None:
        return 0

    deg = degree(G)

    # Shapley values
    sv = {}

    for v in C:
        sv[v] = min(1, (k / (1 + deg[v])))
        for u in G.neighbors(v):
            sv[v] += max(0, ((deg[u] - k + 1) / (deg[u] * (1 + deg[u]))))

    return sv


def shapley_closeness(G, f):
    """
    :param G: Weighted networkx graph
    :param f: A function for the distance
    :return: Shapley value for the characteristic function for all nodes
    """
    # Initialise
    shapley = {}

    for v in G.nodes():
        shapley[v] = 0

    for v in G.nodes():
        distances, nodes = dijkstra(v, G)
        index = len(nodes) - 1
        sum = 0
        prevDistance = -1
        prevSV = -1

        while index > 0:
            if distances[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = (f_dist(distances[index]) / (1 + index)) - sum

            shapley[nodes[index]] += currSV
            sum += f(distances[index]) / (index * (1 + index))
            prevDistance = distances[index]
            prevSV = currSV
            index -= 1
        shapley[v] += f(0) - sum

    return shapley


def f_dist(dist):
    return 1 / (1 + dist)  # We add 1 to D to avoid infinite distance


def dijkstra(start, G: nx.Graph):
    open = PriorityQueue()
    dist = {start: 0}
    increasing_order_dist = PriorityQueue()

    for v in G.nodes():
        if not v == start:
            dist[v] = np.Inf
        increasing_order_dist.add(v, dist[v])
        open.add(v, dist[v])

    while not open.is_empty():
        u = open.pop()
        for v in G.neighbors(u):
            # extract current weight between u and the current neighboor v
            try:
                w = G[u][v]["weight"]
            except KeyError:
                w = 1  # For unweighted graph
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                increasing_order_dist.add(v, dist[v])
                # decrease priority of v
                open.add(v, alt)  # If an element already exists it update the priority
    return sorted_elements(dist, increasing_order_dist)


def sorted_elements(dist, pq: PriorityQueue):
    sorted_list = []
    distances = []
    while not pq.is_empty():
        k = pq.pop()
        sorted_list.append(k)
        distances.append(dist[k])
    return distances, sorted_list

def BFS_for_shapley_closeness(graph, u):
    level = 1
    n = graph.number_of_nodes()
    clevel = [u]
    visited = []
    visited.append(u)
    dist = {}

    while len(visited) < n:
        nlevel = []
        if len(clevel) == 0 and level == 100:
            sys.exit()
        while len(clevel) > 0:
            c = clevel.pop()
            for v in graph[c]:
                if v not in visited:
                    visited.append(v)
                    nlevel.append(v)
                    dist[v] = level
        level += 1
        clevel = nlevel

    return list(dist.keys()), list(dist.values())

def shapley_closeness_unweighted_graph(G, f):
    """
    :param G: Unweighted networkx graph
    :param f: A function for the distance
    :return: Shapley value for the characteristic function for all nodes
    """
    # Initialise
    shapley = {}

    for v in G.nodes():
        shapley[v] = 0

    for v in G.nodes():
        nodes, distances = BFS_for_shapley_closeness(G, v)
        index = len(nodes) - 1
        sum = 0
        prevDistance = -1
        prevSV = -1

        while index > 0:
            if distances[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = (f_dist(distances[index]) / (1 + index)) - sum

            shapley[nodes[index]] += currSV
            sum += f(distances[index]) / (index * (1 + index))
            prevDistance = distances[index]
            prevSV = currSV
            index -= 1
        shapley[v] += f(0) - sum

    return shapley
if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge('A', 'D')
    G.add_edge('D', 'B')
    G.add_edge('A', 'C')
    G.add_edge('A', 'v')
    G.add_edge('B', 'C')
    G.add_edge('D', 'E')
    G.add_edge('F', 'G')
    G.add_edge('G', 'H')
    G.add_edge('A', 'F')
    print(shapley_closeness(G, f_dist))
