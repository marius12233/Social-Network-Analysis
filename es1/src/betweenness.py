import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed
import time
import sys


#local import (running the python command in the src folder)
sys.path.append('../../')
from utils.priorityq import PriorityQueue


def betweenness(G):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in G.nodes()}
        eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in
                 G.nodes()}  # the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        # BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c = tree.pop()

            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i] / spnum[
                    c])  # the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c,
                                             i})]  # each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c,
                                                                i})]  # betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[
                    c]  # betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw, node_btw

def betwenness_clustering(G):
    # saving starting time of the algorithm
    start = time.time()
    #compute the betweenness for each edge
    eb, nb = betweenness(G)
    #insert the edge in a priorityq (sorted by highest betweenness)
    pq = PriorityQueue()
    for i in eb.keys():
        pq.add(i, -eb[i])
    graph = G.copy()
    # at each iteration we remove the highest betweenness edge
    # we can stop the algorithm when there are only 4 cluster (connected component in the graph)
    cc = []
    while len(cc) != 4:
        edge = tuple(sorted(pq.pop()))
        graph.remove_edges_from([edge])
        cc = list(nx.connected_components(graph))
    
    end = time.time()
    # algorithm execution time
    print("Execution time:", end-start)
    # we format the output into a dict
    label = ['first', 'second', 'third', 'fourth']
    final_cluster = {}
    for i in range(4):
        final_cluster[label[i]] = cc[i]
    return final_cluster

def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}

def betweenness_parallel(G, j=8):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}
    #we split the betweenness computation among different jobs
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(compute_btw)(G, X) for X in chunks(G.nodes(), math.ceil(len(G.nodes()) / j)))
    #now it is necessary to aggregate  the results
    for key in edge_btw.keys():
        for res in result:
            edge_btw[key] += res[0][key]
    for key in node_btw.keys():
        for res in result:
            node_btw[key] += res[1][key]
    return edge_btw, node_btw

def compute_btw(G, nodes):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}
    for s in nodes:
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in G.nodes()}
        eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in
                 G.nodes()}  # the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        # BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c = tree.pop()

            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i] / spnum[
                    c])  # the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c,
                                             i})]  # each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c,
                                                                i})]  # betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[
                    c]  # betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex
    return edge_btw, node_btw

def betwenness_clustering_parallel(G, j=1):
    # saving starting time of the algorithm
    start = time.time()
    #compute the betweenness for each edge
    eb, nb = betweenness_parallel(G,j)
    #insert the edge in a priorityq (sorted by highest betweenness)
    pq = PriorityQueue()
    for i in eb.keys():
        pq.add(i, -eb[i])
    graph = G.copy()
    # at each iteration we remove the highest betweenness edge
    # we can stop the algorithm when there are only 4 cluster (connected component in the graph)
    cc = []
    while len(cc) != 4:
        edge = tuple(sorted(pq.pop()))
        graph.remove_edges_from([edge])
        cc = list(nx.connected_components(graph))
    
    end = time.time()
    # algorithm execution time
    print("Execution time:", end-start)
    # we format the output into a dict
    label = ['first', 'second', 'third', 'fourth']
    final_cluster = {}
    for i in range(4):
        final_cluster[label[i]] = cc[i]
    return final_cluster
