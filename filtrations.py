import networkx as nx
import numpy as np
from random import sample
import ctypes

def min_weight_single(G):
    # G is a networkx undirected time labelled graph
    wtG = nx.Graph()
    for edge in G.edges():
        (u,v) = edge
        wt = np.inf
        for nbr in nx.neighbors(G,u):
            if nbr == v: continue
            if abs(G.edges[u,nbr]['time'] - G.edges[u,v]['time']) < wt:
                wt = abs(G.edges[u,nbr]['time'] - G.edges[u,v]['time'])
        for nbr in nx.neighbors(G,v):
            if nbr == u: continue
            if abs(G.edges[v,nbr]['time'] - G.edges[u,v]['time']) < wt:
                wt = abs(G.edges[v,nbr]['time'] - G.edges[u,v]['time'])
        wtG.add_edge(u,v,weight=wt)
    return wtG

def avg_weight_single(G):
    wtG = nx.Graph()
    for edge in G.edges():
        (u,v) = edge
        s = 0
        n = 0
        for nbr in nx.neighbors(G,u):
            if nbr == v: continue
            else:
                s += abs(G.edges[u,nbr]['time'] - G.edges[u,v]['time'])
                n += 1
        for nbr in nx.neighbors(G,v):
            if nbr == u: continue
            else:
                s += abs(G.edges[v,nbr]['time'] - G.edges[u,v]['time'])
                n += 1
        if n != 0:
            wtG.add_edge(u,v,weight=s/n)
    return wtG

def multi_to_single(G):
    Ge = G.copy()
    for (u,v) in Ge.edges():
        s = 0
        t = 0
        R = Ge[u][v]['time']
        for i in R:
           s += i
           t += 1
        if t != 0:
            Ge[u][v]['time'] = s/t
    return Ge

def avg_of_all_diff(G):
    wtG = nx.Graph()
    
    for edge in G.edges():
        (u, v) = edge
        s = 0
        n = 0
        for nbr in nx.neighbors(G, u):
            if nbr == v: 
                continue
            if G.has_edge(u, nbr):
                for i in G[u][nbr]['time']:
                    for j in G[u][v]['time']:
                        s += abs(i - j)
                        n += 1
        for nbr in nx.neighbors(G, v):
            if nbr == u: 
                continue
            if G.has_edge(v, nbr):
                for i in G[v][nbr]['time']:
                    for j in G[u][v]['time']:
                        s += abs(i - j)
                        n += 1
        if n != 0:
            wtG.add_edge(u, v, weight=s / n)
    return wtG

# Load the C++ shared library
lib = ctypes.CDLL('./avg_all_diff.so')

# Define the input data structure
class EdgeWeight(ctypes.Structure):
    _fields_ = [("u", ctypes.c_int), ("v", ctypes.c_int), ("weight", ctypes.c_double)]

lib.assign_weights_avg_of_all_diff.restype = ctypes.POINTER(EdgeWeight)
lib.assign_weights_avg_of_all_diff.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]

def avg_of_all_diff_cpp(G):
    nodes = list(G.nodes())
    node_map = {node: idx for idx, node in enumerate(nodes)}

    edges = []
    times = []
    time_counts = []

    for u, v, data in G.edges(data=True):
        edges.append(node_map[u])
        edges.append(node_map[v])
        times.extend(data['time'])
        time_counts.append(len(data['time']))

    num_nodes = len(nodes)
    num_edges = len(G.edges)
    edges_array = (ctypes.c_int * len(edges))(*edges)
    times_array = (ctypes.c_int * len(times))(*times)
    time_counts_array = (ctypes.c_int * len(time_counts))(*time_counts)

    result_size = ctypes.c_int(0)
    result = lib.assign_weights_avg_of_all_diff(
        num_nodes, num_edges,
        edges_array, times_array, time_counts_array,
        ctypes.byref(result_size)
    )

    wtG = nx.Graph()
    for i in range(result_size.value):
        edge = result[i]
        wtG.add_edge(nodes[edge.u], nodes[edge.v], weight=edge.weight)

    lib.free(result)  # Free allocated memory
    return wtG