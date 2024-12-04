import networkx as nx
import random
import numpy as np
from filtrations import avg_of_all_diff_cpp
import gudhi as gd
import subprocess
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import parse_datasets
import argparse
from tabulate import tabulate

def adj_fillinf(Gw):
    Aw = nx.adjacency_matrix(Gw,weight = 'weight')
    Ad = Aw.todense().astype(np.float64)
    Ad[Ad == 0] = np.inf
    np.fill_diagonal(Ad,0)
    return Ad

def RE(G):
    while True:
        [e1, e2] = random.sample(list(G.edges), 2)
        (u1, v1) = e1
        (u2, v2) = e2
        
        # Ensure no self-loops in the selected edges
        if u1 != v1 and u2 != v2:
            break  # Proceed if both edges are valid (no self-loops)

    c = random.choice([0,1])
    t1 = G[u1][v1]['time']
    t2 = G[u2][v2]['time']
    if c == 0:
        G.add_edge(u1,v2, time = t1)
        G.add_edge(u2,v1, time = t2)
    else:
        G.add_edge(u1,u2, time = t1)
        G.add_edge(v1,v2, time = t2)
    G.remove_edge(u1,v1)
    G.remove_edge(u2,v2)
    return G

def perturb_re(G,per):
    Ge = G.copy()
    for i in range(per):
        Ge = RE(Ge)
    return(Ge)

def process_task_ad_re(G,per):
    local_T = []
    for _ in range(100):
        local_T.append(adj_fillinf(avg_of_all_diff_cpp(perturb_re(G,per))))
    return local_T

def EWLS(G):
    # Get a list of all edges and their timestamp lengths
    edges = list(G.edges)
    edge_times = {e: len(G[e[0]][e[1]]['time']) for e in edges}

    # Group edges by the number of timestamps
    edges_by_length = {}
    for edge, length in edge_times.items():
        edges_by_length.setdefault(length, []).append(edge)

    # Filter groups with more than one edge
    valid_groups = {length: edges for length, edges in edges_by_length.items() if len(edges) > 1}

    # If no valid groups exist, return the graph as is
    if not valid_groups:
        print("No edges with the same number of timestamps.")
        return G

    # Randomly pick a group of edges with the same timestamp length
    t1 = random.choice(list(valid_groups.keys()))
    group = valid_groups[t1]

    # Select two different edges from this group
    e1, e2 = random.sample(group, 2)
    (u1, v1) = e1
    (u2, v2) = e2

    # Swap timestamps
    G[u1][v1]['time'], G[u2][v2]['time'] = G[u2][v2]['time'], G[u1][v1]['time']
    return G

def perturb_ewls(G,perd):
    Ge = G.copy()
    for i in range(perd):
        Ge = EWLS(Ge)
    return(Ge)

def process_task_ad_ewls(G,perd):
    local_T = []
    for _ in range(100):
        local_T.append(adj_fillinf(avg_of_all_diff_cpp(perturb_ewls(G,perd))))
    return local_T

def degree_preserving_edge_swap(graph, num_swaps):
    edges = list(graph.edges())
    nodes = list(graph.nodes())
    for _ in range(num_swaps):  
        (u, v), (x, y) = random.sample(edges, 2)
        if u != y and x != v and not graph.has_edge(u, y) and not graph.has_edge(x, v):
            graph.remove_edge(u, v)
            graph.remove_edge(x, y)
            graph.add_edge(u, y)
            graph.add_edge(x, v)
            edges.remove((u, v))
            edges.remove((x, y))
            edges.append((u, y))
            edges.append((x, v))

    return graph

def random_switch(list1, list2):
    # Ensure both lists are non-empty
    if len(list1) == 0 or len(list2) == 0:
        print("One of the lists is empty, no switch performed.")
        return list1, list2

    # Randomly pick indices from both lists
    idx1 = random.randint(0, len(list1) - 1)
    idx2 = random.randint(0, len(list2) - 1)

    # Swap the elements between the two lists
    list1[idx1], list2[idx2] = list2[idx2], list1[idx1]

    return list1, list2

def perturb_cm(G,es,ts):
    G1 = degree_preserving_edge_swap(G.copy(), es)    
    for u, v in G1.edges():
        G1[u][v].clear()
    E = nx.get_edge_attributes(G, 'time')
    E = list(E.items())
    i = 0
    for (u,v) in G1.edges():
        G1[u][v]['time'] = E[i][1]
        i += 1
    for i in range(ts):
        (e1,e2) =  random.sample(list(G1.edges()), 2)
        (u1,v1) = e1
        (u2,v2) = e2
        G1[u1][v1]['time'],G1[u2][v2]['time'] = random_switch(G1[u1][v1]['time'],G1[u2][v2]['time'])
        G1[u1][v1]['time'].sort()
        G1[u2][v2]['time'].sort()
    return G1

def process_task_ad_cm(G,es,ts):
    local_T = []
    for _ in range(100):
        local_T.append(adj_fillinf(avg_of_all_diff_cpp(perturb_cm(G,es,ts))))
    return local_T

def process_matrix_dim2(Ad):
    skeleton = gd.RipsComplex(distance_matrix=Ad, max_edge_length=5000000000) # specify max edge length here
    simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    barcode = simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(2)
    intervals[intervals == np.inf] = 10000000000 # ideally, 2 * max edge length
    
    return intervals

def process_matrix_dim0(Ad):
    skeleton = gd.RipsComplex(distance_matrix=Ad, max_edge_length=5000000000)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    barcode = simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(0)
    intervals[intervals == np.inf] = 10000000000
    
    return intervals

def process_matrix_dim1(Ad):
    skeleton = gd.RipsComplex(distance_matrix=Ad, max_edge_length=5000000000)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    barcode = simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(1)
    intervals[intervals == np.inf] = 10000000000
    
    return intervals

def filter_diagrams(diagrams, epsilon):
    """
    Filters points from persistence diagrams that are within epsilon distance from the diagonal.

    Parameters:
    - diagrams: list of numpy arrays, where each array is a persistence diagram
                (shape: n_points x 2, columns: [birth, death])
    - epsilon: float, the minimum distance a point must have from the diagonal

    Returns:
    - filtered_diagrams: list of filtered persistence diagrams
    """
    filtered_diagrams = []
    for diagram in diagrams:
        # Compute distances from the diagonal |death - birth|
        distances = np.abs(diagram[:, 1] - diagram[:, 0])
        # Filter points with distance greater than epsilon
        filtered_diagram = diagram[distances > epsilon]
        filtered_diagrams.append(filtered_diagram)
    return filtered_diagrams

def compute_kernel_matrix():
    result = subprocess.run(['./kmp'], check=True)
    if result.returncode != 0:
        raise Exception("C++ program failed to run.")
    kernel_matrix = np.loadtxt('kernel_matrix.csv', delimiter=',')
    return kernel_matrix

def svm_output(mat_train,mat_test,y_train,y_test):
    # Using SVM

    svm_classifier = SVC(kernel='precomputed')
    svm_classifier.fit(mat_train, y_train)

    accuracy = svm_classifier.score(mat_test, y_test)
    return(accuracy)

def run_experiment(dataset,filepath,exp_type,num_perturb,edge_swap,time_swap,diag_filter):
    f = getattr(parse_datasets, dataset, None)
    G = f(filepath)
    T1 = []
    T2 = []

    if exp_type == "re":

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_task_ad_re, G, num_perturb) for _ in range(1)]
            for future in as_completed(futures):
                T1.extend(future.result())
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_task_ad_cm, G, edge_swap, time_swap) for _ in range(1)]
            for future in as_completed(futures):
                T2.extend(future.result())

    if exp_type == "ewls":

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_task_ad_ewls, G, num_perturb) for _ in range(1)]
            for future in as_completed(futures):
                T1.extend(future.result())
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_task_ad_cm, G, edge_swap, time_swap) for _ in range(1)]
            for future in as_completed(futures):
                T2.extend(future.result())

    T = T1 + T2
    label = []
    for i in range(100):
        label += [1]
    for i in range(100):
        label += [2]
    input_diag_trial_dim0 = []
    input_diag_trial_dim1 = []
    input_diag_trial_dim2 = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim0, T)
        input_diag_trial_dim0 = list(results)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim1, T)
        input_diag_trial_dim1 = list(results)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim2, T)
        input_diag_trial_dim2 = list(results)
    X_train_dim0, X_test_dim0, y_train, y_test = train_test_split(input_diag_trial_dim0, label, test_size=0.2, random_state=42)

    X_combined_dim0 = (X_train_dim0 + X_test_dim0)
    X_combined_dim0 = filter_diagrams(X_combined_dim0,diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim0:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n')
                
    pssk_matrix_combined_dim0 = compute_kernel_matrix()

    X_train_dim1, X_test_dim1, y_train, y_test = train_test_split(input_diag_trial_dim1, label, test_size=0.2, random_state=42)

    X_combined_dim1 = (X_train_dim1 + X_test_dim1)
    X_combined_dim1 = filter_diagrams(X_combined_dim1,diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim1:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n')
                
    pssk_matrix_combined_dim1 = compute_kernel_matrix()

    X_train_dim2, X_test_dim2, y_train, y_test = train_test_split(input_diag_trial_dim2, label, test_size=0.2, random_state=42)

    X_combined_dim2 = (X_train_dim2 + X_test_dim2)
    X_combined_dim2 = filter_diagrams(X_combined_dim2,diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim2:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n') 
        
    pssk_matrix_combined_dim2 = compute_kernel_matrix()
    pssk_matrix_combined = np.add(pssk_matrix_combined_dim0, pssk_matrix_combined_dim1, pssk_matrix_combined_dim2)
    
    num_train = len(X_train_dim1)
    pssk_matrix_train = pssk_matrix_combined[:num_train, :num_train]
    pssk_matrix_test = pssk_matrix_combined[num_train:, :num_train]
        
    return(svm_output(pssk_matrix_train,pssk_matrix_test,y_train,y_test))

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Run RE vs CM or EWLS vs CM on real datasets.")
    parser.add_argument("dataset", help="Name of the dataset funtion in parse_datasets.")
    parser.add_argument("filepath", help="Path to the dataset file.")
    parser.add_argument("exp_type", help="Type of experiment : re or ewls.")
    parser.add_argument("num_perturb", type=int, help="Number of re/ewls perturbations.")
    parser.add_argument("edge_swap", type=int, help="Number of edge swaps for CM.")
    parser.add_argument("time_swap", type=int, help="Number of time swaps for CM.")
    parser.add_argument("diag_filter", type=float, help="Persistence threshold.")

    # Parse the arguments
    args = parser.parse_args()

    # Call run_experiment
    accuracy = run_experiment(
        args.dataset, args.filepath, args.exp_type,
        args.num_perturb, args.edge_swap, args.time_swap, args.diag_filter
    )

    # Prepare the table data
    table_data = [[
        args.dataset, f"{args.exp_type.upper()} vs CM", args.num_perturb,
        f"({args.edge_swap}, {args.time_swap})", args.diag_filter, accuracy
    ]]

    # Define the headers
    headers = ["Dataset", "Experiment Type", "Num Perturbations", "CM Parameters", "Diag Filter", "Accuracy"]

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()