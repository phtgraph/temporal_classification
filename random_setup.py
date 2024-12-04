from random import sample
import random
import gudhi as gd
import numpy as np
import networkx as nx
import subprocess
import concurrent.futures
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from filtrations import avg_weight_single

def adj_fillinf(Gw):
    Aw = nx.adjacency_matrix(Gw,weight = 'weight')
    Ad = Aw.todense().astype(np.float64)
    Ad[Ad == 0] = np.inf
    np.fill_diagonal(Ad,0)
    return Ad

def change_graph(G,p):
    n_edges = G.number_of_edges()
    num_ch_edges = int(p * n_edges)
    Ge = G.copy()
    random_edge = sample(list(G.edges()),num_ch_edges)
    random_ptb = sample(list(range(1,5)),1)
    random_sign = sample([0,1],1)
    for edge in random_edge:
        if random_sign[0] == 0:
            Ge.edges[edge[0],edge[1]]['time'] = Ge.edges[edge[0],edge[1]]['time'] - random_ptb[0]
        elif random_sign[0] == 1:
            Ge.edges[edge[0],edge[1]]['time'] = Ge.edges[edge[0],edge[1]]['time'] + random_ptb[0]
    return Ge

def svm_output(mat_train,mat_test,y_train,y_test):

    svm_classifier = SVC(kernel='precomputed')
    svm_classifier.fit(mat_train, y_train)

    accuracy = svm_classifier.score(mat_test, y_test)
    print("Accuracy using SVM : ", accuracy)
    return accuracy

def compute_kernel_matrix():
    # Call the C++ program
    result = subprocess.run(['./kmp'], check=True)
    
    if result.returncode != 0:
        raise Exception("C++ program failed to run.")
    
    # Load the kernel matrix from the CSV file
    kernel_matrix = np.loadtxt('kernel_matrix.csv', delimiter=',')
    return kernel_matrix

def process_matrix_dim2(Ad):
    skeleton = gd.RipsComplex(distance_matrix=Ad, max_edge_length=5000)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    barcode = simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(2)
    intervals[intervals == np.inf] = 10000
    
    return intervals

def process_matrix_dim0(Ad):
    skeleton = gd.RipsComplex(distance_matrix=Ad, max_edge_length=5000)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    barcode = simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(0)
    intervals[intervals == np.inf] = 10000
    
    return intervals

def process_matrix_dim1(Ad):
    skeleton = gd.RipsComplex(distance_matrix=Ad, max_edge_length=5000)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=3)
    barcode = simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(1)
    intervals[intervals == np.inf] = 10000
    
    return intervals

def process_task(G):
    local_T = []
    for _ in range(100):
        local_T.append(adj_fillinf(avg_weight_single(change_graph(G))))
    return local_T

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

def random_exp_pure(v,e,cp,wp,n_cluster,n_copies,diag_filter,dim0wt,dim1wt,dim2wt):

    root_graph = nx.gnm_random_graph(v,e)
    for (u, v) in root_graph.edges():
        root_graph.edges[u, v]['time'] = random.randint(0, 100)

    class_graphs = [change_graph(root_graph,cp) for _ in range(n_cluster)]
    for G in class_graphs:
        for (u, v) in G.edges():
            G.edges[u, v]['time'] = random.randint(0, 100)

    # Apply change_graph function in parallel
    all_graphs = Parallel(n_jobs=-1)(delayed(change_graph)(grph, wp) for grph in class_graphs for _ in range(n_copies))

    # Generate labels
    label = [i for i in range(n_cluster) for _ in range(n_copies)]

    # Apply assign_weights function in parallel
    Lw = Parallel(n_jobs=-1)(delayed(avg_weight_single)(i) for i in all_graphs)

    # Apply adj_fillinf function in parallel
    Lwe = Parallel(n_jobs=-1)(delayed(adj_fillinf)(i) for i in Lw)

    # Generate labels
    label = [i for i in range(n_cluster) for _ in range(n_copies)]

    input_diag_trial_dim0 = []
    input_diag_trial_dim1 = []
    input_diag_trial_dim2 = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim0, Lwe)
        input_diag_trial_dim0 = list(results)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim1, Lwe)
        input_diag_trial_dim1 = list(results)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim2, Lwe)
        input_diag_trial_dim2 = list(results)

    X_train_dim0, X_test_dim0, y_train, y_test = train_test_split(input_diag_trial_dim0, label, test_size=0.2, random_state=42)

    X_combined_dim0 = (X_train_dim0 + X_test_dim0)

    X_combined_dim0 = filter_diagrams(X_combined_dim0, diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim0:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n')
                
    pssk_matrix_combined_dim0 = compute_kernel_matrix()

    X_train_dim1, X_test_dim1, y_train, y_test = train_test_split(input_diag_trial_dim1, label, test_size=0.2, random_state=42)

    X_combined_dim1 = (X_train_dim1 + X_test_dim1)

    X_combined_dim1 = filter_diagrams(X_combined_dim1, diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim1:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n')
                
    pssk_matrix_combined_dim1 = compute_kernel_matrix()

    X_train_dim2, X_test_dim2, y_train, y_test = train_test_split(input_diag_trial_dim2, label, test_size=0.2, random_state=42)

    X_combined_dim2 = (X_train_dim2 + X_test_dim2)

    X_combined_dim2 = filter_diagrams(X_combined_dim2, diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim2:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n') 
        
    pssk_matrix_combined_dim2 = compute_kernel_matrix()

    pssk_matrix_combined = np.add(dim0wt*pssk_matrix_combined_dim0, dim1wt*pssk_matrix_combined_dim1, dim2wt*pssk_matrix_combined_dim2)
    
    num_train = len(X_train_dim1)
    pssk_matrix_train = pssk_matrix_combined[:num_train, :num_train]
    pssk_matrix_test = pssk_matrix_combined[num_train:, :num_train]
        
    return(svm_output(pssk_matrix_train,pssk_matrix_test,y_train,y_test))

def random_exp_mixed(v,e,cp,wp,n_cluster_1,n_cluster_2,n_copies,diag_filter,dim0wt,dim1wt,dim2wt):

    root_graph_1 = nx.gnm_random_graph(v,e)
    for (u, v) in root_graph_1.edges():
        root_graph_1.edges[u, v]['time'] = random.randint(0, 100)
    root_graph_2 = nx.gnm_random_graph(v,e)
    for (u, v) in root_graph_2.edges():
        root_graph_2.edges[u, v]['time'] = random.randint(0, 100)

    class_graphs_1 = [change_graph(root_graph_1,cp) for _ in range(n_cluster_1)]
    for G in class_graphs_1:
        for (u, v) in G.edges():
            G.edges[u, v]['time'] = random.randint(0, 100)

    class_graphs_2 = [change_graph(root_graph_2,cp) for _ in range(n_cluster_2)]
    for G in class_graphs_2:
        for (u, v) in G.edges():
            G.edges[u, v]['time'] = random.randint(0, 100)

    class_graphs = class_graphs_1 + class_graphs_2

    # Apply change_graph function in parallel
    all_graphs = Parallel(n_jobs=-1)(delayed(change_graph)(grph, wp) for grph in class_graphs for _ in range(n_copies))

    # Generate labels
    label = [i for i in range(n_cluster_1 + n_cluster_2) for _ in range(n_copies)]

    # Apply assign_weights function in parallel
    Lw = Parallel(n_jobs=-1)(delayed(avg_weight_single)(i) for i in all_graphs)

    # Apply adj_fillinf function in parallel
    Lwe = Parallel(n_jobs=-1)(delayed(adj_fillinf)(i) for i in Lw)

    # Generate labels
    label = [i for i in range(n_cluster_1 + n_cluster_2) for _ in range(n_copies)]

    input_diag_trial_dim0 = []
    input_diag_trial_dim1 = []
    input_diag_trial_dim2 = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim0, Lwe)
        input_diag_trial_dim0 = list(results)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim1, Lwe)
        input_diag_trial_dim1 = list(results)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_matrix_dim2, Lwe)
        input_diag_trial_dim2 = list(results)

    X_train_dim0, X_test_dim0, y_train, y_test = train_test_split(input_diag_trial_dim0, label, test_size=0.2, random_state=42)

    X_combined_dim0 = (X_train_dim0 + X_test_dim0)

    X_combined_dim0 = filter_diagrams(X_combined_dim0, diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim0:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n')
                
    pssk_matrix_combined_dim0 = compute_kernel_matrix()

    X_train_dim1, X_test_dim1, y_train, y_test = train_test_split(input_diag_trial_dim1, label, test_size=0.2, random_state=42)

    X_combined_dim1 = (X_train_dim1 + X_test_dim1)

    X_combined_dim1 = filter_diagrams(X_combined_dim1, diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim1:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n')
                
    pssk_matrix_combined_dim1 = compute_kernel_matrix()

    X_train_dim2, X_test_dim2, y_train, y_test = train_test_split(input_diag_trial_dim2, label, test_size=0.2, random_state=42)

    X_combined_dim2 = (X_train_dim2 + X_test_dim2)

    X_combined_dim2 = filter_diagrams(X_combined_dim2, diag_filter)
        
    with open('persistent_diagrams.txt', 'w') as f:
        for diagram in X_combined_dim2:
            np.savetxt(f, diagram, fmt='%.2f')
            f.write('\n') 
        
    pssk_matrix_combined_dim2 = compute_kernel_matrix()

    pssk_matrix_combined = np.add(dim0wt*pssk_matrix_combined_dim0, dim1wt*pssk_matrix_combined_dim1, dim2wt*pssk_matrix_combined_dim2)
    
    num_train = len(X_train_dim1)
    pssk_matrix_train = pssk_matrix_combined[:num_train, :num_train]
    pssk_matrix_test = pssk_matrix_combined[num_train:, :num_train]
        
    return(svm_output(pssk_matrix_train,pssk_matrix_test,y_train,y_test))