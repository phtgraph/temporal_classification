import networkx as nx
import random
import numpy as np
from filtrations import avg_of_all_diff_cpp, avg_weight_single
import gudhi as gd
import numpy as np
import networkx as nx
import subprocess
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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

def process_task_ad(G,per):
    local_T = []
    for _ in range(100):
        local_T.append(adj_fillinf(avg_of_all_diff_cpp(perturb_re(G,per))))
    return local_T

def process_task_ad_single(G,per):
    local_T = []
    for _ in range(100):
        local_T.append(adj_fillinf(avg_weight_single(perturb_re(G,per))))
    return local_T

def conv(T,n):
    [i,j] = T
    r = random.sample([4,5,6,7], 1)[0]
    for k in range(0,2 * n,r):
        if k > i and k < j:
            T = T[:-1] + [k] + [T[-1]]
    return T

def assort_mixing(G,i,j,xi,d_max):
    deg_i = G.degree(i)
    deg_j = G.degree(j)
    
    return 1 - xi + xi * (deg_i * deg_j) / (d_max ** 2)

def disassort_mixing(G,i,j,xi,d_max):
    deg_i = G.degree(i)
    deg_j = G.degree(j)
    
    return 1 - xi + xi * ((deg_i - deg_j) ** 2) / (d_max ** 2)

def sample_vertices(G,xi,d_max,mixing_function):
    [v1,v2] = random.sample(list(G.nodes), 2)
    if random.random() < mixing_function(G,v1,v2,xi,d_max):
        return [v1,v2]
    else:
        return sample_vertices(G,xi,d_max,mixing_function)
    
def generate_multi(p,q,n_vertices,n_edges,xi,d_max,mixing_function):
    '''
    p : adding edge probability
    q : removing edge probability
    n_vertices : number of vertices
    n_edges : number of edges
    xi : constant in mixing function
    d_max : maximum degree of vertex
    mixing_function : assort_mixing or disassort_mixing
    '''
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices)) 
    for i in range(n_edges):
        [v1,v2] = sample_vertices(G,xi,d_max,mixing_function)
        if random.random() < p:
            G.add_edge(v1, v2, time=[i,2*n_edges])
    i = n_edges
    for (u,v) in G.edges():
        i += 1
        if random.random() < q:
            L = [G[u][v]['time'][0]] + [i]
            G[u][v]['time'] = L
    for [u,v] in G.edges():
        T = G[u][v]['time'] 
        G[u][v]['time'] = conv(T,n_edges)  
    return G

def generate_single(p,n_vertices,n_edges,xi,d_max,mixing_function):
    '''
    p : adding edge probability
    n_vertices : number of vertices
    n_edges : number of edges
    xi : constant in mixing function
    d_max : maximum degree of vertex
    mixing_function : assort_mixing or disassort_mixing
    '''
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices)) 
    for i in range(n_edges):
        [v1,v2] = sample_vertices(G,xi,d_max,mixing_function)
        if random.random() < p:
            G.add_edge(v1, v2, time=i)
 
    return G

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

    # y_pred = svm_classifier.predict(mat_test)

    accuracy = svm_classifier.score(mat_test, y_test)
    return(accuracy)

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

function_map = {
        "assort_mixing": assort_mixing,
        "dissasort_mixing": disassort_mixing}

def syn_single(v,e,g1_params,g2_params,g3_params,per,diag_filter,dim0wt,dim1wt,dim2wt):
    # g_params = (p,q,xi,d_max,mixing_function)
    G1 = generate_single(g1_params[0], v, e, g1_params[1], g1_params[2], function_map[g1_params[3]])
    G2 = generate_single(g2_params[0], v, e, g2_params[1], g2_params[2], function_map[g2_params[3]])
    G3 = generate_single(g3_params[0], v, e, g3_params[1], g3_params[2], function_map[g3_params[3]])
    T1 = []
    T2 = []
    T3 = []

    # def process_task(G):
    #     return process_task_ad_single(G,per)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_task_ad_single, G1, per) for _ in range(1)]
        for future in as_completed(futures):
            T1.extend(future.result())
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_task_ad_single, G2, per) for _ in range(1)]
        for future in as_completed(futures):
            T2.extend(future.result())
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_task_ad_single, G3, per) for _ in range(1)]
        for future in as_completed(futures):
            T3.extend(future.result())

    T = T1 + T2 + T3
    # print(T)
    label = []
    for i in range(100):
        label += [1]
    for i in range(100):
        label += [2]
    for i in range(100):
        label += [3]
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

def syn_multi(v,e,g1_params,g2_params,g3_params,per,diag_filter,dim0wt,dim1wt,dim2wt):
    # g_params = (p,q,xi,d_max,mixing_function)
    G1 = generate_multi(g1_params[0], g1_params[1], v, e, g1_params[2], g1_params[3], function_map[g1_params[4]])
    G2 = generate_multi(g2_params[0], g2_params[1], v, e, g2_params[2], g2_params[3], function_map[g2_params[4]])
    G3 = generate_multi(g3_params[0], g3_params[1], v, e, g3_params[2], g3_params[3], function_map[g3_params[4]])
    T1 = []
    T2 = []
    T3 = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_task_ad, G1, per) for _ in range(1)]
        for future in as_completed(futures):
            T1.extend(future.result())
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_task_ad, G2, per) for _ in range(1)]
        for future in as_completed(futures):
            T2.extend(future.result())
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_task_ad, G3, per) for _ in range(1)]
        for future in as_completed(futures):
            T3.extend(future.result())

    T = T1 + T2 + T3
    # print(T)
    label = []
    for i in range(100):
        label += [1]
    for i in range(100):
        label += [2]
    for i in range(100):
        label += [3]
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