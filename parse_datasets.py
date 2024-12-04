import csv
import networkx as nx

# Add your custom function to read a data file here
# def custom(filepath):
#   -------
#   return G

def highschool(filepath):
    data = []
    with open(filepath, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                data += lines
            
    cleaned_data = [line.split('\t') for line in data]
    U1 = []
    # Print the result
    for entry in cleaned_data:
        U1 += [entry[1:3] + entry[:1]]
        
    G1 = nx.Graph()
    for entry in U1:
        i, j, t = entry
        t = int(t)  # Convert time label to integer
        i = int(i)
        j = int(j)
        if G1.has_edge(i, j):
            # If the edge exists, update the time attribute to include the new time
            current_times = G1[i][j].get('time', [])
            if isinstance(current_times, list):
                if t not in current_times:
                    current_times.append(t)
            else:
                # Convert singleton to a list and append new time
                G1[i][j]['time'] = [current_times, t]
        else:
            # Create the edge with the time label as a list
            G1.add_edge(i, j, time=[t])

    return G1

def hospital(filepath):
    with open(filepath, 'r') as file:
        file_data = file.readlines()

    G = nx.Graph()

    # Parse the file data and add edges to the graph
    for line in file_data:
        parts = line.strip().split('\t')
        
        if len(parts) != 5:
            continue  # Skip malformed lines

        t, i, j, Si, Sj = parts  # Unpack the values
        t = int(t)   # Convert time to integer
        i = int(i)   # Convert node i to integer
        j = int(j)   # Convert node j to integer

        # If the edge already exists, append the time to the list of times
        if G.has_edge(i, j):
            G[i][j]['time'].append(t)
        else:
            # If the edge doesn't exist, create it with the initial time in a list
            G.add_edge(i, j, time=[t])

    return G

def mit(filepath):
    Gm = nx.Graph()
    with open(filepath, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.startswith('%') or not line.strip():
                continue

            # Parse the line, split on the first space and then on tab
            first_part, *rest = line.strip().split('\t')
            from_node, to_node = map(int, first_part.split())  # Split first part by space
            timestamp = int(rest[1])  # Extract timestamp (4th column)

            # Add edge to the graph, ensuring unique timestamps
            if Gm.has_edge(from_node, to_node):
                # Update timestamp list, ensuring uniqueness
                Gm[from_node][to_node]['time'] = list(set(Gm[from_node][to_node]['time'] + [timestamp]))
            else:
                # Create new edge with a timestamp list
                Gm.add_edge(from_node, to_node, time=[timestamp])

    return Gm

def workplacev2(filepath):
    Gw = nx.Graph() 

    with open(filepath, 'r') as file:
        for line in file:
            # Split the line into t, i, j
            t, i, j = map(int, line.strip().split(' '))
            
            # Add the edge and append the timestamp to its 'time' list
            if Gw.has_edge(i, j):
                # Append the new timestamp if it's not already in the list
                if t not in Gw[i][j]['time']:
                    Gw[i][j]['time'].append(t)
            else:
                # Create a new edge with the timestamp
                Gw.add_edge(i, j, time=[t])
    return Gw