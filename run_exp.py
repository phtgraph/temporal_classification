import json
from tabulate import tabulate
from random_setup import random_exp_pure, random_exp_mixed
from contact_network_models import syn_single, syn_multi

def run_random_exp_pure(parameters):
    results = []
    for idx, param in enumerate(parameters):
        print(f"Running configuration {idx + 1} for random_exp_pure...")
        try:
            # Call the random_exp1 function with unpacked parameters
            accuracy = random_exp_pure(**param)
            results.append({**param, "accuracy": accuracy})
        except Exception as e:
            print(f"Error in configuration {idx + 1}: {e}")
            results.append({**param, "accuracy": "Error"})
    return results

def run_random_exp_mixed(parameters):
    results = []
    for idx, param in enumerate(parameters):
        print(f"Running configuration {idx + 1} for random_exp_mixed...")
        try:
            # Call the random_exp1 function with unpacked parameters
            accuracy = random_exp_mixed(**param)
            results.append({**param, "accuracy": accuracy})
        except Exception as e:
            print(f"Error in configuration {idx + 1}: {e}")
            results.append({**param, "accuracy": "Error"})
    return results

def run_contact_network_single(parameters):
    results = []
    for idx, param in enumerate(parameters):
        print(f"Running configuration {idx + 1} for contact_network_single...")
        try:
            # Call the random_exp1 function with unpacked parameters
            accuracy = syn_single(**param)
            results.append({**param, "accuracy": accuracy})
        except Exception as e:
            print(f"Error in configuration {idx + 1}: {e}")
            results.append({**param, "accuracy": "Error"})
    return results

def run_contact_network_multi(parameters):
    results = []
    for idx, param in enumerate(parameters):
        print(f"Running configuration {idx + 1} for contact_network_multi...")
        try:
            # Call the random_exp1 function with unpacked parameters
            accuracy = syn_multi(**param)
            results.append({**param, "accuracy": accuracy})
        except Exception as e:
            print(f"Error in configuration {idx + 1}: {e}")
            results.append({**param, "accuracy": "Error"})
    return results

def main():

    with open('config.json', 'r') as file:
        config = json.load(file)

    if "random_exp_pure" in config:
        print("\n--- Running random_exp_pure ---")
        parameters = config["random_exp_pure"]["parameters"]
        results = run_random_exp_pure(parameters)
        
        # Display results for random_exp1 in a table
        print("\nResults for random_exp_pure:")
        table = tabulate(results, headers="keys", tablefmt="grid")
        print(table)

    if "random_exp_mixed" in config:
        print("\n--- Running random_exp_mixed ---")
        parameters = config["random_exp_mixed"]["parameters"]
        results = run_random_exp_mixed(parameters)

        updated_results = []
        for result in results:
            n_cluster_1 = result.pop("n_cluster_1")
            n_cluster_2 = result.pop("n_cluster_2")
            updated_result = {"classes": f"{n_cluster_1}-{n_cluster_2}", **result}
            updated_results.append(updated_result)

        print("\nResults for random_exp_mixed:")
        table = tabulate(updated_results, headers='keys', tablefmt="grid")
        print(table)

    if "contact_network_single" in config:
        print("\n--- Running contact_network_single ---")
        parameters = config["contact_network_single"]["parameters"]
        results = run_contact_network_single(parameters)

        print("\nResults for contact_network_single:")
        table = tabulate(results, headers='keys', tablefmt="grid")
        print(table)

    if "contact_network_multi" in config:
        print("\n--- Running contact_network_multi ---")
        parameters = config["contact_network_multi"]["parameters"]
        results = run_contact_network_multi(parameters)

        print("\nResults for contact_network_multi:")
        table = tabulate(results, headers='keys', tablefmt="grid")
        print(table)


if __name__ == "__main__":
    main()
