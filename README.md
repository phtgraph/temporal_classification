# Classification of Temporal Graphs using Persistent Homology

## Preliminaries

1. Make sure you have the requirements in `requirements.txt` installed. To do this, just run `pip install -r requirements.txt` in your python environment.
2. For the C++ part, make sure you have Eigen library installed. Then run the following commands in the terminal :
   ```
   g++ -o kmp kernelization.cpp -I/path/to/eigen -pthread -O3
   g++ -O3 -shared -std=c++17 -fPIC -o avg_all_diff.so avg_all_diff.cpp
   ```

## Contact Network Models and Random Temporal Graphs

1. Add your graph classification parameters in the `config.json` file. Some sample parameters are added, with variable descriptions.
2. Run the script `run_exp.py`.

## Real Datasets

1. The `parse_dataset.py` contains functions to parse files for `highschool, hospital, mit, workplacev2`. You can add your own function to parse a new data from a file.
2. `run_exp_real.py` will perform the classification task of RE vs CM or EWLS vs CM. The usage for this function is as follows :
   ```
   python run_exp_real.py dataset filepath exp_type num_perturb edge_swap time_swap diag_filter

   positional arguments:
     dataset      Name of the dataset funtion in parse_datasets.
     filepath     Path to the dataset file.
     exp_type     Type of experiment : re or ewls.
     num_perturb  Number of re/ewls perturbations.
     edge_swap    Number of edge swaps for CM.
     time_swap    Number of time swaps for CM.
     diag_filter  Persistence threshold.
   ```
   For example,
   ```
   python run_exp_real.py hospital data/detailed_list_of_contacts_Hospital.dat_ re 25 50 50 0
   ```
   You can run `python run_real_exp.py -h` for help.
