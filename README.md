# Classification of Temporal Graphs using Persistent Homology

## Preliminaries

1. Make sure you have the requirements in `requirements.txt` installed. To do this, just run `pip install -r /path/to/requirements.txt` in your python environment.
2. For the C++ part, make sure you have Eigen library installed. Then run the following commands in the terminal :
   ```
   g++ -o kmp kernelization.cpp -I/path/to/eigen -pthread -O3
   g++ -O3 -shared -std=c++17 -fPIC -o avg_all_diff.so avg_all_diff.cpp
   ```

## Contact Network Models and Random Temporal Graphs

Add your graph classification parameters in the `config.json` file. Some sample parameters are added, with variable descriptions.
