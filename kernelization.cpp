#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <functional>
#include <numeric>
#include <Eigen/Dense>
#include <future>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::function;

double gaussianKernel(double x, double bandwidth) {
    return exp(-x * x / (2 * bandwidth * bandwidth)) / (sqrt(2 * M_PI) * bandwidth);
}

double persistenceWeightedGaussianKernel(const MatrixXd &D1, const MatrixXd &D2, 
                                         function<double(VectorXd)> weight = [](VectorXd x) { return 1.0; }, 
                                         double bandwidth = 1.0) {
    int n = D1.rows();
    int m = D2.rows();
    VectorXd ws1(n), ws2(m);
    for (int i = 0; i < n; ++i) ws1(i) = weight(D1.row(i));
    for (int i = 0; i < m; ++i) ws2(i) = weight(D2.row(i));

    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result += ws1(i) * ws2(j) * gaussianKernel((D1.row(i) - D2.row(j)).norm(), bandwidth);
        }
    }
    return result;
}

double persistenceScaleSpaceKernel(const MatrixXd &D1, const MatrixXd &D2, double bandwidth = 1.0) {
    int n = D1.rows();
    MatrixXd DD1(2 * n, 2);
    DD1 << D1, D1.rowwise().reverse();

    int m = D2.rows();
    MatrixXd DD2(2 * m, 2);
    DD2 << D2, D2.rowwise().reverse();

    auto weight_pss = [](VectorXd x) { return x(1) >= x(0) ? 1.0 : -1.0; };
    return 0.5 * persistenceWeightedGaussianKernel(DD1, DD2, weight_pss, bandwidth);
}

MatrixXd pairwisePersistenceDiagramKernels(const vector<MatrixXd> &X, 
                                         const string &kernel = "persistence_scale_space", 
                                         double bandwidth = 1.0) {
    int n = X.size();
    MatrixXd K(n, n);

    if (kernel == "persistence_weighted_gaussian") {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                K(i, j) = persistenceWeightedGaussianKernel(X[i], X[j], [](VectorXd x) { return 1.0; }, bandwidth);
            }
        }
    } else if (kernel == "persistence_scale_space") {
        // Split the computation into tasks
        vector<future<void>> futures;
        int num_threads = thread::hardware_concurrency();
        int chunk_size = (n + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            futures.push_back(async([t, chunk_size, n, &X, &K, bandwidth]() {
                for (int i = t * chunk_size; i < min((t + 1) * chunk_size, n); ++i) {
                    for (int j = 0; j < n; ++j) {
                        K(i, j) = persistenceScaleSpaceKernel(X[i], X[j], bandwidth);
                    }
                }
            }));
        }

        // Wait for all tasks to complete
        for (auto &f : futures) {
            f.get();
        }
    }
    return K;
}

vector<MatrixXd> readPersistentDiagrams(const string &filename) {
    vector<MatrixXd> diagrams;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file" << endl;
        return diagrams;
    }

    string line;
    while (getline(file, line)) {
        vector<VectorXd> diagram;
        do {
            if (line.empty()) break;
            stringstream ss(line);
            VectorXd point(2);
            ss >> point(0) >> point(1);
            diagram.push_back(point);
        } while (getline(file, line) && !line.empty());

        int n = diagram.size();
        MatrixXd mat(n, 2);
        for (int i = 0; i < n; ++i) {
            mat.row(i) = diagram[i];
        }
        diagrams.push_back(mat);
    }

    return diagrams;
}

int main() {

    string filename = "persistent_diagrams.txt";

    vector<MatrixXd> diagrams = readPersistentDiagrams(filename);

    MatrixXd K = pairwisePersistenceDiagramKernels(diagrams, "persistence_scale_space");

    // Save the kernel matrix to a file
    ofstream file("kernel_matrix.csv");
    if (file.is_open()) {
        file << K.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
        file.close();
    } else {
        cerr << "Unable to open file for writing" << endl;
    }

    return 0;
}
