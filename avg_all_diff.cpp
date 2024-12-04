#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <cmath>
#include <iostream>

extern "C" {
    struct EdgeWeight {
        int u, v;
        double weight;
    };

    EdgeWeight* assign_weights_avg_of_all_diff(
        int num_nodes,
        int num_edges,
        int* edges,
        int* times,
        int* time_counts,
        int& result_size
    ) {

        std::vector<std::unordered_map<int, std::vector<int>>> graph(num_nodes);

        int edge_idx = 0, time_idx = 0;
        for (int i = 0; i < num_edges; ++i) {
            int u = edges[edge_idx++];
            int v = edges[edge_idx++];
            int count = time_counts[i];
            std::vector<int> time_list(count);
            for (int j = 0; j < count; ++j) {
                time_list[j] = times[time_idx++];
            }
            graph[u][v] = time_list;
            graph[v][u] = time_list; 
        }

        std::vector<EdgeWeight> edge_weights;

        for (int u = 0; u < num_nodes; ++u) {
            for (auto& [v, times_uv] : graph[u]) {
                if (u >= v) continue; 

                double total_diff = 0;
                int total_count = 0;

                for (auto& [nbr, times_un] : graph[u]) {
                    if (nbr == v) continue;
                    for (int t1 : times_un) {
                        for (int t2 : times_uv) {
                            total_diff += std::abs(t1 - t2);
                            total_count++;
                        }
                    }
                }

                for (auto& [nbr, times_vn] : graph[v]) {
                    if (nbr == u) continue;
                    for (int t1 : times_vn) {
                        for (int t2 : times_uv) {
                            total_diff += std::abs(t1 - t2);
                            total_count++;
                        }
                    }
                }

                if (total_count > 0) {
                    edge_weights.push_back({u, v, total_diff / total_count});
                }
            }
        }

        result_size = edge_weights.size();
        EdgeWeight* result = new EdgeWeight[result_size];
        std::copy(edge_weights.begin(), edge_weights.end(), result);
        return result;
    }
}
