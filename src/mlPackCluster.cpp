#include <plot.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/mean_shift/mean_shift.hpp>
#include <filesystem>
using namespace mlpack;

namespace fs = std::filesystem;
const std::vector<std::string> dataset_names{
    "dataset0.csv", "dataset1.csv",
    "dataset2.csv", "dataset3.csv",
    "dataset4.csv", "dataset5.csv"
};

const std::vector<std::string> colors{"black", "red", "blue", "green",
                                      "cyan", "yellow", "brown", "magenta"};
using DataType = double;
using Coords = std::vector<DataType>;
using PointCoords = std::pair<Coords, Coords>;
using Clusters = std::unordered_map<size_t, PointCoords>;

void PlotClusters(const Clusters& clusters, const std::string& name, const std::string& file_name){
    plotcpp::Plot plt;
    plt.SetTerminal("png");
    plt.SetOutput(file_name);
    plt.SetTitle(name);
    plt.SetXLabel("x");
    plt.SetYLabel("y");
    plt.SetAutoscale();
    plt.GnuplotCommand("set grid");

    auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
    for(auto& cluster:clusters){
        std::stringstream params;
        auto color_index = cluster.first % colors.size();
        params <<  "lc rgb '" << colors[color_index] << "' pt 7";
        plt.AddDrawing(draw_state, 
            plotcpp::Points(cluster.second.first.begin(), cluster.second.first.end(),
            cluster.second.second.begin(),
             params.str()));
    }
    plt.EndDraw2D(draw_state);
    plt.Flush();
}

void kmeanClustering(const arma::mat& inputs,
                   size_t num_clusters, const std::string& name){

    // Perform K-means clustering
    arma::Row<size_t> assignments;
    KMeans<> kmeans;
    kmeans.Cluster(inputs, num_clusters, assignments);

    Clusters plot_clusters;
    for (size_t i = 0; i != inputs.n_cols; ++i) {
        auto cluster_idx = assignments[i];
        plot_clusters[cluster_idx].first.push_back(inputs.at(0, i));
        plot_clusters[cluster_idx].second.push_back(inputs.at(1, i));
    }

  PlotClusters(plot_clusters, "K-Means", name + "-kmeans.png");
}

void DBScanClustering(const arma::mat& inputs, const std::string& name){

    // Perform DBSCAN clustering
    arma::Row<size_t> assignments;
    DBSCAN<> dbscan(0.1, 15);
    dbscan.Cluster(inputs, assignments);

    Clusters plot_clusters;
    for (size_t i = 0; i != inputs.n_cols; ++i) {
        auto cluster_idx = assignments[i];
        plot_clusters[cluster_idx].first.push_back(inputs.at(0, i));
        plot_clusters[cluster_idx].second.push_back(inputs.at(1, i));
    }

  PlotClusters(plot_clusters, "DBSCAN", name + "-dbscan.png");
}

void DMeanShiftClustering(const arma::mat& inputs,
                   const std::string& name){

    // Perform Mean Shift clustering
    arma::Row<size_t> assignments;
    arma::mat centroids;

    MeanShift<> meanshift;
    auto radius = meanshift.EstimateRadius(inputs);
    meanshift.Radius(radius);
    meanshift.Cluster(inputs, assignments, centroids);

    Clusters plot_clusters;
    for (size_t i = 0; i != inputs.n_cols; ++i) {
        auto cluster_idx = assignments[i];
        plot_clusters[cluster_idx].first.push_back(inputs.at(0, i));
        plot_clusters[cluster_idx].second.push_back(inputs.at(1, i));
    }

  PlotClusters(plot_clusters, "Mean Shift", name + "-meanshift.png");
}

void GMMClustering(const arma::mat& inputs, size_t num_clusters, const std::string& name){
    GMM gmm(num_clusters,2);

    KMeans<> kmeans;
    size_t max_iterations = 250;
    double tolerance = 1e-10;
    EMFit<KMeans<>, NoConstraint> em(max_iterations, tolerance, kmeans);
    gmm.Train(inputs, 3, false, em);
     
    arma::Row<size_t> assignments;
    gmm.Classify(inputs, assignments);

    Clusters plot_clusters;
    for (size_t i = 0; i != inputs.n_cols; ++i) {
        auto cluster_idx = assignments[i];
        plot_clusters[cluster_idx].first.push_back(inputs.at(0, i));
        plot_clusters[cluster_idx].second.push_back(inputs.at(1, i));
    }
    PlotClusters(plot_clusters, "GMM", name + "-gmm.png");
}


namespace fs = std::filesystem;
int main(int argc, char** argv){
    if(argc <=1){
        std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
        return 1;
    }

    fs::path base_dir = fs::path(argv[1]);
    if(!fs::exists(base_dir) || !fs::is_directory(base_dir)){
        std::cerr << "Invalid dataset path: " << base_dir << std::endl;
        return 1;
    }

    for(const auto& dataset_name:dataset_names){
        fs::path file_path = base_dir / dataset_name;
        if(!fs::exists(file_path)){
            std::cerr << "File not found: " << file_path << std::endl;
            continue;
        }
        arma::mat dataset;
        mlpack::data::DatasetInfo info;

        data::Load(file_path, dataset, info, true);
        arma::Row<size_t> labels;
        labels = arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
         // remove label row
        dataset.shed_row(dataset.n_rows - 1);
        // remove index row
        dataset.shed_row(0);

        auto num_samples = dataset.n_cols;
        auto num_features = dataset.n_rows;
        std::size_t num_clusters =
            std::set<double>(labels.begin(), labels.end()).size();
        if (num_clusters < 2)
          num_clusters = 3;

        std::cout << dataset_name << "\n"
                  << "Num samples: " << num_samples
                  << " num features: " << num_features
                  << " num clusters: " << num_clusters << std::endl;

        kmeanClustering(dataset, num_clusters, dataset_name);
        DBScanClustering(dataset, dataset_name);
        DMeanShiftClustering(dataset, dataset_name);

        GMMClustering(dataset, num_clusters, dataset_name);
    }

    return 0;
}