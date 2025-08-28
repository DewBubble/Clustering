#include <dlib/clustering.h>
#include <dlib/matrix.h>
#include <plot.h>
#include<filesystem>
#include<iostream>
#include<unordered_map>


using namespace dlib;
namespace fs = std::filesystem;
using SampleType = dlib::matrix<double,1,1>;
using Samples = std::vector<SampleType>;

const std::vector<std::string> data_names{"dataset0.csv", "dataset1.csv",
                                          "dataset2.csv", "dataset3.csv",
                                          "dataset4.csv", "dataset5.csv"};

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
    plt.SetXLabel("X-axis");
    plt.SetYLabel("Y-axis");
    plt.SetAutoscale();
    plt.GnuplotCommand("set grid");

    auto draw_state = plt.StartDraw2D<Coords::const_iterator>();
    for(auto& cluster:clusters){
        std::stringstream params;
        params << "lc rgb '" << colors[cluster.first] << "' pt 7";
        plotcpp::Points(
            cluster.second.first.begin(),
            cluster.second.first.end(),
            cluster.second.second.begin(),
            std::to_string(cluster.first) + " cls", params.str());
        
    }
    plt.EndDraw2D(draw_state);
    plt.Flush();
}

template<typename T>
void hierachicalClustering(const T& inputs, size_t num_clusters, const std::string& name){
    matrix<double> dist(inputs.nr(), inputs.nr());
    for(long r=0; r<dist.nr();++r){
        for(long c=0;c<dist.nc();++c){
            dist(r,c) = length(subm(inputs, r,0,1,2) - subm(inputs, c, 0,1,2));
        }
    }

    std::vector<unsigned long> clusters;
    bottom_up_cluster(dist, clusters, num_clusters);
    Clusters plot_clusters;
    for (long i = 0; i != inputs.nr(); i++) {
        auto cluster_idx = clusters[i];
        plot_clusters[cluster_idx].first.push_back(inputs(i, 0));
        plot_clusters[cluster_idx].second.push_back(inputs(i, 1));
    }

  PlotClusters(plot_clusters, "Agglomerative clustering", name + "-aggl.png");
}

template<typename T>
void graphClustering(const T& inputs,  const std::string& name) {
    std::vector<sample_pair> edges;
    for(long i=0;i<inputs.nr();i++){
        for(long j=0; j<inputs.nr();j++){
            edges.emplace_back(i, j, length(subm(inputs, i,0,1,2) - subm(inputs, j, 0,1,2)));
        }
    }

    std::vector<unsigned long> clusters;
    const auto num_clusters = chinese_whispers(edges, clusters);

    std::cout << "Num clusters detected: " << num_clusters << std::endl;
    Clusters plot_clusters;
    for (long i = 0; i != inputs.nr(); i++) {
        auto cluster_idx = clusters[i];
        plot_clusters[cluster_idx].first.push_back(inputs(i, 0));
        plot_clusters[cluster_idx].second.push_back(inputs(i, 1));
    }

    PlotClusters(plot_clusters, "Graph clustering", name + "-graph.png");
}

template <typename I>
void DoGraphNewmanClustering(const I& inputs, const std::string& name) {
  std::vector<sample_pair> edges;
  for (long i = 0; i < inputs.nr(); ++i) {
    for (long j = 0; j < inputs.nr(); ++j) {
      auto dist = length(subm(inputs, i, 0, 1, 2) - subm(inputs, j, 0, 1, 2));
      if (dist < 0.5)
        edges.push_back(sample_pair(i, j, dist));
    }
  }
  remove_duplicate_edges(edges);

  std::vector<unsigned long> clusters;
  const auto num_clusters = newman_cluster(edges, clusters);
  std::cout << "Num clusters detected: " << num_clusters << std::endl;
  Clusters plot_clusters;
  for (long i = 0; i != inputs.nr(); i++) {
    auto cluser_idx = clusters[i];
    plot_clusters[cluser_idx].first.push_back(inputs(i, 0));
    plot_clusters[cluser_idx].second.push_back(inputs(i, 1));
  }

  PlotClusters(plot_clusters, "Graph Newman clustering",
               name + "-graph-newman.png");
}

template <typename T>
void kmeanClustering(const T& inputs, size_t num_clusters, const std::string& name) {
    // K-means clustering implementation
    typedef matrix<double, 2,1> sample_type;
    typedef radial_basis_kernel<sample_type> kernel_type;
    kcentroid<kernel_type> kc(kernel_type(0.1),0.01,8);
    kkmeans<kernel_type> kmeans(kc);

    std::vector<sample_type> samples;
    samples.reserve(inputs.nr());
    for(long i=0;i<inputs.nr(); ++i){
        samples.push_back(dlib::trans(dlib::subm(inputs, i,0,1,2)));
    }

    std:vector<sample_type> initial_centers;
    pick_initial_centers(num_clusters, initial_centers, samples, kmeans.get_kernel());

    kmeans.set_number_of_centers(num_clusters);
    kmeans.train(samples, initial_centers);
    
    std::vector<unsigned long> clusters;
    Clusters plot_clusters;
    for (long i = 0; i != inputs.nr(); i++) {
        auto cluster_idx = kmeans(samples[i]);
        plot_clusters[cluster_idx].first.push_back(inputs(i, 0));
        plot_clusters[cluster_idx].second.push_back(inputs(i, 1));
    }

    PlotClusters(plot_clusters, "K-Means", name + "-kmeans.png");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_file>" << std::endl;
        return 1;
    }

    auto base_dir = fs::path(argv[1]);
    for(auto& dataset : data_names){
        auto dataset_path = base_dir / dataset;
        // Load your dataset and call the clustering functio
        if(fs::exists(dataset_path)){
            std::ifstream file(dataset_path);
            matrix<DataType> data;
            file>>data;

            auto inputs = dlib::subm(data, 0,1,data.nr(),2);
            auto labels = dlib::subm(data, 0,3,data.nr(),1);

            auto num_samples = inputs.nr();
            auto num_features = inputs.nc();

            std::size_t num_clusters = std::set(labels.begin(), labels.end()).size();

            if (num_clusters < 2)
                num_clusters = 3;

            std::cout << dataset << "\n"
                << "Num samples: " << num_samples
                << " num features: " << num_features
                << " num clusters: " << num_clusters << std::endl;
            hierachicalClustering(inputs, num_clusters, dataset);
        }
    }

    return 0;
}