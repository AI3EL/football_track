//
// Created by Paul Racanière on 07/12/2018.
//

#ifndef FOOTBALL_TRACK_CLUSTERING_H
#define FOOTBALL_TRACK_CLUSTERING_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Struct to hold informations about color clusters in an image
struct color_cluster {
    size_t K;
    vector<int> labels, sizes;
    vector<Vec3b> data, centroids;
    double compactness = -1.0;

    color_cluster(const vector<Vec3b> &data, size_t K) :
            K(K), data(data), centroids(K), labels(data.size()), sizes(K) {}

    color_cluster() : K(0) {}
};

void k_means_vec3b(color_cluster &cluster, size_t K, size_t rand_rep, double eps, size_t max_rep);

#endif //FOOTBALL_TRACK_CLUSTERING_H
