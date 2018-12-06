#ifndef FOOTBALL_TRACK_MAIN_HPP
#define FOOTBALL_TRACK_MAIN_HPP

#include <opencv2/opencv.hpp>
#include "image.h"
#include <string>
#include <queue>
#include "graphCuts.h"

using namespace cv;
using namespace std;


struct color_cluster{
    size_t K;
    vector<int> labels, sizes;
    vector<Vec3b> data, centroids;
    double compactness = -1.0;

    color_cluster(const vector<Vec3b>& data, size_t K):
    K(K), data(data), centroids(K), labels(data.size()), sizes(K){}
    color_cluster() : K(0){}
};


void assign_labels(const vector<Vec3b>& points, vector<int>& labels, vector<Vec3b>& centroids);
double compute_compactness(const vector<Vec3b>& points, vector<int>& labels, vector<Vec3b>& centroids);
void compute_centroids(const vector<Vec3b>& points, vector<int>& labels, vector<Vec3b>& centroids, vector<int>& cluster_sizes);
void k_means_vec3b(color_cluster& cluster, size_t K, size_t rand_rep, double eps, size_t max_rep);



template<typename T>
vector<T> image_to_vect(const Image<T>& inpt){
    vector<T> res;
    for(int x=0; x<inpt.width(); x++){
        for(int y=0; y<inpt.height(); y++){
            res.push_back(inpt(x,y));
        }
    }
    return res;
}

vector<Vec3b> image_to_vect_select(const Image<Vec3b>& inpt, Vec3b color, float dst, vector<pair<int, int> >& vect_to_im);

// Shows the image in a miniature
template<typename T>
void imshow_quarter(string str, const Image<T>& src){
    Size display_size(900, 450);
    Image<T> dezoomed;
    resize(src, dezoomed, display_size);
    imshow(str, dezoomed);
}

template<typename T>
void imshow_half(string str, const Image<T>& src){
    Size display_size(1800, 450);
    Image<T> dezoomed;
    resize(src, dezoomed, display_size);
    imshow(str, dezoomed);
}


#endif //FOOTBALL_TRACK_MAIN_HPP
