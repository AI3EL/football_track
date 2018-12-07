#include "clustering.h"
#include "graphCuts.h"

void assign_labels(const vector<Vec3b> &points, vector<int> &labels, vector<Vec3b> &centroids);

double compute_compactness(const vector<Vec3b> &points, vector<int> &labels, vector<Vec3b> &centroids);

void compute_centroids(const vector<Vec3b> &points, vector<int> &labels, vector<Vec3b> &centroids,
                       vector<int> &cluster_sizes);

void assign_labels(const vector<Vec3b> &points, vector<int> &labels, vector<Vec3b> &centroids) {
    for (int i = 0; i < points.size(); i++) {
        int new_label = 0;
        double min = sqdst(points[i], centroids[0]);
        for (int j = 1; j < centroids.size(); j++) {
            if (min > sqdst(points[i], centroids[j])) {
                min = sqdst(points[i], centroids[j]);
                new_label = j;
            }
        }
        labels[i] = new_label;
    }
}

double compute_compactness(const vector<Vec3b> &points, vector<int> &labels, vector<Vec3b> &centroids) {
    double res = 0.0;
    for (int i = 0; i < points.size(); i++) {
        res += sqdst(points[i], centroids[labels[i]]);
    }
    return res;
}

void compute_centroids(const vector<Vec3b> &points, vector<int> &labels, vector<Vec3b> &centroids,
                       vector<int> &cluster_sizes) {
    vector<vector<int> > new_centroids(centroids.size());
    for (int i = 0; i < centroids.size(); i++) {
        new_centroids[i].push_back(0);
        new_centroids[i].push_back(0);
        new_centroids[i].push_back(0);
    }

    vector<int> new_cluster_sizes(centroids.size(), 0);
    for (int i = 0; i < points.size(); i++) {
        new_cluster_sizes[labels[i]]++;
        new_centroids[labels[i]][0] += points[i].val[0];
        new_centroids[labels[i]][1] += points[i].val[1];
        new_centroids[labels[i]][2] += points[i].val[2];
    }

    for (int i = 0; i < centroids.size(); i++) {
        if (new_cluster_sizes[i]) {
            new_centroids[i][0] /= new_cluster_sizes[i];
            new_centroids[i][1] /= new_cluster_sizes[i];
            new_centroids[i][2] /= new_cluster_sizes[i];
        }
        cluster_sizes[i] = new_cluster_sizes[i];
    }

    for (int i = 0; i < centroids.size(); i++) {
        centroids[i] = Vec3b((uchar) new_centroids[i][0], (uchar) new_centroids[i][1], (uchar) new_centroids[i][2]);
    }
}

/*
 * K : number of clusters
 * rand_rep : number of time we repeat the algorithm with another random intialization
 * eps : when compactness is below we stop
 * max_rep : when rep is above we stop
 */
void k_means_vec3b(color_cluster &cluster, size_t K, size_t rand_rep, double eps, size_t max_rep) {
    if (!cluster.data.size()) {
        cout << " Empty dataset ! " << endl;
        cluster.compactness = 0.0;
        return;
    }
    color_cluster best_clusters(cluster.data, cluster.K), cur_clusters(cluster.data, K);
    for (int r = 0; r < rand_rep; r++) {
        for (int i = 0; i < K; ++i) {
            int blue = rand() % 256;
            int green = rand() % 256;
            int red = rand() % 256;
            cur_clusters.centroids[i] = Vec3b(uchar(blue), uchar(green), uchar(red));
        }
        assign_labels(cluster.data, cur_clusters.labels, cur_clusters.centroids);
        compute_centroids(cluster.data, cur_clusters.labels, cur_clusters.centroids, cur_clusters.sizes);
        cur_clusters.compactness = compute_compactness(cluster.data, cur_clusters.labels, cur_clusters.centroids);
        size_t rep = 1;
        while (cur_clusters.compactness > eps && rep < max_rep) {
            assign_labels(cluster.data, cur_clusters.labels, cur_clusters.centroids);
            compute_centroids(cluster.data, cur_clusters.labels, cur_clusters.centroids, cur_clusters.sizes);
            cur_clusters.compactness = compute_compactness(cluster.data, cur_clusters.labels, cur_clusters.centroids);
            rep++;
        }
        if (r) {
            if (cur_clusters.compactness < best_clusters.compactness) {
                best_clusters.compactness = cur_clusters.compactness;
                best_clusters.labels = cur_clusters.labels;
                best_clusters.centroids = cur_clusters.centroids;
                best_clusters.sizes = cur_clusters.sizes;
            }
        } else {
            best_clusters.compactness = cur_clusters.compactness;
            best_clusters.labels = cur_clusters.labels;
            best_clusters.centroids = cur_clusters.centroids;
            best_clusters.sizes = cur_clusters.sizes;
        }
    }
    cluster = best_clusters;
}

vector<Vec3b>
image_to_vect_select(const Image<Vec3b> &inpt, Vec3b color, float dst, vector<pair<int, int> > &vect_to_im) {
    vector<Vec3b> res;
    vect_to_im.resize(0);
    for (int x = 0; x < inpt.width(); x++) {
        for (int y = 0; y < inpt.height(); y++) {
            if (sqdst(inpt(x, y), color) > dst) {
                res.push_back(inpt(x, y));
                vect_to_im.emplace_back(x, y);
            }
        }
    }
    return res;
}