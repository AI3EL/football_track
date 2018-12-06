#include "main.hpp"

int main(int argn, char **argc) {

    size_t click_num = 0;
    const size_t FPC = 5;  // fpc = frame per click
    const size_t MIN_AREA = 1500;
    const int BLUR_FACTOR = 5;  // used to have faster graph cut
    const int EROSION_TYPE = MORPH_ELLIPSE;
    const size_t EROSION_SIZE = 4;
    const size_t DILATE_SIZE = EROSION_SIZE + 3;

    VideoCapture cap("../data/footdata2.mp4"); // Ouvre la vidéo
    if (!cap.isOpened()) {  // Vérifie l'ouverture
        cerr << "ERREUR : Lecture de vidéo impossible" << endl;
        return -1;
    }

    Image<Vec3b> I;
    cap >> I;

    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(100);
    Image<uchar> move_mask, move_mask_only_field(I.width(), I.height()), eroded, eroded_dilated, final_I(I.width(), I.height());

    vector<Vec3b> freq_colors;  // Vector holding the most frequent colors of each potential player detected for each image
    color_cluster team_colors;  // Color cluster using freq_colors as data, the centroids are the estimated colors of shirts of each team

    while(I.width() > 0 && I.height() > 0) {

        cout << "Click number : " << click_num++ << endl;

        for(int i=0; i<FPC; i++) cap >> I;

        // Loss of details to have faster graph cut
        Image<Vec3b> I_blur;
        resize(I, I_blur, Size(I.width()/BLUR_FACTOR, I.height()/BLUR_FACTOR));

        // Graph cut
        Graph<float,float,float> g(I_blur.width()*I_blur.height(), /*estimated # of edges*/ I.width()*I.height()*6);
        g.add_node(I_blur.width()*I_blur.height());
        for(int i=0; i<I_blur.width(); i++){
            for(int j=0; j<I_blur.height(); j++){
                // Source and sink weights
                g.add_tweights(I_blur.height()*i + j, source_weight(I_blur(i, j)), sink_weight(I_blur(i, j)));

                // Inter nodes weights :
                if(i>0)
                    g.add_edge(I_blur.height()*i + j, I_blur.height()*(i-1) + j, curve_weight(I_blur(i,j), I_blur(i-1,j)), curve_weight(I_blur(i,j), I_blur(i-1,j)));
                if(i<I_blur.width()-1)
                    g.add_edge(I_blur.height()*i + j, I_blur.height()*(i+1) + j, curve_weight(I_blur(i,j), I_blur(i+1,j)), curve_weight(I_blur(i,j), I_blur(i+1,j)));
                if(j>0)
                    g.add_edge(I_blur.height()*i + j, I_blur.height()*i + j-1, curve_weight(I_blur(i,j), I_blur(i,j-1)), curve_weight(I_blur(i,j), I_blur(i,j-1)));
                if(j<I_blur.height()-1)
                    g.add_edge(I_blur.height()*i + j, I_blur.height()*i + j+1, curve_weight(I_blur(i,j), I_blur(i,j+1)), curve_weight(I_blur(i,j), I_blur(i,j+1)));
            }
        }
        float flow = g.maxflow();
        Image<uchar> field_mask_blur(I_blur.width(), I_blur.height()), field_mask(I_blur.width(), I_blur.height());
        for(int i=0; i<I_blur.width(); i++){
            for(int j=0; j<I_blur.height(); j++){
                field_mask_blur(i,j) = (g.what_segment(i*I_blur.height() + j) == Graph<float,float,float>::SOURCE ?(uchar) 0 : (uchar) 255);
            }
        }

        // Deleting too small cc /!\ depends of BLUR_FACTOR /!\

        Image<int> connected_comps, stats;
        Mat centroids;
        int ccN = connectedComponentsWithStats(field_mask_blur, connected_comps, stats, centroids);  // Number of connected components
        for(int x=0; x<field_mask_blur.width(); x++){
            for(int y=0; y<field_mask_blur.height(); y++){
                field_mask_blur(x,y) = ((stats(CC_STAT_AREA, connected_comps(x,y)) > 1000 && connected_comps(x,y)) ? (uchar) 0 : (uchar) 255);  // Can be optimized
            }
        }
        resize(field_mask_blur, field_mask, Size(I.width(), I.height()));

        // Defining I_only_field
        Image<Vec3b> I_only_field(I.width(), I.height());
        for(int x=0; x<I_only_field.width(); x++){
            for(int y=0; y<I_only_field.height(); y++){
                I_only_field(x,y) = (field_mask(x,y) == (uchar) 255 ? I(x,y) : Vec3b(0,0,0));  // Can be optimized
            }
        }

        // Defining move_mask and move_mask_only_field
        pMOG2->apply(I, move_mask);
        for(int x=0; x<I_only_field.width(); x++){
            for(int y=0; y<I_only_field.height(); y++){
                move_mask_only_field(x,y) = (field_mask(x,y) == (uchar) 255 ? move_mask(x,y) : (uchar) 0);  // Can be optimized
            }
        }

        // Erosion and dilation
        Mat erode_ker = getStructuringElement( EROSION_TYPE,
                                                   Size( 2*EROSION_SIZE + 1, 2*EROSION_SIZE+1 ),
                                                   Point( EROSION_SIZE, EROSION_SIZE ) );
        Mat dilate_ker = getStructuringElement( EROSION_TYPE,
                                               Size( 2*DILATE_SIZE + 1, 2*DILATE_SIZE+1 ),
                                               Point( DILATE_SIZE, DILATE_SIZE ) );
        erode( move_mask_only_field, eroded, erode_ker );
        dilate( eroded, eroded_dilated, dilate_ker );

        // Deleting some connected components
        Image<int> cc_eroded_dilated, stats_eroded_dilated;
        Mat centroids_eroded_dilated;
        int ccn_eroded_dilated = connectedComponentsWithStats(eroded_dilated, cc_eroded_dilated, stats_eroded_dilated, centroids_eroded_dilated);  // Number of connected components
        for(int x=0; x<eroded_dilated.width(); x++){
            for(int y=0; y<eroded_dilated.height(); y++){
                float h_w_ratio = float(stats_eroded_dilated(CC_STAT_HEIGHT, cc_eroded_dilated(x,y))) / float(stats_eroded_dilated(CC_STAT_WIDTH, cc_eroded_dilated(x,y)));
                final_I(x,y) = (stats_eroded_dilated(CC_STAT_AREA, cc_eroded_dilated(x,y)) < 1500
                        || h_w_ratio > 7) ? (uchar) 0 : eroded_dilated(x,y);  // Can be optimized
            }
        }

        // Defining bounding boxes where there should be numbers
        // Could be optimized by reusing the upper part
        Image<int> cc_final_I, stats_final_I;
        Mat centroids_final_I;
        int ccn_final_I = connectedComponentsWithStats(final_I, cc_final_I, stats_final_I, centroids_final_I);  // Number of connected components
        vector<Rect> boundRect;
        Scalar color( 255, 0, 0);
        for(int i=1; i<ccn_final_I; i++){
            int px = stats_final_I(CC_STAT_LEFT, i) + stats_final_I(CC_STAT_WIDTH, i)/8;
            int py = stats_final_I(CC_STAT_TOP, i) + stats_final_I(CC_STAT_HEIGHT, i)/7;
            int w = (stats_final_I(CC_STAT_WIDTH, i)*6)/8;
            int h = (stats_final_I(CC_STAT_HEIGHT, i))/3;
            boundRect.emplace_back(Point2i(px, py), Size(w, h));
            // rectangle( I, boundRect[i-1].tl(), boundRect[i-1].br(), color, 2, 8, 0 );
        }

        vector<Image<Vec3b> > boxes;
        for(int i=0; i<boundRect.size(); i++){
            boxes.emplace_back(boundRect[i].width, boundRect[i].height);
            for(int x=0; x<boundRect[i].width; x++){
                for(int y=0; y<boundRect[i].height; y++){
                    boxes[i](x,y) = I(boundRect[i].x + x, boundRect[i].y + y);
                }
            }
            // imshow("box " + to_string(i), boxes[i]);
        }

        // Color clustering of all the boxes in the image
        vector<color_cluster> clusters(boxes.size());
        vector<Vec3b> most_freq_col(boxes.size());
        vector<int> most_freq_cluster(boxes.size());
        vector<vector<pair<int,int> > > vect_to_im(boxes.size());
        for(int i=0; i<boxes.size(); ++i){
            // Clustering of colors inside one box
            clusters[i] = color_cluster(image_to_vect_select(boxes[i], Vec3b(48,154,123), 1000, vect_to_im[i]), 3);  // Delete grass
            k_means_vec3b(clusters[i], 2,  3, 1000, 20);
            most_freq_col[i] = (clusters[i].sizes[0] > clusters[i].sizes[1] ? clusters[i].centroids[0] : clusters[i].centroids[1]);
            most_freq_cluster[i] = (clusters[i].sizes[0] > clusters[i].sizes[1] ? 0 : 1);
            cout << "Box " + to_string(i) << endl;
            for(int j=0; j<clusters[i].centroids.size();++j){
              cout << "Color : " << clusters[i].centroids[j] << ", size of the cluster : " << clusters[i].sizes[j] << endl;
            }
            cout << endl;
        }

        // Add the most frequent colors of this image to the vector of most frequent colors in all the images
        freq_colors.insert(freq_colors.begin(),most_freq_col.begin(), most_freq_col.end());

        // Compute the clustering over team_colors once and for all when we have enough data
        if(click_num == 10){
            team_colors = color_cluster(freq_colors,2);
            k_means_vec3b(team_colors, 2, 3, 1000, 20);
            cout << "Color of team 0 : " << team_colors.centroids[0] << " " << team_colors.sizes[0] << endl;
            cout << "Color of team 1 " << team_colors.centroids[1] << " " << team_colors.sizes[1] << endl;
        }

        /*
         * TODO : add collision detector when :
         * 1/ clustering outputs 2 centroids and each is near a team color (two different team on same picture)
         * 2/ clustering outputs 2 far away cc (line and a white player)
        */

        // Once we have an approximation of team colors, we try to detect the numbers by looking at the less frequent color in the cluster
        if(click_num > 10){
            vector<Image<Vec3b> > NB_boxes;
            size_t th = 3000 ;
            for(int i=0; i<boxes.size(); ++i){
                NB_boxes.emplace_back(boxes[i].width(), boxes[i].height(), Vec3b(0,255,0));
                // TODO : change to detect when there are no player
                int team = (sqdst(most_freq_col[i], team_colors.centroids[0]) > sqdst(most_freq_col[i], team_colors.centroids[1]) ? 1 : 0);
                for(int j=0; j<clusters[i].data.size(); j++){
                    NB_boxes[i](vect_to_im[i][j].first, vect_to_im[i][j].second) = (clusters[i].labels[j] == most_freq_cluster[i] ? Vec3b(0,0,0) : Vec3b(255,255,255));
                }
                cout << "Team of box " + to_string(i) + " : " + to_string(team) << endl;
                imshow("NB_box " + to_string(i), NB_boxes[i]);
            }
        }

        imshow_quarter("I",I);
        /*
        imshow_quarter("I_only_field",I_only_field);
        imshow_quarter("field_mask",field_mask);
        imshow_quarter("move_mask", move_mask);
        imshow_quarter("move_mask_only_field", move_mask_only_field);
        imshow_quarter("final_I", final_I);
        imshow_quarter("eroded_dilated", eroded_dilated);
        */
        waitKey();
        cout << endl;
    }
    waitKey(0);

    return 0;
}

void assign_labels(const vector<Vec3b>& points, vector<int>& labels, vector<Vec3b>& centroids){
    for(int i=0; i<points.size(); i++){
        int new_label =0;
        double min = sqdst(points[i], centroids[0]);
        for(int j=1; j<centroids.size(); j++){
            if(min > sqdst(points[i], centroids[j])){
                min = sqdst(points[i], centroids[j]);
                new_label = j;
            }
        }
        labels[i] = new_label;
    }
}

double compute_compactness(const vector<Vec3b>& points, vector<int>& labels, vector<Vec3b>& centroids){
    double res =0.0;
    for(int i=0; i<points.size(); i++){
        res += sqdst(points[i], centroids[labels[i]]);
    }
    return res;
}

void compute_centroids(const vector<Vec3b>& points, vector<int>& labels, vector<Vec3b>& centroids, vector<int>& cluster_sizes){
    vector<vector<int> > new_centroids(centroids.size());
    for(int i=0; i<centroids.size(); i++){
        new_centroids[i].push_back(0); new_centroids[i].push_back(0); new_centroids[i].push_back(0);
    }

    vector<int> new_cluster_sizes(centroids.size(), 0);
    for(int i=0; i<points.size();i++){
        new_cluster_sizes[labels[i]]++;
        new_centroids[labels[i]][0] += points[i].val[0];
        new_centroids[labels[i]][1] += points[i].val[1];
        new_centroids[labels[i]][2] += points[i].val[2];
    }

    for(int i=0; i<centroids.size();i++){
        if(new_cluster_sizes[i]){
            new_centroids[i][0] /= new_cluster_sizes[i];
            new_centroids[i][1] /= new_cluster_sizes[i];
            new_centroids[i][2] /= new_cluster_sizes[i];
        }
        cluster_sizes[i] = new_cluster_sizes[i];
    }

    for(int i=0; i<centroids.size(); i++){
        centroids[i] = Vec3b((uchar) new_centroids[i][0], (uchar) new_centroids[i][1], (uchar) new_centroids[i][2]);
    }
}

/*
 * K : number of clusters
 * rand_rep : number of time we repeat the algorithm with another random intialization
 * eps : when compactness is below we stop
 * max_rep : when rep is above we stop
 */
void k_means_vec3b(color_cluster& cluster, size_t K, size_t rand_rep, double eps, size_t max_rep){
    if(!cluster.data.size()){
        cout << " Empty dataset ! " << endl;
        cluster.compactness = 0.0;
        return;
    }
    color_cluster best_clusters(cluster.data, cluster.K), cur_clusters(cluster.data, K);
    for(int r=0; r<rand_rep; r++){
        for(int i=0;i<K;++i){
            int blue = rand() % 256;
            int green = rand() % 256;
            int red = rand() % 256;
            cur_clusters.centroids[i] = Vec3b(uchar(blue),uchar(green),uchar(red));
        }
        assign_labels(cluster.data, cur_clusters.labels, cur_clusters.centroids);
        compute_centroids(cluster.data, cur_clusters.labels, cur_clusters.centroids, cur_clusters.sizes);
        cur_clusters.compactness = compute_compactness(cluster.data, cur_clusters.labels, cur_clusters.centroids);
        size_t rep = 1;
        while(cur_clusters.compactness > eps && rep < max_rep){
            assign_labels(cluster.data, cur_clusters.labels, cur_clusters.centroids);
            compute_centroids(cluster.data, cur_clusters.labels, cur_clusters.centroids, cur_clusters.sizes);
            cur_clusters.compactness = compute_compactness(cluster.data, cur_clusters.labels, cur_clusters.centroids);
            rep++;
        }
        if(r){
            if(cur_clusters.compactness < best_clusters.compactness){
                best_clusters.compactness = cur_clusters.compactness;
                best_clusters.labels = cur_clusters.labels;
                best_clusters.centroids = cur_clusters.centroids;
                best_clusters.sizes = cur_clusters.sizes;
            }
        }
        else{
            best_clusters.compactness = cur_clusters.compactness;
            best_clusters.labels = cur_clusters.labels;
            best_clusters.centroids = cur_clusters.centroids;
            best_clusters.sizes = cur_clusters.sizes;
        }
    }
    cluster = best_clusters;
}

vector<Vec3b> image_to_vect_select(const Image<Vec3b>& inpt, Vec3b color, float dst, vector<pair<int, int> >& vect_to_im){
    vector<Vec3b> res;
    vect_to_im.resize(0);
    for(int x=0; x<inpt.width(); x++){
        for(int y=0; y<inpt.height(); y++){
            if(sqdst(inpt(x,y), color) > dst) {
                res.push_back(inpt(x,y));
                vect_to_im.emplace_back(x,y);
            }
        }
    }
    return res;
}

