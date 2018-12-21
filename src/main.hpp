#ifndef FOOTBALL_TRACK_MAIN_HPP
#define FOOTBALL_TRACK_MAIN_HPP

#include <opencv2/opencv.hpp>
#include "image.h"
#include <string>
#include <queue>
#include "clustering.h"
#include "field_cut.h"
#include "Player.hpp"

using namespace cv;
using namespace std;

// Put all pixels of an image in a vector
template<typename T>
vector<T> image_to_vect(const Image<T> &inpt) {
    vector<T> res;
    for (int x = 0; x < inpt.width(); x++) {
        for (int y = 0; y < inpt.height(); y++) {
            res.push_back(inpt(x, y));
        }
    }
    return res;
}

// Put some pixels of an image in a vector and keep in vect_to_im indices of the selected pixels
vector<Vec3b>
image_to_vect_select(const Image<Vec3b> &inpt, Vec3b color, float dst, vector<pair<int, int> > &vect_to_im);

// Shows the image in a miniature
template<typename T>
void imshow_quarter(string str, const Image<T> &src) {
    Size display_size(900, 450);
    Image<T> dezoomed;
    resize(src, dezoomed, display_size);
    imshow(str, dezoomed);
}

template<typename T>
void imshow_half(string str, const Image<T> &src) {
    Size display_size(1800, 450);
    Image<T> dezoomed;
    resize(src, dezoomed, display_size);
    imshow(str, dezoomed);
}


#endif //FOOTBALL_TRACK_MAIN_HPP
