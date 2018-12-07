#ifndef FOOTBALL_TRACK_FIELD_CUT_H
#define FOOTBALL_TRACK_FIELD_CUT_H

#include <opencv2/opencv.hpp>
#include "image.h"
#include "graphCuts.h"

using namespace cv;
using namespace std;

void field_blur_cut(const Image<Vec3b>& I, const int& blur_factor, Image<uchar>& field_mask_blur);

#endif //FOOTBALL_TRACK_FIELD_CUT_H
