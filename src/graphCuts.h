#ifndef TP4_GRAPHCUTS_H
#define TP4_GRAPHCUTS_H

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "maxflow/graph.h"
#include "image.h"

inline float sqdst(const Vec3b &x, const Vec3b &y) {
    float res = 0.0;
    for (int i = 0; i < 2; i++) res += (float(x[i]) - float(y[i])) * (float(x[i]) - float(y[i]));
    return res;
}

inline float curve_weight(const Vec3b &x, const Vec3b &y) {
    float v = sqdst(x, y);
    return 1000 / (1 + v);
}

inline float source_weight(const Vec3b &x) {
    float v = sqdst(x, Vec3b(35, 145, 110));
    return 1000 / (1 + v);
}

inline float sink_weight(const Vec3b &x) {
    float v = sqdst(x, Vec3b(80, 130, 120));
    return 1000 / (1 + v);
}

#endif //TP4_GRAPHCUTS_H
