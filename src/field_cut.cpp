#include "field_cut.h"

void field_cut(const Image<Vec3b>& I, const int& blur_factor, Image<uchar>& field_mask_blur) {
    // Loss of details to have faster graph cut
    Image<Vec3b> I_blur;
    resize(I, I_blur, Size(I.width() / blur_factor, I.height() / blur_factor));

    // Graph cut
    Graph<float, float, float> g(I_blur.width() * I_blur.height(), /*estimated # of edges*/
                                 I.width() * I.height() * 6);
    g.add_node(I_blur.width() * I_blur.height());
    for (int i = 0; i < I_blur.width(); i++) {
        for (int j = 0; j < I_blur.height(); j++) {
            // Source and sink weights
            g.add_tweights(I_blur.height() * i + j, source_weight(I_blur(i, j)), sink_weight(I_blur(i, j)));

            // Inter nodes weights :
            if (i > 0)
                g.add_edge(I_blur.height() * i + j, I_blur.height() * (i - 1) + j,
                           curve_weight(I_blur(i, j), I_blur(i - 1, j)),
                           curve_weight(I_blur(i, j), I_blur(i - 1, j)));
            if (i < I_blur.width() - 1)
                g.add_edge(I_blur.height() * i + j, I_blur.height() * (i + 1) + j,
                           curve_weight(I_blur(i, j), I_blur(i + 1, j)),
                           curve_weight(I_blur(i, j), I_blur(i + 1, j)));
            if (j > 0)
                g.add_edge(I_blur.height() * i + j, I_blur.height() * i + j - 1,
                           curve_weight(I_blur(i, j), I_blur(i, j - 1)),
                           curve_weight(I_blur(i, j), I_blur(i, j - 1)));
            if (j < I_blur.height() - 1)
                g.add_edge(I_blur.height() * i + j, I_blur.height() * i + j + 1,
                           curve_weight(I_blur(i, j), I_blur(i, j + 1)),
                           curve_weight(I_blur(i, j), I_blur(i, j + 1)));
        }
    }
    float flow = g.maxflow();
    field_mask_blur = Image<uchar>(I_blur.width(), I_blur.height());
    for (int i = 0; i < I_blur.width(); i++) {
        for (int j = 0; j < I_blur.height(); j++) {
            field_mask_blur(i, j) = (g.what_segment(i * I_blur.height() + j) == Graph<float, float, float>::SOURCE
                                     ? (uchar) 0 : (uchar) 255);
        }
    }
}