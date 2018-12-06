#include "graphCuts.h"

using namespace std;
using namespace cv;

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////
/*

void testGCuts()
{
	Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);
	g.add_node(2);
	g.add_tweights( 0,   /* capacities */  1, 5 );
	g.add_tweights( 1,   /* capacities */  6, 1 );
	g.add_edge( 0, 1,    /* capacities */  4, 3 );
	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<2;i++)
		if (g.what_segment(i) == Graph<int,int,int>::SOURCE)
			cout << i << " is in the SOURCE set" << endl;
		else
			cout << i << " is in the SINK set" << endl;
}

int main() {
	Image<Vec3b> I= Image<Vec3b>(imread("../fishes.jpg"));
    Graph<float,float,float> g(I.width()*I.height(), /*estimated # of edges*/ I.width()*I.height()*6);
    g.add_node(I.width()*I.height());
    for(int i=0; i<I.width(); i++){
        for(int j=0; j<I.height(); j++){
            // Source and sink weights
            g.add_tweights(I.height()*i + j, source_weight(I(i, j)), sink_weight(I(i, j)));

            // Inter nodes weights :
            if(i>0)
                g.add_edge(I.height()*i + j, I.height()*(i-1) + j, curve_weight(I(i,j), I(i-1,j)), curve_weight(I(i,j), I(i-1,j)));
            if(i<I.width()-1)
                g.add_edge(I.height()*i + j, I.height()*(i+1) + j, curve_weight(I(i,j), I(i+1,j)), curve_weight(I(i,j), I(i+1,j)));
            if(j>0)
                g.add_edge(I.height()*i + j, I.height()*i + j-1, curve_weight(I(i,j), I(i,j-1)), curve_weight(I(i,j), I(i,j-1)));
            if(j<I.height()-1)
                g.add_edge(I.height()*i + j, I.height()*i + j+1, curve_weight(I(i,j), I(i,j+1)), curve_weight(I(i,j), I(i,j+1)));
        }
    }
    float flow = g.maxflow();
    Image<float> R(I.width(), I.height());
    for(int i=0; i<I.width(); i++){
        for(int j=0; j<I.height(); j++){
            R(i,j) = (g.what_segment(i*I.height() + j) == Graph<float,float,float>::SOURCE ? 0 : 255);
        }
    }
    imshow("I",I);
    imshow("R",R);
    waitKey(0);
	return 0;
}
*/