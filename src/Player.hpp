//
// Created by abel on 19/12/18.
//

#ifndef FOOTBALL_TRACK_PLAYER_HPP
#define FOOTBALL_TRACK_PLAYER_HPP
#include <opencv2/opencv.hpp>

using namespace cv;

class Player {
public:
    static int instance;
    int index;
    int number = 0;
    Point position;
    int tracking_point_id;
    int age;
    int box_id;
    bool team;

    Player() : index(instance++){}
    Player(Point position, int box_id, int track_id) : index(instance++), position(position),
    tracking_point_id(track_id), box_id(box_id), age(0) {}
};


#endif //FOOTBALL_TRACK_PLAYER_HPP
