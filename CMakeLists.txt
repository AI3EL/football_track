cmake_minimum_required(VERSION 3.12)
project(football_track)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
link_directories(/media/abel/DATA/Anaconda/lib)
add_executable(football_track
        src/main.cpp src/main.hpp
        src/image.cpp src/image.h
        src/graphCuts.cpp src/graphCuts.h
        src/maxflow/graph.cpp
        src/field_cut.cpp src/field_cut.h
        src/clustering.cpp src/clustering.h src/Player.cpp src/Player.hpp)


target_link_libraries(football_track ${OpenCV_LIBS})