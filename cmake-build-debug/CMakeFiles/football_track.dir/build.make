# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /media/abel/DATA/clion-2018.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /media/abel/DATA/clion-2018.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/football_track.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/football_track.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/football_track.dir/flags.make

CMakeFiles/football_track.dir/src/main.cpp.o: CMakeFiles/football_track.dir/flags.make
CMakeFiles/football_track.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/football_track.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/football_track.dir/src/main.cpp.o -c "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/main.cpp"

CMakeFiles/football_track.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/football_track.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/main.cpp" > CMakeFiles/football_track.dir/src/main.cpp.i

CMakeFiles/football_track.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/football_track.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/main.cpp" -o CMakeFiles/football_track.dir/src/main.cpp.s

CMakeFiles/football_track.dir/src/image.cpp.o: CMakeFiles/football_track.dir/flags.make
CMakeFiles/football_track.dir/src/image.cpp.o: ../src/image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/football_track.dir/src/image.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/football_track.dir/src/image.cpp.o -c "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/image.cpp"

CMakeFiles/football_track.dir/src/image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/football_track.dir/src/image.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/image.cpp" > CMakeFiles/football_track.dir/src/image.cpp.i

CMakeFiles/football_track.dir/src/image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/football_track.dir/src/image.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/image.cpp" -o CMakeFiles/football_track.dir/src/image.cpp.s

CMakeFiles/football_track.dir/src/maxflow/graph.cpp.o: CMakeFiles/football_track.dir/flags.make
CMakeFiles/football_track.dir/src/maxflow/graph.cpp.o: ../src/maxflow/graph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/football_track.dir/src/maxflow/graph.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/football_track.dir/src/maxflow/graph.cpp.o -c "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/maxflow/graph.cpp"

CMakeFiles/football_track.dir/src/maxflow/graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/football_track.dir/src/maxflow/graph.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/maxflow/graph.cpp" > CMakeFiles/football_track.dir/src/maxflow/graph.cpp.i

CMakeFiles/football_track.dir/src/maxflow/graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/football_track.dir/src/maxflow/graph.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/src/maxflow/graph.cpp" -o CMakeFiles/football_track.dir/src/maxflow/graph.cpp.s

# Object files for target football_track
football_track_OBJECTS = \
"CMakeFiles/football_track.dir/src/main.cpp.o" \
"CMakeFiles/football_track.dir/src/image.cpp.o" \
"CMakeFiles/football_track.dir/src/maxflow/graph.cpp.o"

# External object files for target football_track
football_track_EXTERNAL_OBJECTS =

football_track: CMakeFiles/football_track.dir/src/main.cpp.o
football_track: CMakeFiles/football_track.dir/src/image.cpp.o
football_track: CMakeFiles/football_track.dir/src/maxflow/graph.cpp.o
football_track: CMakeFiles/football_track.dir/build.make
football_track: /usr/local/lib/libopencv_objdetect.so.3.4.2
football_track: /usr/local/lib/libopencv_ml.so.3.4.2
football_track: /usr/local/lib/libopencv_superres.so.3.4.2
football_track: /usr/local/lib/libopencv_shape.so.3.4.2
football_track: /usr/local/lib/libopencv_videostab.so.3.4.2
football_track: /usr/local/lib/libopencv_video.so.3.4.2
football_track: /usr/local/lib/libopencv_photo.so.3.4.2
football_track: /usr/local/lib/libopencv_stitching.so.3.4.2
football_track: /usr/local/lib/libopencv_dnn.so.3.4.2
football_track: /usr/local/lib/libopencv_calib3d.so.3.4.2
football_track: /usr/local/lib/libopencv_features2d.so.3.4.2
football_track: /usr/local/lib/libopencv_flann.so.3.4.2
football_track: /usr/local/lib/libopencv_highgui.so.3.4.2
football_track: /usr/local/lib/libopencv_videoio.so.3.4.2
football_track: /usr/local/lib/libopencv_imgcodecs.so.3.4.2
football_track: /usr/local/lib/libopencv_imgproc.so.3.4.2
football_track: /usr/local/lib/libopencv_core.so.3.4.2
football_track: CMakeFiles/football_track.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable football_track"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/football_track.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/football_track.dir/build: football_track

.PHONY : CMakeFiles/football_track.dir/build

CMakeFiles/football_track.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/football_track.dir/cmake_clean.cmake
.PHONY : CMakeFiles/football_track.dir/clean

CMakeFiles/football_track.dir/depend:
	cd "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track" "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track" "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug" "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug" "/media/abel/DATA/Cours/3A/INF573 Analyse d'images .../Project/football_track/cmake-build-debug/CMakeFiles/football_track.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/football_track.dir/depend

