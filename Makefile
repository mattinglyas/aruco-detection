# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/Documents/aruco-detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/Documents/aruco-detection

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/pi/Documents/aruco-detection/CMakeFiles /home/pi/Documents/aruco-detection/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/pi/Documents/aruco-detection/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named aruco-detection

# Build rule for target.
aruco-detection: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 aruco-detection
.PHONY : aruco-detection

# fast build rule for target.
aruco-detection/fast:
	$(MAKE) -f CMakeFiles/aruco-detection.dir/build.make CMakeFiles/aruco-detection.dir/build
.PHONY : aruco-detection/fast

#=============================================================================
# Target rules for targets named image-capture

# Build rule for target.
image-capture: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 image-capture
.PHONY : image-capture

# fast build rule for target.
image-capture/fast:
	$(MAKE) -f CMakeFiles/image-capture.dir/build.make CMakeFiles/image-capture.dir/build
.PHONY : image-capture/fast

aruco-detection.o: aruco-detection.cpp.o

.PHONY : aruco-detection.o

# target to build an object file
aruco-detection.cpp.o:
	$(MAKE) -f CMakeFiles/aruco-detection.dir/build.make CMakeFiles/aruco-detection.dir/aruco-detection.cpp.o
.PHONY : aruco-detection.cpp.o

aruco-detection.i: aruco-detection.cpp.i

.PHONY : aruco-detection.i

# target to preprocess a source file
aruco-detection.cpp.i:
	$(MAKE) -f CMakeFiles/aruco-detection.dir/build.make CMakeFiles/aruco-detection.dir/aruco-detection.cpp.i
.PHONY : aruco-detection.cpp.i

aruco-detection.s: aruco-detection.cpp.s

.PHONY : aruco-detection.s

# target to generate assembly for a file
aruco-detection.cpp.s:
	$(MAKE) -f CMakeFiles/aruco-detection.dir/build.make CMakeFiles/aruco-detection.dir/aruco-detection.cpp.s
.PHONY : aruco-detection.cpp.s

aruco-kalman.o: aruco-kalman.cpp.o

.PHONY : aruco-kalman.o

# target to build an object file
aruco-kalman.cpp.o:
	$(MAKE) -f CMakeFiles/aruco-detection.dir/build.make CMakeFiles/aruco-detection.dir/aruco-kalman.cpp.o
.PHONY : aruco-kalman.cpp.o

aruco-kalman.i: aruco-kalman.cpp.i

.PHONY : aruco-kalman.i

# target to preprocess a source file
aruco-kalman.cpp.i:
	$(MAKE) -f CMakeFiles/aruco-detection.dir/build.make CMakeFiles/aruco-detection.dir/aruco-kalman.cpp.i
.PHONY : aruco-kalman.cpp.i

aruco-kalman.s: aruco-kalman.cpp.s

.PHONY : aruco-kalman.s

# target to generate assembly for a file
aruco-kalman.cpp.s:
	$(MAKE) -f CMakeFiles/aruco-detection.dir/build.make CMakeFiles/aruco-detection.dir/aruco-kalman.cpp.s
.PHONY : aruco-kalman.cpp.s

image-capture.o: image-capture.cpp.o

.PHONY : image-capture.o

# target to build an object file
image-capture.cpp.o:
	$(MAKE) -f CMakeFiles/image-capture.dir/build.make CMakeFiles/image-capture.dir/image-capture.cpp.o
.PHONY : image-capture.cpp.o

image-capture.i: image-capture.cpp.i

.PHONY : image-capture.i

# target to preprocess a source file
image-capture.cpp.i:
	$(MAKE) -f CMakeFiles/image-capture.dir/build.make CMakeFiles/image-capture.dir/image-capture.cpp.i
.PHONY : image-capture.cpp.i

image-capture.s: image-capture.cpp.s

.PHONY : image-capture.s

# target to generate assembly for a file
image-capture.cpp.s:
	$(MAKE) -f CMakeFiles/image-capture.dir/build.make CMakeFiles/image-capture.dir/image-capture.cpp.s
.PHONY : image-capture.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... aruco-detection"
	@echo "... rebuild_cache"
	@echo "... image-capture"
	@echo "... aruco-detection.o"
	@echo "... aruco-detection.i"
	@echo "... aruco-detection.s"
	@echo "... aruco-kalman.o"
	@echo "... aruco-kalman.i"
	@echo "... aruco-kalman.s"
	@echo "... image-capture.o"
	@echo "... image-capture.i"
	@echo "... image-capture.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

