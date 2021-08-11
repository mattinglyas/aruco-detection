#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/types.hpp>

#include "aruco-kalman.h"

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>

/* Default values for certain settings */
#define DEF_CALIBRATION "./calibration.yml"
#define DEF_MARKER_LENGTH (0.1f)

// from https://docs.opencv.org/3.4/df/d4a/tutorial_charuco_detection.html
static bool readCameraParameters(std::string filename, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients, int &width, int &height) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_coefficients;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    return (camera_matrix.size() == cv::Size(3,3));
}

int main(int argc, char ** argv) {
    /*************************************************************************
     * Define parameters and constants
     * 
     *************************************************************************/
    char * calibration_filename = (char*)DEF_CALIBRATION;
    char * input_filename = nullptr;
    int conf_width = 0;
    int conf_height = 0;
    int read_width = 0;
    int read_height = 0;
    float marker_length = DEF_MARKER_LENGTH;

    /* drawing settings */
    bool draw_pose = true;
    bool draw_matrix = true;
    bool draw_bounds = true;

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    /*************************************************************************
     * Parse arguments from command line (this is a very bad way to do arg
     *  parsing. TODO look into a more elegant way to do command line args)
     * 
     *************************************************************************/

    if (argc > 1) {
        calibration_filename = argv[1];

        if (argc > 2) {
            marker_length = std::stof(argv[2]);

            if (argc > 4) {
                read_width = std::atoi(argv[3]);
                read_height = std::atoi(argv[4]);

                if (argc > 5) {
                    input_filename = argv[5];
                }
            }
        }
    }

    /*************************************************************************
     * Read calibration file and open camera
     * 
     *************************************************************************/

    if (readCameraParameters(calibration_filename, camera_matrix, dist_coeffs, conf_width, conf_height) == false) {
        std::cerr << "ERROR: Invalid calibration file " << calibration_filename << std::endl;
        return 1;
    } 
    
    /* WARNING this assumes that the fov of the camera is the same */
    if (read_width != 0) {
        /* Scale the camera matrix by the target resolution */
        double scale = double(read_width) / double(conf_width); 
        camera_matrix = camera_matrix * scale;
        std::cout << "scaling camera matrix by " << scale << std::endl;
    } else {
        /* Assume same resolution as config */
        read_width = conf_width;
        read_height = conf_height;
    }

    cv::VideoCapture camera;
    if (input_filename != nullptr) {
        camera = cv::VideoCapture(input_filename);
    } else {
        camera = cv::VideoCapture(0);
    }

    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open webcam" << std::endl;
        return 1;
    }
    
    std::cout << "setting resolution to " << read_width << " x " << read_height << std::endl;
    camera.set(cv::CAP_PROP_FRAME_WIDTH, read_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, read_height);
    
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    /*************************************************************************
     * Set up kalman filters for each marker
     *   using an array for o(1) access time, but costs more memory
     * 
     *************************************************************************/
    cv::KalmanFilter kfs[250];
    for (int i = 0; i < 250; i++) {
        initKalmanFilter(kfs[i], 0);
    }
    
    /*************************************************************************
     * Detect aruco markers and run pose estimation
     * 
     *************************************************************************/

    double start;
    double end;
    double dt = 0;

    while(true) {
        /* Used in profiling performance */
        start = (double)cv::getTickCount();

        cv::Mat frame;
        camera >> frame;

        if (frame.empty()) {
            return 0;
        }

        std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;
        std::vector<int> marker_ids;

        cv::aruco::detectMarkers(
            frame, 
            dictionary, 
            marker_corners, 
            marker_ids, 
            parameters, 
            rejected_candidates
        );

        /* Draw and run pose estimation if markers were detected in frame */
        if (marker_ids.size() > 0) {
            if (draw_bounds)
            {
                cv::aruco::drawDetectedMarkers(
                    frame, 
                    marker_corners, 
                    marker_ids
                );
            }

            std::vector<cv::Vec3d> rvecs, tvecs;

            cv::aruco::estimatePoseSingleMarkers(
                marker_corners,
                marker_length,
                camera_matrix,
                dist_coeffs,
                rvecs,
                tvecs
            );

            for (int i = 0; i < marker_ids.size(); i++) {
                int marker_id = marker_ids[i];
                cv::KalmanFilter kf = kfs[marker_id];

                updateTransitionMatrix(kf, dt);
                cv::Mat translation_estimated(3, 1, CV_64F);
                cv::Mat rotation_estimated(3, 3, CV_64F);
                cv::Mat speed_estimated(3, 1, CV_64F);

                predictKalmanFilter(kf, translation_estimated, rotation_estimated, speed_estimated);

                cv::Mat measurements(6, 1, CV_64F);

                fillMeasurements(measurements, tvecs[i], rvecs[i]);

                /* TODO find a way to reject bad pose estimations */
                updateKalmanFilter(kf, measurements, translation_estimated, rotation_estimated, speed_estimated);

                cv::Mat rotation_estimated_euler = rot2euler(rotation_estimated);

                /* Draw pose estimation markers and other data on top of marker */
                if (draw_pose)
                    cv::aruco::drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], translation_estimated, 0.1f);

                if (draw_matrix)
                {
                    std::stringstream stream;
                    stream << "R: " << std::setprecision(2) << rvecs[i];
                    cv::putText(frame, stream.str(), cv::Point(marker_corners[i][2].x - 50, marker_corners[i][2].y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 3);
                    cv::putText(frame, stream.str(), cv::Point(marker_corners[i][2].x - 50, marker_corners[i][2].y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
                    stream.clear();
                    stream.str("");
                    stream << "T: " << std::setprecision(2) << tvecs[i];
                    cv::putText(frame, stream.str(), cv::Point(marker_corners[i][2].x - 50, marker_corners[i][2].y + 50), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 3);
                    cv::putText(frame, stream.str(), cv::Point(marker_corners[i][2].x - 50, marker_corners[i][2].y + 50), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
                    // stream.clear();
                    // stream.str("");
                    // stream << "T_e: " << translation_estimated;
                    // cv::putText(frame, stream.str(), cv::Point(marker_corners[i][0].x - 50, marker_corners[i][0].y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 3);
                    // cv::putText(frame, stream.str(), cv::Point(marker_corners[i][0].x - 50, marker_corners[i][0].y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
                }
            }
        }

        /* Calculate FPS assuming no time is taken between frames (this is not the case due to waitkey) */
        end = (double)cv::getTickCount();
        dt = (end - start) / cv::getTickFrequency();

        std::stringstream stream2;
        stream2 << "fps: " << 1/dt;
        cv::putText(frame, stream2.str(), cv::Point(40,40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        /* Display processed frame */
        cv::imshow("Camera", frame);
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
        else if (c == 'm')
            draw_matrix = !draw_matrix;
        else if (c == 'p')
            draw_pose = !draw_pose;
        else if (c == 'b')
            draw_bounds = !draw_bounds;
    }

    return 0;
}
