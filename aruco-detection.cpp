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

#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {
    Mat marker_image;

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();

    VideoCapture camera(0);
    if (!camera.isOpened()) {
        cerr << "ERROR: Could not open webcam" << endl;
        return 1;
    }

    namedWindow("Camera", CV_WINDOW_AUTOSIZE);

    while(true) {
        struct timespec begin, end;
        double wall_time;
        clock_gettime(CLOCK_MONOTONIC, &begin);

        Mat frame;
        camera >> frame;

        vector<vector<Point2f>> marker_corners, rejected_candidates;
        vector<int> marker_ids;

        detectMarkers(frame, dictionary, marker_corners, marker_ids, parameters, rejected_candidates);

        for (int i = 0; i < marker_ids.size(); i++) {
            line(frame, marker_corners[i][0], marker_corners[i][1], Scalar(0,255,0), 1);
            line(frame, marker_corners[i][1], marker_corners[i][2], Scalar(0,255,0), 1);
            line(frame, marker_corners[i][2], marker_corners[i][3], Scalar(0,255,0), 1);
            line(frame, marker_corners[i][3], marker_corners[i][0], Scalar(0,255,0), 1);

            stringstream stream;
            stream << marker_ids[i];
            putText(frame, stream.str(), marker_corners[i][0], FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 1, LINE_AA);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        wall_time = end.tv_sec - begin.tv_sec;
        wall_time += (end.tv_nsec - begin.tv_nsec) / 1000000000.00;

        stringstream stream2;
        stream2 << "fps: " << 1/wall_time;
        putText(frame, stream2.str(), Point(40,40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, LINE_AA);

        imshow("Camera", frame);
        waitKey(1);
    }

    return 0;
}