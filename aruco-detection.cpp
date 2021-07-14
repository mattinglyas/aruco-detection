#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/types.hpp>
#include <sstream>


using namespace std;
using namespace cv;

int main(int argc, char ** argv) {
    Mat marker_image;

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    //aruco::drawMarker(dictionary, 33, 200, marker_image, 1);

    //imshow("detecting this marker", marker_image);
    //waitKey(1000);

    Mat detection_image = imread("./testimage.png");

    imshow("detecting in this image", detection_image);
    waitKey(1000);

    Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();

    vector<vector<Point2f>> marker_corners, rejected_candidates;

    vector<int> marker_ids;

    detectMarkers(detection_image, dictionary, marker_corners, marker_ids, parameters, rejected_candidates);

    for (int i = 0; i < marker_ids.size(); i++) {
        line(detection_image, marker_corners[i][0], marker_corners[i][1], Scalar(0,255,0), 1);
        line(detection_image, marker_corners[i][1], marker_corners[i][2], Scalar(0,255,0), 1);
        line(detection_image, marker_corners[i][2], marker_corners[i][3], Scalar(0,255,0), 1);
        line(detection_image, marker_corners[i][3], marker_corners[i][0], Scalar(0,255,0), 1);

        stringstream stream;
        stream << marker_ids[i];
        putText(detection_image, stream.str(), marker_corners[i][0], FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0), 1, LINE_AA);
    }

    imshow("detecting in this image", detection_image);
    waitKey(0);

    return 0;
}