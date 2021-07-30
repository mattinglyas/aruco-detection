#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#define IMAGE_PATH "/home/pi/Pictures/aruco_data/"

#include <sstream>

int main(int argc, char ** argv) {    
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open webcam" << std::endl;
        return 1;
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    int num = 0;

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    for (;;) {
        cv::Mat frame;
        camera >> frame;
        cv::imshow("Camera", frame);

        char c = cv::waitKey(20);

        if (c == 'c') {
            std::stringstream stream;
            stream << IMAGE_PATH << num << ".jpg";
            cv::imwrite(stream.str(), frame);
            num++;

            std::cout << "wrote image: " << stream.str() << std::endl;
        }

        if (c == 'q') {
            break;
        }
    }

    return 0;
}   