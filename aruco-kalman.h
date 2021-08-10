#ifndef _aruco_kalman
#define _aruco_kalman

#include <opencv2/video/tracking.hpp>

void updateTransitionMatrix(cv::KalmanFilter &kf, double dt);
void initKalmanFilter(cv::KalmanFilter &kf, double dt);
void predictKalmanFilter( cv::KalmanFilter &kf,
                          cv::Mat &translation_estimated,
                          cv::Mat &rotation_estimated,
                          cv::Mat &speed_estimated);
cv::Mat DoubleMatFromVec3b(cv::Vec3b in);
cv::Mat rot2euler(const cv::Mat &rotationMatrix);
cv::Mat euler2rot(const cv::Mat &euler);
void fillMeasurements(cv::Mat &measurements,
                      const cv::Vec3d translation_measured,
                      const cv::Vec3d rvec);
void updateKalmanFilter(cv::KalmanFilter &KF,
                        cv::Mat &measurement,
                        cv::Mat &translation_estimated,
                        cv::Mat &rotation_estimated,
                        cv::Mat &speed_estimated);
#endif 