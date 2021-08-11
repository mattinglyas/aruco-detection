// from https://github.com/Pold87/aruco-localization/blob/master/detect_board.cpp

#include "aruco-kalman.h"
#include <opencv2/calib3d.hpp>

void updateTransitionMatrix(cv::KalmanFilter &kf, double dt) {
    // dynamic model
    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]  
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
    
    // position
    double dt2 = 0.5 * pow(dt, 2);
    kf.transitionMatrix.at<double>(0,3) = dt;
    kf.transitionMatrix.at<double>(1,4) = dt;
    kf.transitionMatrix.at<double>(2,5) = dt;
    kf.transitionMatrix.at<double>(3,6) = dt;
    kf.transitionMatrix.at<double>(4,7) = dt;
    kf.transitionMatrix.at<double>(5,8) = dt;
    kf.transitionMatrix.at<double>(0,6) = dt2;
    kf.transitionMatrix.at<double>(1,7) = dt2;
    kf.transitionMatrix.at<double>(2,8) = dt2;

    // rotation
    kf.transitionMatrix.at<double>(9,12) = dt;
    kf.transitionMatrix.at<double>(10,13) = dt;
    kf.transitionMatrix.at<double>(11,14) = dt;
    kf.transitionMatrix.at<double>(12,15) = dt;
    kf.transitionMatrix.at<double>(13,16) = dt;
    kf.transitionMatrix.at<double>(14,17) = dt;
    kf.transitionMatrix.at<double>(9,15) = dt2;
    kf.transitionMatrix.at<double>(10,16) = dt2;
    kf.transitionMatrix.at<double>(11,17) = dt2;

    // measurement model
    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

    kf.measurementMatrix.at<double>(0,0) = 1; // x
    kf.measurementMatrix.at<double>(1,1) = 1; // y
    kf.measurementMatrix.at<double>(2,2) = 1; // z
    kf.measurementMatrix.at<double>(3,9) = 1; // roll
    kf.measurementMatrix.at<double>(4,10) = 1; // pitch
    kf.measurementMatrix.at<double>(5,11) = 1; // yaw
}

void initKalmanFilter(cv::KalmanFilter &kf, double dt) {
    kf.init(18, 6, 0, CV_64F); // init kalman filter

    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(0.0001)); // process noise
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(0.0001)); // measurement noise
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1)); // error covariance

    updateTransitionMatrix(kf, dt);
}

void predictKalmanFilter( cv::KalmanFilter &kf,
                          cv::Mat &translation_estimated,
                          cv::Mat &rotation_estimated,
                          cv::Mat &speed_estimated) {

    // First predict, to update the internal statePre variable
    // This will give us a prediction of the variables, even without
    // a detected marker
    cv::Mat estimated = kf.predict();

    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    
    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);

    // Estimated speed
    speed_estimated.at<double>(0) = estimated.at<double>(3);
    speed_estimated.at<double>(1) = estimated.at<double>(4);
    speed_estimated.at<double>(2) = estimated.at<double>(5);
    
    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);
}

// Converts a given Rotation Matrix to Euler angles
cv::Mat rot2euler(const cv::Mat & rotationMatrix)
{
    cv::Mat euler(3,1,CV_64F);

    double m00 = rotationMatrix.at<double>(0,0);
    double m02 = rotationMatrix.at<double>(0,2);
    double m10 = rotationMatrix.at<double>(1,0);
    double m11 = rotationMatrix.at<double>(1,1);
    double m12 = rotationMatrix.at<double>(1,2);
    double m20 = rotationMatrix.at<double>(2,0);
    double m22 = rotationMatrix.at<double>(2,2);

    double bank, attitude, heading;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        bank = 0;
        attitude = CV_PI/2;
        heading = atan2(m02,m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        bank = 0;
        attitude = -CV_PI/2;
        heading = atan2(m02,m22);
    }
    else
    {
        bank = atan2(-m12,m11);
        attitude = asin(m10);
        heading = atan2(-m20,m00);
    }

    euler.at<double>(0) = bank;
    euler.at<double>(1) = attitude;
    euler.at<double>(2) = heading;

    return euler;
}

// Converts a given Euler angles to Rotation Matrix
cv::Mat euler2rot(const cv::Mat & euler)
{
    cv::Mat rotationMatrix(3,3,CV_64F);

    double bank = euler.at<double>(0);
    double attitude = euler.at<double>(1);
    double heading = euler.at<double>(2);

    // Assuming the angles are in radians.
    double ch = cos(heading);
    double sh = sin(heading);
    double ca = cos(attitude);
    double sa = sin(attitude);
    double cb = cos(bank);
    double sb = sin(bank);

    double m00, m01, m02, m10, m11, m12, m20, m21, m22;

    m00 = ch * ca;
    m01 = sh*sb - ch*sa*cb;
    m02 = ch*sa*sb + sh*cb;
    m10 = sa;
    m11 = ca*cb;
    m12 = -ca*sb;
    m20 = -sh*ca;
    m21 = sh*sa*cb + ch*sb;
    m22 = -sh*sa*sb + ch*cb;

    rotationMatrix.at<double>(0,0) = m00;
    rotationMatrix.at<double>(0,1) = m01;
    rotationMatrix.at<double>(0,2) = m02;
    rotationMatrix.at<double>(1,0) = m10;
    rotationMatrix.at<double>(1,1) = m11;
    rotationMatrix.at<double>(1,2) = m12;
    rotationMatrix.at<double>(2,0) = m20;
    rotationMatrix.at<double>(2,1) = m21;
    rotationMatrix.at<double>(2,2) = m22;

    return rotationMatrix;
}

cv::Mat Vec3b2Mat(cv::Vec3b in)
{
    cv::Mat mat(3,1, CV_64F);
    mat.at <double>(0,0) = in [0];
    mat.at <double>(1,0) = in [1];
    mat.at <double>(2,0) = in [2];

    return mat;
};

void fillMeasurements(cv::Mat &measurements,
                      const cv::Vec3d translation_measured,
                      const cv::Vec3d rvec) {
    
    // Set measurement to predict
    measurements.at<double>(0) = translation_measured [0]; // x
    measurements.at<double>(1) = translation_measured [1]; // y
    measurements.at<double>(2) = translation_measured [2]; // z
    measurements.at<double>(3) = rvec [0];      // roll
    measurements.at<double>(4) = rvec [1];      // pitch
    measurements.at<double>(5) = rvec [2];      // yaw
}

void updateKalmanFilter(cv::KalmanFilter &KF,
                        cv::Mat &measurement,
                        cv::Mat &translation_estimated,
                        cv::Mat &rotation_estimated,
                        cv::Mat &speed_estimated) {

    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);

    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);

    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);

    // Estimated speed
    speed_estimated.at<double>(0) = estimated.at<double>(3);
    speed_estimated.at<double>(1) = estimated.at<double>(4);
    speed_estimated.at<double>(2) = estimated.at<double>(5);

    rotation_estimated = euler2rot(eulers_estimated);
}