#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <vector>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, const vector<uchar> &status);
void reduceVector(vector<int> &v, const vector<uchar> &status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, map<int, Vector3d> &id_points, const Eigen::Matrix3d &R2, const Eigen::Vector3d &t2, const double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void reject_outlier(const Eigen::Matrix3d &R2, const Eigen::Vector3d &t2, const std::vector<size_t> &index_3D, const std::vector<cv::Point2f> &pts_3D);
    void rejectWithF(const std::vector<size_t> &index_3D, const std::vector<cv::Point2f> &pts_3D);
    void rejectWith_two_view(vector<uchar> &status, const vector<uchar> &flag_3D, const vector<Eigen::Vector3d> &un_cur_p1s,
                    const vector<Eigen::Vector3d> &un_forw_p2s, const Vector3d &t1, const Matrix3d &R1, const Vector3d &t2, const Matrix3d &R2);
    void rejectWith_three_view(vector<uchar> &status, const vector<uchar> &need_to_check,
                    const vector<Eigen::Vector3d> &points_0, const vector<Eigen::Vector3d> &points_1, const vector<Eigen::Vector3d> &points_2,
                    const Vector3d &t0, const Matrix3d &R0, const Vector3d &t1, const Matrix3d &R1, const Vector3d &t2, const Matrix3d &R2);

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> prev_ids, ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;
    double pprev_time;

    Eigen::Matrix3d R0,R1;
    Eigen::Vector3d t0,t1;

    static int n_id;
};
