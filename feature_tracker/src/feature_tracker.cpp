#include "feature_tracker.h"

#define SQR(x) ((x)*(x))

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

float distance(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, const vector<uchar> &status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, const vector<uchar> &status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
    t0.z() = -999;
    t1.z() = -999;
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<pair<cv::Point2f, cv::Point2f>, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(make_pair(cur_pts[i], forw_pts[i]), ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<pair<cv::Point2f, cv::Point2f>, int>> &a, const pair<int, pair<pair<cv::Point2f, cv::Point2f>, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first.second) == 255)
        {
            cur_pts.push_back(it.second.first.first);
            forw_pts.push_back(it.second.first.second);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first.second, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, map<int, Vector3d> &id_points, const Eigen::Matrix3d &R2, const Eigen::Vector3d &t2, const double _cur_time)
{
    cv::Mat img;
    vector<size_t> index_3D;
    vector<cv::Point2f> pts_3D;
    
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %.2f ms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        float distance_3D = 0;
        vector<uchar> status;
        vector<float> err;
        forw_pts = cur_pts;
        if (id_points.size() > 0)
        { /* 先投影3D点 */
            Eigen::Vector2d local_uv;
            map<int, Vector3d>::iterator it;
            for (int i = 0; i < int(cur_pts.size()); i++)
            {
                it = id_points.find(ids[i]);
                if(it != id_points.end())
                {
                    m_camera->spaceToPlane(it->second, local_uv);
                    cv::Point2f uv_tmp(local_uv.x(), local_uv.y());
                    if (inBorder(uv_tmp))
                    {
                        forw_pts[i] = uv_tmp;
                        index_3D.push_back(i);
                        pts_3D.push_back(uv_tmp);
                        float dis = distance(cur_pts[i], forw_pts[i]);
                        if(dis > distance_3D) distance_3D = dis;
                    }
                }
            }
        }
        /* 然后跟踪所有点 */
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(11, 11), 3,
                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                cv::OPTFLOW_USE_INITIAL_FLOW);
        { /* 反向跟踪 */
            int n_before = 0, n_after =  0;
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = cur_pts;
            cv::calcOpticalFlowPyrLK(forw_img, cur_img, forw_pts, reverse_pts, reverse_status, err, cv::Size(11, 11), 1, 
                                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                    cv::OPTFLOW_USE_INITIAL_FLOW);
            for(size_t i = 0; i < status.size(); i++)
            {
                if(status[i]) {
                    n_before++;
                }
                if(status[i] && reverse_status[i] && distance(cur_pts[i], reverse_pts[i]) <= 0.5 && inBorder(forw_pts[i])) {
                    n_after++;
                } else {
                    status[i] = 0;
                }
            }
            ROS_INFO("tracking: %d %d", n_before, n_after);
        }

        int tracked_3D = 0, tracked_good_3D = 0;
        for (size_t i = 0; i < index_3D.size(); i++)
            if(status[index_3D[i]])
            {
                tracked_3D++;
                if(distance(forw_pts[index_3D[i]], pts_3D[i]) <= 5)
                    tracked_good_3D++;
            }
        ROS_INFO("3D point after tracking: %d, %d, %d", tracked_good_3D, tracked_3D, int(index_3D.size()));
        if (PUB_THIS_FRAME)
        { /* 3D点 index重映射 */
            int status_0 = 0;
            for(size_t i = 0, j = 0; index_3D.size()>0 && i < forw_pts.size(); i++)
            {
                if(status[i] == 0)
                {
                    status_0++;
                    if(index_3D[j] == i)
                    {
                        index_3D.erase(index_3D.begin() + j);
                        pts_3D.erase(pts_3D.begin() + j);
                    }
                }
                else
                {
                    if(index_3D[j] == i)
                    {
                        index_3D[j] -= status_0;
                        j++;
                        if(j >= index_3D.size())
                            break;
                    }
                }
            }
        }
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_INFO("temporal optical flow costs: %.2f ms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        if(t1.z() < -998.9 || t2.z() < -998.9 || cur_time-prev_time>1.5/FREQ || (t2 - t1).norm() <= 0.02)
            rejectWithF(index_3D, pts_3D);
        else
            rejectWith_predicted_Pose(R2, t2, index_3D, pts_3D);

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %.2f ms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %.2f ms", t_t.toc());

        prev_ids = ids;
        addPoints();
    }
    else
    {
        prev_ids = ids;
    }

    cur_img = forw_img;
    prev_pts = cur_pts;
    cur_pts = forw_pts;
    undistortedPoints();
    pprev_time = prev_time;
    prev_time = cur_time;
    R0 = R1;
    t0 = t1;
    R1 = R2;
    t1 = t2;
}

void FeatureTracker::rejectWithF(const std::vector<size_t> &index_3D, const std::vector<cv::Point2f> &pts_3D)
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        int tracked_3D = 0, tracked_good_3D = 0;
        for (size_t i = 0; i < index_3D.size(); i++)
            if(status[index_3D[i]])
            {
                tracked_3D++;
                if(distance(forw_pts[index_3D[i]], pts_3D[i]) <= 5)
                    tracked_good_3D++;
            }
        ROS_DEBUG("3D point after FM ransac: %d, %d, %d", tracked_good_3D, tracked_3D, int(index_3D.size()));
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu", size_a, forw_pts.size());
        ROS_DEBUG("FM ransac costs: %.2f ms", t_f.toc());
    }
}

void FeatureTracker::rejectWith_predicted_Pose(const Eigen::Matrix3d &R2, const Eigen::Vector3d &t2, const std::vector<size_t> &index_3D, const std::vector<cv::Point2f> &pts_3D)
{
    if (forw_pts.size() > 0)
    {
        ROS_DEBUG("reject by predicted Pose begins");
        TicToc t_f;
        int size_a = cur_pts.size();
        vector<uchar> status(size_a, 1);
        vector<Eigen::Vector3d> un_cur_p1s(size_a), un_forw_p2s(size_a);
        for (int i = 0; i < size_a; i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            un_cur_p1s[i] = tmp_p / tmp_p.z();

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            un_forw_p2s[i] = tmp_p / tmp_p.z();           
        }
        if(t0.z() > -998.9 && t1.z() > -998.9 && t2.z() > -998.9 && (t2 - t0).norm() > 0.02 &&
            cur_time-prev_time<1.5/FREQ && prev_time-pprev_time<1.5/FREQ) {
            /* 三视图几何校验(点-点重投影误差)
                1> 利用已知的帧间位姿变换在第1和第3帧中逐点进行三角化求解3D点
                2> 将上述3D点逐点投影到第2帧图像
                3> 验证点-点重投影误差 */
            int three_view_count = 0;
            vector<uchar> need_to_check;
            vector<Eigen::Vector3d> un_pre_p0s(size_a);
            for (int i = 0; i < size_a; i++)
            {
                int id = -1;
                for (size_t j = 0; j < prev_ids.size(); j++)
                {
                    if(prev_ids[j] == ids[i]) 
                    {
                        id = j;
                        break;
                    }
                }
                if(id < 0) {
                    status[i] = 0;
                    need_to_check.push_back(0);
                } else {
                    three_view_count++;
                    need_to_check.push_back(1);
                    Eigen::Vector3d tmp_p;
                    m_camera->liftProjective(Eigen::Vector2d(prev_pts[id].x, prev_pts[id].y), tmp_p);
                    un_pre_p0s[i] = tmp_p / tmp_p.z();
                }
            }
            for (size_t i = 0; i < index_3D.size(); i++)
            {
                if(need_to_check[index_3D[i]] && distance(forw_pts[index_3D[i]], pts_3D[i]) <= 5)
                    need_to_check[index_3D[i]] = 2; //> 标记3D点
            }
            ROS_DEBUG("three view count: %d", three_view_count);

            rejectWith_three_view(status, need_to_check, un_pre_p0s, un_cur_p1s, un_forw_p2s, t0, R0, t1, R1, t2, R2);
        } else {
            /* 双视图几何校验(点-线重投影误差)
                1> 利用已知的帧间位姿变换构造本质矩阵E
                2> 分别求解两帧图像中点投影到另一帧图像后与极线的距离
                3> 验证最大点-线重投影距离 */
            vector<uchar> flag_3D(size_a, 0);
            for (size_t i = 0; i < index_3D.size(); i++)
            {
                if(distance(forw_pts[index_3D[i]], pts_3D[i]) <= 5)
                    flag_3D[index_3D[i]] = 1; //> 标记3D点
            }

            rejectWith_two_view(status, flag_3D, un_cur_p1s, un_forw_p2s, t1, R1, t2, R2);
        }

        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("reject: %d -> %lu", size_a, forw_pts.size());
        ROS_DEBUG("reject costs: %.2f ms", t_f.toc());
    }
}

void FeatureTracker::rejectWith_two_view(vector<uchar> &status, const vector<uchar> &flag_3D, const vector<Eigen::Vector3d> &un_cur_p1s,
            const vector<Eigen::Vector3d> &un_forw_p2s, const Vector3d &t1, const Matrix3d &R1, const Vector3d &t2, const Matrix3d &R2)
{
    Eigen::Matrix3d R  = R2.transpose() * R1;
    Eigen::Vector3d t  = R2.transpose() * (t1 - t2);
    Eigen::Matrix3d R_ = R1.transpose() * R2;
    Eigen::Vector3d t_ = R1.transpose() * (t2 - t1);

    float dist1, dist2;
    int size_a = status.size();
    Eigen::Matrix3d skew_t;
    skew_t <<     0, -t.z(),  t.y(),
            t.z(),      0, -t.x(),
            -t.y(),  t.x(),      0;
    Eigen::Matrix3d E = skew_t * R;
    skew_t <<     0, -t_.z(),  t_.y(),
            t_.z(),      0, -t_.x(),
            -t_.y(),  t_.x(),      0;
    Eigen::Matrix3d E_ = skew_t * R_;
    int count_3D=0, count_2D=0;
    float ave_diff_3D=0, max_diff_3D=0, min_diff_3D=999;
    float ave_diff_2D=0, max_diff_2D=0, min_diff_2D=999;
    for (int i = 0; i < size_a; i++)
    {
        Eigen::Vector3d epipolar_l;
        epipolar_l = E * un_cur_p1s[i];
        dist1 = std::abs(epipolar_l[0] * un_forw_p2s[i].x() + epipolar_l[1] * un_forw_p2s[i].y() +epipolar_l[2]) /
                            std::sqrt(SQR(epipolar_l[0]) + SQR(epipolar_l[1]));
        epipolar_l = E_ * un_forw_p2s[i];
        dist2 = std::abs(epipolar_l[0] * un_cur_p1s[i].x() + epipolar_l[1] * un_cur_p1s[i].y() +epipolar_l[2]) /
                            std::sqrt(SQR(epipolar_l[0]) + SQR(epipolar_l[1]));
        float max_dis = std::max(dist1, dist2);
        if(max_dis > 0.003)
            status[i] = 0;
        else
            status[i] = 1;
        if(flag_3D[i]) {
            count_3D++;
            ave_diff_3D += max_dis;
            if(max_dis > max_diff_3D) max_diff_3D = max_dis;
            if(max_dis < min_diff_3D) min_diff_3D = max_dis;

        } else {
            count_2D++;
            ave_diff_2D += max_dis;
            if(max_dis > max_diff_2D) max_diff_2D = max_dis;
            if(max_dis < min_diff_2D) min_diff_2D = max_dis;
        }
    }

    if(count_3D > 0) ROS_DEBUG("reject with two view 3D: %f, %f, %f", ave_diff_3D/count_3D, max_diff_3D, min_diff_3D);
    if(count_2D > 0) ROS_DEBUG("reject with two view 2D: %f, %f, %f", ave_diff_2D/count_2D, max_diff_2D, min_diff_2D);
}

void FeatureTracker::rejectWith_three_view(vector<uchar> &status, const vector<uchar> &need_to_check,
    const vector<Eigen::Vector3d> &points_0, const vector<Eigen::Vector3d> &points_1, const vector<Eigen::Vector3d> &points_2,
    const Vector3d &t0, const Matrix3d &R0, const Vector3d &t1, const Matrix3d &R1, const Vector3d &t2, const Matrix3d &R2)
{
    Eigen::Matrix3d R  = R0.transpose() * R2;
    Eigen::Vector3d t  = R0.transpose() * (t2 - t0);
    Eigen::Matrix3d R_  = R0.transpose() * R1;
    Eigen::Vector3d t_  = R0.transpose() * (t1 - t0);

    int count_3D=0, count_2D=0, count_depth=0;
    float ave_diff_3D=0, max_diff_3D=0, min_diff_3D=999;
    float ave_diff_2D=0, max_diff_2D=0, min_diff_2D=999;
    float ave_depth=0, max_depth=0, min_depth=9999;
    for (unsigned int i = 0; i < points_2.size(); i++)
    { /* 第0 2帧求解深度, 第1帧校验 */
        if(status[i] == 0 || need_to_check[i] == 0)
            continue;

        Eigen::Vector3d f;
        Eigen::Matrix<double, 3, 4> P;
        Eigen::MatrixXd svd_A(2 * 2, 4);

        P.leftCols<3>() = Eigen::Matrix3d::Identity();
        P.rightCols<1>() = Eigen::Vector3d::Zero();
        f = points_0[i].normalized();
        svd_A.row(0) = f[0] * P.row(2) - f[2] * P.row(0);
        svd_A.row(1) = f[1] * P.row(2) - f[2] * P.row(1);

        P.leftCols<3>() = R.transpose();
        P.rightCols<1>() = -R.transpose() * t;
        f = points_2[i].normalized();
        svd_A.row(2) = f[0] * P.row(2) - f[2] * P.row(0);
        svd_A.row(3) = f[1] * P.row(2) - f[2] * P.row(1);
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        
        double svd_dep = svd_V[2] / svd_V[3];
        Eigen::Vector3d points_3D = Vector3d(points_0[i].x(), points_0[i].y(), 1.0) * svd_dep;
        Eigen::Vector3d prdicted_p1 = R_.transpose()  * (points_3D - t_);
        prdicted_p1 /= prdicted_p1.z();
        double diff = (prdicted_p1 - points_1[i]).norm();

        if(svd_dep < 0.1 || svd_dep>500) {
            status[i] = 0;
            continue;
        }
        if(need_to_check[i] == 2) {
            count_3D++;
            ave_diff_3D += diff;
            if(diff > max_diff_3D) max_diff_3D = diff;
            if(diff < min_diff_3D) min_diff_3D = diff;

        } else {
            count_2D++;
            ave_diff_2D += diff;
            if(diff > max_diff_2D) max_diff_2D = diff;
            if(diff < min_diff_2D) min_diff_2D = diff;

            float thres;

            if(svd_dep < 50) thres = 0.005;
            else if(svd_dep < 100) thres = 0.004;
            else if(svd_dep < 200) thres = 0.003;
            else if(svd_dep < 300) thres = 0.002;
            else thres = 0.001;

            if(diff > thres) status[i] = 0;
        }
        if(status[i]) {
            count_depth++;
            ave_depth += svd_dep;
            if(svd_dep > max_depth) max_depth = svd_dep;
            if(svd_dep < min_depth) min_depth = svd_dep;
        }
    }

    if(count_3D > 0) ROS_DEBUG("reject with three view 3D: %f, %f, %f", ave_diff_3D/count_3D, max_diff_3D, min_diff_3D);
    if(count_2D > 0) ROS_DEBUG("reject with three view 2D: %f, %f, %f", ave_diff_2D/count_2D, max_diff_2D, min_diff_2D);
    if(count_depth > 0) ROS_DEBUG("reject with three view depth: %f, %f, %f", ave_depth/count_depth, max_depth, min_depth);
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
