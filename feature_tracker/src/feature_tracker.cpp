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

float distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
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

void FeatureTracker::readImage(const cv::Mat &_img, map<int, Vector3d> &id_points, Eigen::Matrix3d R1, Eigen::Vector3d t1, double _cur_time)
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
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
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
            int n_good = 0, n_bad=  0;
            float err_limit = 30.0;
            vector<float> sorted_err;
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = cur_pts;
            cv::calcOpticalFlowPyrLK(forw_img, cur_img, forw_pts, reverse_pts, reverse_status, err, cv::Size(11, 11), 1, 
                                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                    cv::OPTFLOW_USE_INITIAL_FLOW);
            for(size_t i = 0; i < status.size(); i++)
            {
                if(status[i] && reverse_status[i] && distance(cur_pts[i], reverse_pts[i]) <= 0.5)
                {
                    sorted_err.push_back(distance(cur_pts[i], forw_pts[i]));
                }
            }
            if(sorted_err.size() > 2)
            {
                std::sort(sorted_err.begin(), sorted_err.end());
                err_limit = sorted_err[sorted_err.size()/2] * 2.0;
            }
            err_limit = err_limit < 0.5 ? 0.5 : err_limit;
            err_limit = err_limit < distance_3D ? distance_3D + 0.1 : err_limit;
            for(size_t i = 0; i < status.size(); i++)
            {
                if(status[i] && reverse_status[i] && distance(cur_pts[i], reverse_pts[i]) <= 0.5)
                {
                    if(distance(cur_pts[i], forw_pts[i]) <= err_limit) {
                        n_good++;
                        status[i] = 1;
                    } else {
                        n_bad++;
                        status[i] = 0;
                    }
                }
                else
                    status[i] = 0;
            }
            ROS_INFO("tracking: %5.2f %5.2f, %d %d", err_limit, distance_3D, n_good, n_bad);
        }
        for (size_t i = 0; i < forw_pts.size(); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        int tracked_3D = 0, tracked_good_3D = 0;
        for (size_t i = 0; i < index_3D.size(); i++)
            if(status[index_3D[i]])
            {
                tracked_3D++;
                if(distance(forw_pts[index_3D[i]], pts_3D[i]) <= 5)
                    tracked_good_3D++;
            }
        ROS_INFO("3D point: %d, %d, %d", tracked_good_3D, tracked_3D, int(index_3D.size()));
        if (PUB_THIS_FRAME)
        {
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
        ROS_INFO("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        if(t1.z() < -998.9 || t0.z() < -998.9 || cur_time-prev_time>1.5/FREQ)
            rejectWithF(index_3D, pts_3D);
        else
            rejectWith_predicted_Pose(R1, t1, index_3D, pts_3D);

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

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
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    cur_img = forw_img;
    prev_pts = cur_pts;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    R0 = R1;
    t0 = t1;
}

void FeatureTracker::rejectWithF(std::vector<size_t> index_3D, std::vector<cv::Point2f> pts_3D)
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
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::rejectWith_predicted_Pose(Eigen::Matrix3d R1, Eigen::Vector3d t1, std::vector<size_t> index_3D, std::vector<cv::Point2f> pts_3D)
{
    Eigen::Matrix3d R  = R1.transpose() * R0;
    Eigen::Vector3d t  = R1.transpose() * (t0 - t1);
    Eigen::Matrix3d R_ = R0.transpose() * R1;
    Eigen::Vector3d t_ = R0.transpose() * (t1 - t0);

    if (forw_pts.size() > 0)
    {
        ROS_DEBUG("reject by predicted Pose begins");
        TicToc t_f;
        vector<Eigen::Vector3d> un_cur_p1s(cur_pts.size()), un_forw_p2s(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            un_cur_p1s[i] = tmp_p / tmp_p.z();

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            un_forw_p2s[i] = tmp_p / tmp_p.z();           
        }

        float dist1, dist2;
        vector<uchar> status;
        vector<float> distances;
        Eigen::Matrix3d skew_t;
        skew_t <<     0, -t.z(),  t.y(),
                  t.z(),      0, -t.x(),
                 -t.y(),  t.x(),      0;
        Eigen::Matrix3d E = skew_t * R;
        skew_t <<     0, -t_.z(),  t_.y(),
                  t_.z(),      0, -t_.x(),
                 -t_.y(),  t_.x(),      0;
        Eigen::Matrix3d E_ = skew_t * R_;
        for (unsigned int i = 0; i < cur_pts.size(); i++)
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
                status.push_back(0);
            else
                status.push_back(1);
            distances.push_back(max_dis);
        }

        int count = 0;
        float dis_ave = 0.0, dis_min = 999, dis_max = 0;
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            if(status[i])
            {
                count++;
                dis_ave += distances[i];
                if(distances[i]>dis_max) dis_max = distances[i];
                if(distances[i]<dis_min) dis_min = distances[i];
            }
        }
        if(count>0) ROS_DEBUG("point tracked dis: %f, %f, %f", dis_ave/count, dis_max, dis_min);
        count = 0; dis_ave = 0.0; dis_min = 999; dis_max = 0;
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            if(status[i]==0)
            {
                count++;
                dis_ave += distances[i];
                if(distances[i]>dis_max) dis_max = distances[i];
                if(distances[i]<dis_min) dis_min = distances[i];
            }
        }
        if(count>0) ROS_DEBUG("point not tracked dis: %f, %f, %f", dis_ave/count, dis_max, dis_min);

        float dis_3D = 0, max_3D = 0, min_3D = 999;
        int size_a = cur_pts.size();
        int tracked_3D = 0, tracked_good_3D = 0;
        for (size_t i = 0; i < index_3D.size(); i++)
            if(status[index_3D[i]])
            {
                tracked_3D++;
                if(distance(forw_pts[index_3D[i]], pts_3D[i]) <= 5)
                    tracked_good_3D++;
                dis_3D += distances[index_3D[i]];
                if(distances[index_3D[i]]>max_3D) max_3D = distances[index_3D[i]];
                if(distances[index_3D[i]]<min_3D) min_3D = distances[index_3D[i]];
            }
        if(tracked_3D>0) ROS_DEBUG("3D point dis: %f, %f, %f", dis_3D/tracked_3D, max_3D, min_3D);
        ROS_DEBUG("3D point after reject: %d, %d, %d", tracked_good_3D, tracked_3D, int(index_3D.size()));
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("reject: %d -> %lu", size_a, forw_pts.size());
        ROS_DEBUG("reject costs: %fms", t_f.toc());
    }
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
