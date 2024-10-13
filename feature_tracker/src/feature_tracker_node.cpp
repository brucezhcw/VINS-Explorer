#include <mutex>
#include <thread>
#include <condition_variable>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

std::mutex m_buf;
std::condition_variable con;
vector<sensor_msgs::PointCloud::ConstPtr> point_3D_buf;
vector<nav_msgs::Odometry::ConstPtr> imu_forward_buf;
queue<sensor_msgs::ImageConstPtr> img_buf;
map<int, Vector3d> id_points;

void point_3D_callback(const sensor_msgs::PointCloud::ConstPtr &point_3D_msg)
{
    m_buf.lock();
    point_3D_buf.push_back(point_3D_msg);
    m_buf.unlock();
}

void imu_forward_callback(const nav_msgs::Odometry::ConstPtr &forward_msg)
{
    m_buf.lock();
    imu_forward_buf.push_back(forward_msg);
    m_buf.unlock();
    con.notify_one();
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img_buf.push(img_msg);
    m_buf.unlock();
    con.notify_one();
}

sensor_msgs::ImageConstPtr get_oneimage()
{
    double valid_time = 1.0/FREQ*0.5;
    sensor_msgs::ImageConstPtr img_msg;

    if (imu_forward_buf.empty() || img_buf.empty())
        return nullptr;

    id_points.clear();
    if (imu_forward_buf.back()->header.stamp.toSec() >= img_buf.front()->header.stamp.toSec() &&
        imu_forward_buf.front()->header.stamp.toSec() <= img_buf.front()->header.stamp.toSec())
    { /* 有效的IMU递推位姿 */
        img_msg = img_buf.front();
        img_buf.pop();
        double img_time = img_msg->header.stamp.toSec();
        sensor_msgs::PointCloud::ConstPtr point_3D_msg = nullptr;
        for(unsigned int i = 0; i<point_3D_buf.size(); i++)
        {
            if(point_3D_buf[i]->header.stamp.toSec() < img_time)
                point_3D_msg = point_3D_buf[i];
            else
                break;
        }
        if (point_3D_msg != nullptr && point_3D_msg->header.stamp.toSec() + valid_time*5 >= img_time)
        {
            int point_count = point_3D_msg->points.size();
            if (point_count > 0)
            {
                int i = 0, j = imu_forward_buf.size() - 1;
                for( ; i+1 <= j; i++)
                {
                    if(imu_forward_buf[i+1]->header.stamp.toSec() > img_time)
                        break;
                }
                for( ; j-1 >= i; j--)
                {
                    if(imu_forward_buf[j-1]->header.stamp.toSec() < img_time)
                        break;
                }
                Eigen::Matrix<double, 15, 1> sqrt_cov;
                for (int i = 0; i < 15; i++)
                    sqrt_cov[i] = imu_forward_buf[i]->pose.covariance[i];
                int last_track_num = imu_forward_buf[i]->twist.twist.angular.x + 0.5;
                double latest_image_time = imu_forward_buf[i]->twist.twist.angular.y;
                int solver_flag = imu_forward_buf[i]->twist.twist.angular.z + 0.5;
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4);
                for (int i = 0; i < sqrt_cov.size(); ++i) {
                    oss << sqrt_cov(i) << " ";
                }
                ROS_INFO_STREAM("predicted IMU state sqrt_cov: " << oss.str());
                ROS_INFO("solver_flag: %d, latest_image_time: %.3f, point: %d, dt: %.3f", solver_flag, latest_image_time, last_track_num, imu_forward_buf[i]->header.stamp.toSec()-latest_image_time);
                if(solver_flag==1 && last_track_num>0 && fabs(img_time-latest_image_time)<2.0 && sqrt_cov.head<3>().maxCoeff()<0.01 &&
                                                                                                 sqrt_cov.segment<3>(6).maxCoeff()<0.01)
                {
                    double imu_time_i = imu_forward_buf[i]->header.stamp.toSec();
                    Vector3d vio_t_i(imu_forward_buf[i]->pose.pose.position.x, imu_forward_buf[i]->pose.pose.position.y, imu_forward_buf[i]->pose.pose.position.z);
                    Quaterniond vio_q_i;
                    vio_q_i.w() = imu_forward_buf[i]->pose.pose.orientation.w;
                    vio_q_i.x() = imu_forward_buf[i]->pose.pose.orientation.x;
                    vio_q_i.y() = imu_forward_buf[i]->pose.pose.orientation.y;
                    vio_q_i.z() = imu_forward_buf[i]->pose.pose.orientation.z;
                    if (i != j)
                    {
                        double imu_time_j = imu_forward_buf[j]->header.stamp.toSec();
                        Vector3d vio_t_j(imu_forward_buf[j]->pose.pose.position.x, imu_forward_buf[j]->pose.pose.position.y, imu_forward_buf[j]->pose.pose.position.z);
                        Quaterniond vio_q_j;
                        vio_q_j.w() = imu_forward_buf[j]->pose.pose.orientation.w;
                        vio_q_j.x() = imu_forward_buf[j]->pose.pose.orientation.x;
                        vio_q_j.y() = imu_forward_buf[j]->pose.pose.orientation.y;
                        vio_q_j.z() = imu_forward_buf[j]->pose.pose.orientation.z;

                        double t_ratio = (img_time - imu_time_i) / (imu_time_j - imu_time_i);
                        vio_t_i = vio_t_i *(1 - t_ratio) + vio_t_j * t_ratio;
                        vio_q_i = vio_q_i.slerp(t_ratio, vio_q_j);
                        vio_q_i.normalize();
                    }
                    Vector3d tic;
                    Quaterniond qic;
                    tic.x() = point_3D_msg->channels[point_count].values[0];
                    tic.y() = point_3D_msg->channels[point_count].values[1];
                    tic.z() = point_3D_msg->channels[point_count].values[2];
                    qic.w() = point_3D_msg->channels[point_count].values[3];
                    qic.x() = point_3D_msg->channels[point_count].values[4];
                    qic.y() = point_3D_msg->channels[point_count].values[5];
                    qic.z() = point_3D_msg->channels[point_count].values[6];
                    for(int i=0; i<point_count; i++)
                    {
                        int feature_id = point_3D_msg->channels[i].values[0] + 0.5;
                        double x = point_3D_msg->points[i].x;
                        double y = point_3D_msg->points[i].y;
                        double z = point_3D_msg->points[i].z;
                        Vector3d w_pts_i, pts_i;
                        w_pts_i << x, y, z;
                        pts_i = qic.toRotationMatrix().transpose() * (vio_q_i.toRotationMatrix().transpose() * (w_pts_i - vio_t_i) - tic);
                        id_points.insert(make_pair(feature_id, pts_i));
                    }
                    ROS_INFO("image time: %.3f 3D point time: %.3f count: %d", img_time, point_3D_msg->header.stamp.toSec(), point_count);
                    ROS_INFO("image_t: %.4f %.4f %.4f image_q: %.4f %.4f %.4f %.4f", vio_t_i.x(), vio_t_i.y(), vio_t_i.z(), vio_q_i.w(), vio_q_i.x(), vio_q_i.y(), vio_q_i.z());
                }
            }
        }

        for (auto it = imu_forward_buf.begin(); it != imu_forward_buf.end(); )
        {
            if (it->get()->header.stamp.toSec() < img_time - valid_time)
                it = imu_forward_buf.erase(it);
            else
                break;
        }
        for (auto it = point_3D_buf.begin(); it != point_3D_buf.end(); )
        {
            if (it->get()->header.stamp.toSec() < img_time - valid_time*5)
                it = point_3D_buf.erase(it);
            else
                break;
        }
        return img_msg;
    }
    else if (imu_forward_buf.front()->header.stamp.toSec() > img_buf.front()->header.stamp.toSec())
    { /* 不可能存在有效的IMU递推位姿 */
        img_msg = img_buf.front();
        img_buf.pop();
        for (auto it = point_3D_buf.begin(); it != point_3D_buf.end(); )
        {
            if (it->get()->header.stamp.toSec() < img_msg->header.stamp.toSec() - valid_time*5)
                it = point_3D_buf.erase(it);
            else
                break;
        }
        return img_msg;
    }
    else
        return nullptr;
}

void process()
{
    while (true)
    {
        sensor_msgs::ImageConstPtr img_msg;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (img_msg = get_oneimage()) != nullptr;
                 });
        lk.unlock();
        if(img_msg == nullptr)
            continue;

        double img_msg_time = img_msg->header.stamp.toSec();
        ROS_INFO("new image coming %.3f", img_msg_time);
        if(first_image_flag)
        {
            first_image_flag = false;
            first_image_time = img_msg_time;
            last_image_time = img_msg_time;
            continue;
        }
        // detect unstable camera stream
        if (img_msg_time - last_image_time > 1.0 || img_msg_time < last_image_time)
        {
            ROS_WARN("image discontinue! reset the feature tracker!");
            first_image_flag = true; 
            last_image_time = 0;
            pub_count = 1;
            std_msgs::Bool restart_flag;
            restart_flag.data = true;
            pub_restart.publish(restart_flag);
            continue;
        }
        last_image_time = img_msg_time;
        // frequency control
        if (round(1.0 * pub_count / (img_msg_time - first_image_time)) <= FREQ)
        {
            PUB_THIS_FRAME = true;
            // reset the frequency control
            if (abs(1.0 * pub_count / (img_msg_time - first_image_time) - FREQ) < 0.01 * FREQ)
            {
                first_image_time = img_msg_time;
                pub_count = 0;
            }
        }
        else
            PUB_THIS_FRAME = false;

        cv_bridge::CvImageConstPtr ptr;
        if (img_msg->encoding == "8UC1")
        {
            sensor_msgs::Image img;
            img.header = img_msg->header;
            img.height = img_msg->height;
            img.width = img_msg->width;
            img.is_bigendian = img_msg->is_bigendian;
            img.step = img_msg->step;
            img.data = img_msg->data;
            img.encoding = "mono8";
            ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        }
        else
            ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

        cv::Mat show_img = ptr->image;
        TicToc t_r;
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            ROS_DEBUG("processing camera %d", i);
            if (i != 1 || !STEREO_TRACK)
                trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), id_points, img_msg_time);
            else
            {
                if (EQUALIZE)
                {
                    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                    clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
                }
                else
                    trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
            }

#if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
        }

        for (unsigned int i = 0;; i++)
        {
            bool completed = false;
            for (int j = 0; j < NUM_OF_CAM; j++)
                if (j != 1 || !STEREO_TRACK)
                    completed |= trackerData[j].updateID(i);
            if (!completed)
                break;
        }

        if (PUB_THIS_FRAME)
        {
            pub_count++;
            sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 id_of_point;
            sensor_msgs::ChannelFloat32 u_of_point;
            sensor_msgs::ChannelFloat32 v_of_point;
            sensor_msgs::ChannelFloat32 velocity_x_of_point;
            sensor_msgs::ChannelFloat32 velocity_y_of_point;

            feature_points->header = img_msg->header;
            feature_points->header.frame_id = "world";

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                auto &un_pts = trackerData[i].cur_un_pts;
                auto &cur_pts = trackerData[i].cur_pts;
                auto &ids = trackerData[i].ids;
                auto &pts_velocity = trackerData[i].pts_velocity;
                for (unsigned int j = 0; j < ids.size(); j++)
                {
                    if (trackerData[i].track_cnt[j] > 1)
                    {
                        int p_id = ids[j];
                        geometry_msgs::Point32 p;
                        p.x = un_pts[j].x;
                        p.y = un_pts[j].y;
                        p.z = 1;

                        feature_points->points.push_back(p);
                        id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                        u_of_point.values.push_back(cur_pts[j].x);
                        v_of_point.values.push_back(cur_pts[j].y);
                        velocity_x_of_point.values.push_back(pts_velocity[j].x);
                        velocity_y_of_point.values.push_back(pts_velocity[j].y);
                    }
                }
            }
            feature_points->channels.push_back(id_of_point);
            feature_points->channels.push_back(u_of_point);
            feature_points->channels.push_back(v_of_point);
            feature_points->channels.push_back(velocity_x_of_point);
            feature_points->channels.push_back(velocity_y_of_point);
            ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
            // skip the first image; since no optical speed on frist image
            if (!init_pub)
            {
                init_pub = 1;
            }
            else
                pub_img.publish(feature_points);

            if (SHOW_TRACK)
            {
                ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
                //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
                cv::Mat stereo_img = ptr->image;

                for (int i = 0; i < NUM_OF_CAM; i++)
                {
                    cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                    cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                    for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                    {
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                        //draw speed line
                        /*
                        Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                        Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                        Vector3d tmp_prev_un_pts;
                        tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                        tmp_prev_un_pts.z() = 1;
                        Vector2d tmp_prev_uv;
                        trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                        cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                        */
                        //char name[10];
                        //sprintf(name, "%d", trackerData[i].ids[j]);
                        //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    }
                }
                //cv::imshow("vis", stereo_img);
                //cv::waitKey(5);
                pub_match.publish(ptr->toImageMsg());
            }
        }
        ROS_INFO("whole feature tracker processing costs: %f\n", t_r.toc());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    ros::Subscriber sub_point_3D = n.subscribe("/vins_estimator/point_3D", 100, point_3D_callback);
    ros::Subscriber sub_imu_forward = n.subscribe("/vins_estimator/imu_propagate", 1000, imu_forward_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?