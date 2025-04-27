# VINS-Explorer
## A Super Tightly Coupled Visual-Inertial State Estimator

基于 [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono), 进行一些算法、工程上的精度、鲁棒性等指标的优化方法探索，  
偏重于视觉信息和惯性信息的深度融合**VIO** 

## 核心算法

相较于传统的VINS系统：IMU惯性信息以预积分的形式作用于后端，在优化过程中提供视觉关键帧之间的一个位姿约束功能， 
前端还是依赖纯粹的视觉方法完成特征点的跟踪 

本系统：首次将IMU惯性信息(相邻视觉帧之间高精度IMU递推位姿)应用于视觉前端，更进一步的融合了视觉传感器和IMU惯性传感器。 
从理论上讲，**Mono**相机需要首先完成特征点的深度估计，IMU惯性信息才能起作用，动态场景中每帧图像一般少部分点有效。 
然而，**Stereo**相机可以直接获得特征点深度信息，每一帧都会有更多的3D点，相应的IMU惯性信息可以更加充分的应用到前端跟踪，性能估计更优。 

**by the way**，IMU提供的初始估计对于像**RGBD**相机、**激光雷达**这些全体点云都自带深度信息的传感器最适合不过了。 
**LIO**相对于**LO**，高精度的IMU递推位姿可以为**ICP**、**NDT**提供一个近似于真值的**位姿以及点云投影**， 
可以极大的减少点云配准算法的迭代次数和陷入局部极小或者发散的概率，能极大提升雷达里程计算法精度和鲁棒性。 

IMU递推位姿的误差随时间呈现指数形式增加，不同的IMU型号误差增速差异巨大。 
但是短时间内的误差都会很小，视觉帧之间的时间差也就50 100ms样子，像这种时间差，IMU都可以提供非常精准的位姿估计。 
甚至于消费级IMU在这种时间差内的误差都完全可以接受。因此该算法可以直接在大多数机器人应用上落地。 
目前的核心工作都是为KLT光流跟踪算法库函数cv::calcOpticalFlowPyrLK()提供特征点在第二帧图像中的坐标，但是这么做并不彻底， 
因为高精度的IMU递推位姿以及已有3D点这些信息其实是可以做类似 rejectWithF() 的操作，不再需要RANSAC

**2025年04月25日**： 
Debug; 
跟踪图像上独立显示绿色的3D点以及光流; 
实现了一个在光流跟踪过程中粗略的外点剔除策略; 
适当设定function_tolerance可以大大节省后端不必要的优化时间 

**2024年10月15日**： 
实现光流前、后双向跟踪逻辑 

**2024年10月13日**： 
实现了一个初步版本：基于IMU递推位姿进行已经完成深度估计的3D点在第二帧图像中的投影，将其作为初值用于光流跟踪 

## 未来工作

Mono VIO算法持续优化：前端跟踪，有高精度IMU递推位姿、若干3D点投影前提下，rejectWithF() 的逻辑应当考虑利用高精度的位姿先验和3D点先验，不再是RANSAC算法 
Mono VIO算法向Stereo VIO、线特征算法迁移测试 

## 测试环境

i7-10870H CPU@2.20GHZ 
Windows11 WSL2 
Ubuntu 20.04.6LTS 
ROS-noetic 
Ceres1.14.0  OpenCV4.2.0  Eigen3.3.7 

run example
```
    source ~/catkin_ws/devel/setup.bash
    roslaunch vins_estimator tum.launch 
    roslaunch vins_estimator vins_rviz.launch
    rosbag play YOUR_PATH_TO_DATASET/dataset-slides3_512_16.bag 
```

**What does VINS-Explorer do based on VINS-Mono?** 

![VINS-Explorer.png](https://github.com/brucezhcw/VINS-Explorer/blob/develop/support_files/image/VINS-Explorer.png) 

