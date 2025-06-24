<div align = "center">
  <h1>
     VINS-Explorer: A Super Tightly Coupled Visual-Inertial State Estimator
  </h1>
</div>

![VINS-Explorer.png](https://github.com/brucezhcw/VINS-Explorer/blob/develop/support_files/image/VINS-Explorer.png) 

本系统在 [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) 的基础上，针对视觉惯性里程计(**VIO**)的精度与鲁棒性，
不断地在探索算法与工程层面的优化方法，致力于视觉与惯性信息的深度融合

## 与传统 VINS 的区别：
### 1.传统方法：
IMU 惯性信息（以预积分形式）主要作用于后端优化，为视觉关键帧间提供位姿约束。前端特征跟踪则完全依赖纯视觉方法。
### 2.本系统：
首次将 IMU 惯性信息引入视觉前端。利用相邻视觉帧间高精度的 IMU 递推位姿，实现更早、更紧密的传感器融合。后端采用分步优化方法，
第一步快速优化：只优化滑窗内极少数视觉特征点，并将优化后残差异常的观测剔除掉；第二步联合优化：异常观测更少、规模更小，初值更准确的鲁棒优化

## 前端融合的优势：
### 1.提升特征跟踪鲁棒性：
利用 IMU 递推位姿插值获取当前帧的精确初始位姿估计，可将已有的 3D 地图点准确投影到当前帧,
极大增强光流跟踪的可靠性，尤其在高动态场景下效果显著。
#### 理论说明：
Mono 相机需先估计特征点深度，IMU 前端融合才能有效，高动态下有效点较少。
而 Stereo 相机能直接获得深度信息，每帧拥有更多有效 3D 点，因此 IMU 信息在前端能发挥更大作用，预期性能更优。
### 2.强化误匹配点剔除 ： 
IMU 为视觉关键帧提供了可靠的位姿初始估计(即帧间位姿变换已知)。视觉特征点的匹配关系必须符合该 IMU 提供的几何约束，从而更有效地剔除误匹配点。

## 后端融合的优势：
### 1.提升鲁棒性：
第一步优化的基本原理类似前端三视图几何校验，固定住已经优化过多次的较老视觉帧和从先验IMU递推位姿插值得到的最新视觉帧，
只优化特征点。优化后的视觉残差较大的观测即异常观测。
第一步优化本身结构简单可靠，其为第二步优化提供了更准确的初值并剔除了异常观测，增强了第二步优化的可靠性
### 2.加速 ：
第一步优化本身待优化参数很少，优化问题结构简单，可以使用更高效的Ceres优化策略。
第二步优化具有了更准确的初值、更小的优化问题规模，优化速度肯定更快

## 前端融合思想的普适性：
该前端融合思想（利用 IMU 提供高精度初始位姿估计）尤其适合自带深度信息的传感器，如 RGB-D 相机或激光雷达。
例如，在 LIO (LiDAR-Inertial Odometry) 中，相比于纯 LO (LiDAR Odometry)，高精度的 IMU 递推位姿能为 ICP/NDT 等点云配准算法提供
近乎真值的初始位姿和点云投影预测。这能显著减少迭代次数，降低陷入局部极小值或发散的风险，从而大幅提升雷达里程计的精度和鲁棒性。

## 可行性分析 (IMU 误差)：
IMU 递推位姿的误差确实随时间呈指数增长，且不同 IMU 性能差异较大。
然而，视觉帧间的时间间隔通常很短（约 50-100 ms）。在此短时间尺度内，即使是消费级 IMU 也能提供足够精确的位姿估计，其误差完全可接受。
因此，本系统提出的算法具有良好的实用性和落地潜力，适用于广泛的机器人应用场景。 

## 未来工作以及版本迭代记录

Mono VIO算法持续优化

尝试该算法向Stereo VIO方案迁移调优，推荐一个SOTA SVIO开源方案: [Voxel_SVIO](https://github.com/ZikangYuan/voxel_svio)

**2025年06月22日**： 
前后端联合调优

**2025年06月02日**： 
实现了后端两步优化方法 optimization_with_fixed_Pose() + optimization():
在optimization_with_fixed_Pose中，我们固定了pose信息，只允许优化特征点深度信息，并对优化后的残差进行校验。异常残差将不会存在于第二步优化中。
因为在所有状态信息中，pose velocity bias都是相对比较准确的，唯独depth具有较大不确定信息。所以如果首先固定其他状态，只优化depth，则：
要么depth优化到较为准确的状态，要么残差、状态异常，对于后者我们可以在正式联合优化中及早剔除，减少联合优化参数规模、避免异常观测带来的负面影响。
看起来是多加了一个优化步骤，实际上在第二步优化中因为整体参数规模的减小和较为准确的初值估计，实际迭代次数大大减少，两步总耗时实际是降低的。
理论上，两步优化法应该更快、更鲁棒。

**2025年05月18日**： 
实现了基于高精度的IMU递推位姿进行较为精细的outlier剔除方法(三视图几何校验功能) rejectWith_three_view(): 
利用已知的帧间位姿变换在第1和第3帧中逐点进行三角化求解3D点，将上述3D点逐点投影到第2帧图像，验证点-点重投影误差。
该方法是一种比较严格的outlier剔除方法，可以极大程度的剔除掉outlier, 当然也会牺牲一点点计算时间。
当三视图校验方法条件不满足时：沿用之前的双视图几何校验功能 rejectWith_two_view()，验证点-线重投影误差

**2025年05月05日**： 
rejectWith_predicted_Pose() 代替 rejectWithF(): 利用高精度的IMU递推位姿构造本质矩阵，基于对极几何约束剔除outlier。
可以节省一部分时间，也不再像cv::findFundamentalMat()那样需要最少点对数量达标才能运行。即使1对点对也可以验证对极约束

**2025年04月25日**： 
Debug; 
跟踪图像上独立显示绿色的3D点以及光流; 
实现了一个在光流跟踪过程中粗略的外点剔除策略; 
适当设定function_tolerance可以大大节省后端不必要的优化时间 

**2024年10月15日**： 
实现光流前、后双向跟踪逻辑 

**2024年10月13日**： 
实现了一个初步版本：基于IMU递推位姿进行已经完成深度估计的3D点在第二帧图像中的投影，将其作为初值用于光流跟踪 

## 测试demo和环境

https://www.bilibili.com/video/BV1T9NXzRE3p/

https://www.youtube.com/watch?v=bTT2PX7oCCw

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
