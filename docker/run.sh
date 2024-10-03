#!/bin/bash
trap : SIGTERM SIGINT

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 LAUNCH_FILE" >&2
  exit 1
fi

roscore &
ROSCORE_PID=$!
sleep 1

rviz -d ../config/vins_rviz_config.rviz &
RVIZ_PID=$!


docker run \
  -it \
  --rm \
  --net=host \
  -v /home/brucezhcw/catkin_ws:/persist/catkin_ws \
  --name BruceZhcw_VSLAM \
  ros:vins-mono \
  /bin/bash -c \
  "cd /persist/catkin_ws/; \
  catkin config \
        --env-cache \
        --extend /opt/ros/kinetic \
       --cmake-args \
         -DCMAKE_BUILD_TYPE=Release; \
     catkin build; \
     source devel/setup.bash; \
     roslaunch vins_estimator ${1}"

wait $ROSCORE_PID
wait $RVIZ_PID

if [[ $? -gt 128 ]]
then
    kill $ROSCORE_PID
    kill $RVIZ_PID
fi
