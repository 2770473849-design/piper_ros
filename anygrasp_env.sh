#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source "$HOME/piper_ros/devel/setup.bash"

export ROS_PACKAGE_PATH="$HOME/gazebo_link_attacher_ws/src${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"
export PYTHONPATH="$HOME/gazebo_link_attacher_ws/devel/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/gazebo_link_attacher_ws/devel/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GAZEBO_PLUGIN_PATH="$HOME/gazebo_link_attacher_ws/devel/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"
