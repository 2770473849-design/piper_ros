# Gazebo 连续抓放稳定化记录

日期：2026-07-23

## 环境

- Ubuntu 20.04.6 LTS
- ROS1 Noetic
- Gazebo Classic 11
- Piper + MoveIt
- gazebo_ros_link_attacher

## 已实现功能

Gazebo 中的红色物理方块可以完成接近、夹持、附着、抬升、运输、放置和解除约束，并可在不重启 Gazebo 的情况下连续执行多轮 Pick and Place。

## 重复抓放漂移的根因

`gazebo_ros_link_attacher` 在 detach 时只调用 `j.joint->Detach()`，旧 joint 仍保存在缓存中。第二次 attach 复用旧 joint 时，原代码只调用 `Attach()`，没有重新加载 joint 状态，因此物块发生异常漂移。

## 修复

文件：

`~/gazebo_link_attacher_ws/src/gazebo_ros_link_attacher/src/gazebo_ros_link_attacher.cpp`

复用 joint 时改为：

`j.joint->Attach(j.l1, j.l2);`

`j.joint->Load(j.l1, j.l2, ignition::math::Pose3d());`

随后在 `~/gazebo_link_attacher_ws` 中执行 `catkin_make`，编译成功。

## 方块重置稳定化

文件：

`~/piper_ros/src/piper_anygrasp_demo/scripts/spawn_test_cube_node.py`

方块已存在时调用 `/gazebo/set_model_state` 重置位置，不再 delete + respawn，避免 link-attacher 缓存旧的 `cube_link` 指针。

## 验证结果

- 第一次抓放正常；
- 不重启 Gazebo 的第二轮抓放正常；
- 使用重置脚本的第三轮抓放正常；
- Gazebo 红色物块与 RViz 绿色规划物块均正常；
- 当前支持连续多轮完整 Pick and Place。

## 额外环境故障

异常重启后曾出现 NVIDIA 525/535 驱动混装，导致 NVML 版本不匹配和 `X_GLXCreateContext` 失败。清理 525 并重新安装完整的 535 驱动后，Gazebo 渲染恢复。
