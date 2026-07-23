#!/usr/bin/env python3

import sys

import rospy
import moveit_commander

from gazebo_msgs.srv import (
    DeleteModel,
    DeleteModelRequest,
    SpawnModel,
    SpawnModelRequest,
)
from geometry_msgs.msg import Pose, PoseStamped


MODEL_NAME = "anygrasp_test_cube"

GAZEBO_SPAWN_SERVICE = "/gazebo/spawn_sdf_model"
GAZEBO_DELETE_SERVICE = "/gazebo/delete_model"

CUBE_SIZE = 0.03
CUBE_MASS = 0.05

# 立方体绕任一中心轴的转动惯量：
# I = (1 / 6) * m * side^2
CUBE_INERTIA = (
    CUBE_MASS * CUBE_SIZE * CUBE_SIZE / 6.0
)


def make_cube_sdf():
    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>

    <link name="cube_link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia}</ixx>
          <iyy>{inertia}</iyy>
          <izz>{inertia}</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>{size} {size} {size}</size>
          </box>
        </geometry>

        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>{size} {size} {size}</size>
          </box>
        </geometry>

        <material>
          <ambient>0.8 0.1 0.1 1.0</ambient>
          <diffuse>0.8 0.1 0.1 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""".format(
        name=MODEL_NAME,
        mass=CUBE_MASS,
        inertia=CUBE_INERTIA,
        size=CUBE_SIZE,
    )


def main():
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node("spawn_test_cube_node")

    cube_x = float(rospy.get_param("~x", 0.25))
    cube_y = float(rospy.get_param("~y", 0.0))
    cube_z = float(rospy.get_param("~z", 0.015))

    planning_frame = rospy.get_param(
        "~planning_frame",
        "dummy_link",
    )

    rospy.loginfo(
        "Waiting for Gazebo model services..."
    )

    try:
        rospy.wait_for_service(
            GAZEBO_SPAWN_SERVICE,
            timeout=10.0,
        )
        rospy.wait_for_service(
            GAZEBO_DELETE_SERVICE,
            timeout=10.0,
        )
    except rospy.ROSException as error:
        rospy.logerr(
            "Gazebo model service unavailable: %s",
            str(error),
        )
        return

    spawn_model = rospy.ServiceProxy(
        GAZEBO_SPAWN_SERVICE,
        SpawnModel,
    )

    delete_model = rospy.ServiceProxy(
        GAZEBO_DELETE_SERVICE,
        DeleteModel,
    )

    # 若同名方块已存在，先删除。
    try:
        delete_request = DeleteModelRequest()
        delete_request.model_name = MODEL_NAME
        delete_model(delete_request)
        rospy.sleep(0.3)
    except rospy.ServiceException:
        pass

    gazebo_pose = Pose()
    gazebo_pose.position.x = cube_x
    gazebo_pose.position.y = cube_y
    gazebo_pose.position.z = cube_z
    gazebo_pose.orientation.w = 1.0

    spawn_request = SpawnModelRequest()
    spawn_request.model_name = MODEL_NAME
    spawn_request.model_xml = make_cube_sdf()
    spawn_request.robot_namespace = ""
    spawn_request.initial_pose = gazebo_pose
    spawn_request.reference_frame = "world"

    rospy.loginfo(
        "Spawning Gazebo cube at world: "
        "[%.4f, %.4f, %.4f]",
        cube_x,
        cube_y,
        cube_z,
    )

    try:
        spawn_response = spawn_model(
            spawn_request
        )
    except rospy.ServiceException as error:
        rospy.logerr(
            "Gazebo spawn service failed: %s",
            str(error),
        )
        return

    if not spawn_response.success:
        rospy.logerr(
            "Gazebo rejected cube: %s",
            spawn_response.status_message,
        )
        return

    rospy.loginfo(
        "Gazebo cube spawned successfully."
    )

    # 在 MoveIt Planning Scene 中加入同位置碰撞方块。
    scene = moveit_commander.PlanningSceneInterface(
        synchronous=True
    )

    rospy.sleep(1.0)

    moveit_pose = PoseStamped()
    moveit_pose.header.stamp = rospy.Time.now()
    moveit_pose.header.frame_id = planning_frame

    moveit_pose.pose.position.x = cube_x
    moveit_pose.pose.position.y = cube_y
    moveit_pose.pose.position.z = cube_z
    moveit_pose.pose.orientation.w = 1.0

    scene.remove_world_object(
        MODEL_NAME
    )

    rospy.sleep(0.3)

    scene.add_box(
        MODEL_NAME,
        moveit_pose,
        size=(
            CUBE_SIZE,
            CUBE_SIZE,
            CUBE_SIZE,
        ),
    )

    rospy.sleep(1.0)

    known_objects = (
        scene.get_known_object_names()
    )

    if MODEL_NAME not in known_objects:
        rospy.logerr(
            "Cube was not added to MoveIt Planning Scene."
        )
        return

    rospy.loginfo(
        "MoveIt collision cube added in frame '%s'.",
        planning_frame,
    )

    rospy.loginfo(
        "Cube size: %.3f × %.3f × %.3f m",
        CUBE_SIZE,
        CUBE_SIZE,
        CUBE_SIZE,
    )

    rospy.loginfo(
        "Test cube synchronization completed successfully."
    )

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
