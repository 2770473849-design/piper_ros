#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import moveit_commander
import rospy

from geometry_msgs.msg import PoseStamped


def wait_for_object(scene, object_name, expected_known, timeout=5.0):
    deadline = time.time() + timeout

    while not rospy.is_shutdown() and time.time() < deadline:
        known = object_name in scene.get_known_object_names()

        if known == expected_known:
            return True

        rospy.sleep(0.1)

    return False


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("safe_real_table_collision")

    frame_id = rospy.get_param("~frame_id", "arm_base")
    object_name = rospy.get_param(
        "~object_name",
        "safe_real_worktable",
    )

    # Cropped forward working area. It deliberately excludes
    # the robot base and the 10 mm mounting plate region.
    center_x = float(rospy.get_param("~center_x", 0.325))
    center_y = float(rospy.get_param("~center_y", 0.265))

    size_x = float(rospy.get_param("~size_x", 0.450))
    size_y = float(rospy.get_param("~size_y", 0.370))

    table_top_z = float(
        rospy.get_param("~table_top_z", -0.018)
    )
    thickness = float(
        rospy.get_param("~thickness", 0.040)
    )

    if size_x <= 0.0 or size_y <= 0.0 or thickness <= 0.0:
        raise RuntimeError(
            "Collision-box dimensions must be positive."
        )

    scene = moveit_commander.PlanningSceneInterface()

    rospy.loginfo(
        "Waiting for MoveIt planning-scene connection..."
    )
    rospy.sleep(1.5)

    # Remove an older instance before adding the updated one.
    scene.remove_world_object(object_name)
    wait_for_object(
        scene,
        object_name,
        expected_known=False,
        timeout=3.0,
    )

    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now()

    pose.pose.position.x = center_x
    pose.pose.position.y = center_y

    # add_box uses the box centre, not its upper surface.
    pose.pose.position.z = (
        table_top_z - thickness / 2.0
    )

    pose.pose.orientation.w = 1.0

    scene.add_box(
        object_name,
        pose,
        size=(size_x, size_y, thickness),
    )

    if not wait_for_object(
        scene,
        object_name,
        expected_known=True,
        timeout=5.0,
    ):
        raise RuntimeError(
            "MoveIt did not confirm collision object '{}'."
            .format(object_name)
        )

    rospy.loginfo(
        "Table collision object confirmed: %s",
        object_name,
    )
    rospy.loginfo(
        "Frame=%s, x=[%.3f, %.3f], y=[%.3f, %.3f], "
        "top_z=%.4f m",
        frame_id,
        center_x - size_x / 2.0,
        center_x + size_x / 2.0,
        center_y - size_y / 2.0,
        center_y + size_y / 2.0,
        table_top_z,
    )
    rospy.loginfo(
        "This node only publishes planning-scene geometry. "
        "No robot command is sent."
    )

    rospy.spin()


if __name__ == "__main__":
    main()
