#!/usr/bin/env python3

import sys

import rospy
import moveit_commander


ARM_GROUP = "arm"


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("arm_pose_inspect_node")

    try:
        robot = moveit_commander.RobotCommander()
        arm_group = moveit_commander.MoveGroupCommander(
            ARM_GROUP
        )

        # 等待 MoveIt 接收到当前关节状态。
        rospy.sleep(1.0)

        group_names = robot.get_group_names()
        planning_frame = arm_group.get_planning_frame()
        end_effector_link = arm_group.get_end_effector_link()
        current_joints = arm_group.get_current_joint_values()
        current_pose = arm_group.get_current_pose(
            end_effector_link
        ).pose

        rospy.loginfo(
            "Available MoveIt groups: %s",
            group_names,
        )
        rospy.loginfo(
            "Planning frame: %s",
            planning_frame,
        )
        rospy.loginfo(
            "End-effector link: %s",
            end_effector_link,
        )
        rospy.loginfo(
            "Current arm joints: %s",
            [
                round(value, 6)
                for value in current_joints
            ],
        )

        rospy.loginfo(
            "Current position: "
            "x=%.6f, y=%.6f, z=%.6f",
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        )

        rospy.loginfo(
            "Current orientation: "
            "qx=%.6f, qy=%.6f, qz=%.6f, qw=%.6f",
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w,
        )

    except Exception as error:
        rospy.logerr(
            "Failed to inspect arm pose: %s",
            str(error),
        )

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
