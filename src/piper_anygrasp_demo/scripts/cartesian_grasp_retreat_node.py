#!/usr/bin/env python3

import copy
import math
import sys

import rospy
import moveit_commander


ARM_GROUP = "arm"


def normalize_quaternion(orientation):
    values = [
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]

    norm = math.sqrt(
        sum(value * value for value in values)
    )

    if norm < 1e-8:
        raise ValueError("Quaternion norm is zero.")

    return [
        value / norm
        for value in values
    ]


def local_positive_z_axis(quaternion):
    qx, qy, qz, qw = quaternion

    axis = [
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ]

    axis_norm = math.sqrt(
        sum(value * value for value in axis)
    )

    if axis_norm < 1e-8:
        raise ValueError("Computed local +Z axis is invalid.")

    return [
        value / axis_norm
        for value in axis
    ]


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("cartesian_grasp_retreat_node")

    retreat_distance = float(
        rospy.get_param("~retreat_distance", 0.01)
    )
    eef_step = float(
        rospy.get_param("~eef_step", 0.001)
    )
    velocity = float(
        rospy.get_param("~velocity", 0.05)
    )
    acceleration = float(
        rospy.get_param("~acceleration", 0.05)
    )

    if not 0.0 < retreat_distance <= 0.20:
        rospy.logerr(
            "~retreat_distance must be in (0, 0.20]."
        )
        return

    if not 0.0 < eef_step <= 0.01:
        rospy.logerr(
            "~eef_step must be in (0, 0.01]."
        )
        return

    try:
        robot = moveit_commander.RobotCommander()

        arm = moveit_commander.MoveGroupCommander(
            ARM_GROUP
        )

        rospy.sleep(1.0)

        planning_frame = arm.get_planning_frame()
        end_effector_link = arm.get_end_effector_link()

        current_pose = arm.get_current_pose(
            end_effector_link
        ).pose

        quaternion = normalize_quaternion(
            current_pose.orientation
        )

        approach_axis = local_positive_z_axis(
            quaternion
        )

        # 撤退方向与接近方向相反，即沿局部 -Z。
        retreat_pose = copy.deepcopy(current_pose)

        retreat_pose.position.x -= (
            retreat_distance * approach_axis[0]
        )
        retreat_pose.position.y -= (
            retreat_distance * approach_axis[1]
        )
        retreat_pose.position.z -= (
            retreat_distance * approach_axis[2]
        )

        retreat_pose.orientation.x = quaternion[0]
        retreat_pose.orientation.y = quaternion[1]
        retreat_pose.orientation.z = quaternion[2]
        retreat_pose.orientation.w = quaternion[3]

        rospy.loginfo(
            "===== Cartesian grasp retreat ====="
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
            "Current grasp position: "
            "[%.6f, %.6f, %.6f]",
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        )

        rospy.loginfo(
            "Local +Z approach axis: "
            "[%.6f, %.6f, %.6f]",
            approach_axis[0],
            approach_axis[1],
            approach_axis[2],
        )

        rospy.loginfo(
            "Retreat distance along local -Z: %.6f m",
            retreat_distance,
        )

        rospy.loginfo(
            "Retreat target position: "
            "[%.6f, %.6f, %.6f]",
            retreat_pose.position.x,
            retreat_pose.position.y,
            retreat_pose.position.z,
        )

        arm.set_start_state_to_current_state()

        plan, fraction = arm.compute_cartesian_path(
            [retreat_pose],
            eef_step,
            True,
        )

        rospy.loginfo(
            "Cartesian retreat path fraction: %.3f",
            fraction,
        )

        if fraction < 0.999:
            rospy.logerr(
                "Cartesian retreat path is incomplete: "
                "fraction=%.3f",
                fraction,
            )
            return

        if not plan.joint_trajectory.points:
            rospy.logerr(
                "Cartesian retreat trajectory contains no points."
            )
            return

        retimed_plan = arm.retime_trajectory(
            robot.get_current_state(),
            plan,
            velocity,
            acceleration,
        )

        rospy.loginfo(
            "Executing Cartesian retreat..."
        )

        success = arm.execute(
            retimed_plan,
            wait=True,
        )

        arm.stop()

        if not success:
            rospy.logerr(
                "Cartesian retreat execution failed."
            )
            return

        rospy.sleep(0.3)

        final_pose = arm.get_current_pose(
            end_effector_link
        ).pose

        rospy.loginfo(
            "Final position: [%.6f, %.6f, %.6f]",
            final_pose.position.x,
            final_pose.position.y,
            final_pose.position.z,
        )

        rospy.loginfo(
            "Cartesian grasp retreat completed successfully."
        )

    except Exception as error:
        rospy.logerr(
            "Cartesian retreat failed: %s",
            str(error),
        )

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
