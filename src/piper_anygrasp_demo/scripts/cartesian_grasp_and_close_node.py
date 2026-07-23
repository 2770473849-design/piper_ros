#!/usr/bin/env python3

import copy
import math
import sys

import rospy
import moveit_commander

from geometry_msgs.msg import PoseStamped
from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


ARM_GROUP = "arm"
GRIPPER_SERVICE = "/joint_moveit_ctrl_gripper"
DEFAULT_INPUT_TOPIC = (
    "/detected_grasp_pose_in_planning_frame"
)


class CartesianGraspApproach:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            DEFAULT_INPUT_TOPIC,
        )

        self.planning_frame = rospy.get_param(
            "~planning_frame",
            "dummy_link",
        )

        self.max_distance = float(
            rospy.get_param(
                "~max_distance",
                0.03,
            )
        )

        self.max_lateral_error = float(
            rospy.get_param(
                "~max_lateral_error",
                0.003,
            )
        )

        self.orientation_tolerance = float(
            rospy.get_param(
                "~orientation_tolerance",
                0.03,
            )
        )

        self.eef_step = float(
            rospy.get_param(
                "~eef_step",
                0.001,
            )
        )

        self.velocity = float(
            rospy.get_param(
                "~velocity",
                0.05,
            )
        )

        self.acceleration = float(
            rospy.get_param(
                "~acceleration",
                0.05,
            )
        )

        self.gripper_close = float(
            rospy.get_param(
                "~gripper_close",
                0.0,
            )
        )

        if not 0.0 < self.max_distance <= 0.20:
            raise ValueError(
                "~max_distance must be in (0, 0.20]."
            )

        if not 0.0 < self.max_lateral_error <= 0.05:
            raise ValueError(
                "~max_lateral_error must be in (0, 0.05]."
            )

        if not 0.0 < self.eef_step <= 0.01:
            raise ValueError(
                "~eef_step must be in (0, 0.01]."
            )

        if not 0.0 <= self.gripper_close <= 0.035:
            raise ValueError(
                "~gripper_close must be in [0, 0.035]."
            )

        self.robot = moveit_commander.RobotCommander()

        self.arm_group = (
            moveit_commander.MoveGroupCommander(
                ARM_GROUP
            )
        )

        self.end_effector_link = (
            self.arm_group.get_end_effector_link()
        )

        rospy.loginfo(
            "Waiting for gripper service: %s",
            GRIPPER_SERVICE,
        )

        rospy.wait_for_service(
            GRIPPER_SERVICE,
            timeout=10.0,
        )

        self.gripper_service = rospy.ServiceProxy(
            GRIPPER_SERVICE,
            JointMoveitCtrl,
        )

        rospy.sleep(1.0)

        self.executing = False
        self.completed = False

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for grasp target: %s",
            self.input_topic,
        )

        rospy.loginfo(
            "Planning frame: %s",
            self.arm_group.get_planning_frame(),
        )

        rospy.loginfo(
            "End-effector link: %s",
            self.end_effector_link,
        )

    @staticmethod
    def normalize_quaternion(orientation):
        values = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "quaternion contains NaN or infinity"
            )

        norm = math.sqrt(
            sum(value * value for value in values)
        )

        if norm < 1e-8:
            raise ValueError(
                "quaternion norm is zero"
            )

        return [
            value / norm
            for value in values
        ]

    @staticmethod
    def local_positive_z_axis(quaternion):
        qx, qy, qz, qw = quaternion

        axis = [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (
                qx * qx + qy * qy
            ),
        ]

        norm = math.sqrt(
            sum(value * value for value in axis)
        )

        return [
            value / norm
            for value in axis
        ]

    @staticmethod
    def quaternion_angle(
        first,
        second,
    ):
        dot = abs(
            sum(
                a * b
                for a, b in zip(first, second)
            )
        )

        dot = max(0.0, min(1.0, dot))

        return 2.0 * math.acos(dot)

    def pose_callback(self, message):
        if self.executing or self.completed:
            return

        self.executing = True

        try:
            frame_id = message.header.frame_id.strip()

            if frame_id != self.planning_frame:
                raise ValueError(
                    "expected frame '{}', received '{}'".format(
                        self.planning_frame,
                        frame_id,
                    )
                )

            target_position = message.pose.position

            position_values = [
                target_position.x,
                target_position.y,
                target_position.z,
            ]

            if not all(
                math.isfinite(value)
                for value in position_values
            ):
                raise ValueError(
                    "position contains NaN or infinity"
                )

            target_quaternion = (
                self.normalize_quaternion(
                    message.pose.orientation
                )
            )

            current_pose = (
                self.arm_group.get_current_pose(
                    self.end_effector_link
                ).pose
            )

            current_quaternion = (
                self.normalize_quaternion(
                    current_pose.orientation
                )
            )

            delta = [
                target_position.x
                - current_pose.position.x,

                target_position.y
                - current_pose.position.y,

                target_position.z
                - current_pose.position.z,
            ]

            distance = math.sqrt(
                sum(value * value for value in delta)
            )

            approach_axis = (
                self.local_positive_z_axis(
                    target_quaternion
                )
            )

            forward_distance = sum(
                delta[index]
                * approach_axis[index]
                for index in range(3)
            )

            lateral_squared = max(
                0.0,
                distance * distance
                - forward_distance * forward_distance,
            )

            lateral_error = math.sqrt(
                lateral_squared
            )

            orientation_error = (
                self.quaternion_angle(
                    current_quaternion,
                    target_quaternion,
                )
            )

            rospy.loginfo(
                "===== Cartesian grasp approach ====="
            )

            rospy.loginfo(
                "Current position: "
                "[%.6f, %.6f, %.6f]",
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z,
            )

            rospy.loginfo(
                "Target grasp position: "
                "[%.6f, %.6f, %.6f]",
                target_position.x,
                target_position.y,
                target_position.z,
            )

            rospy.loginfo(
                "Local +Z axis: "
                "[%.6f, %.6f, %.6f]",
                approach_axis[0],
                approach_axis[1],
                approach_axis[2],
            )

            rospy.loginfo(
                "Total distance: %.6f m",
                distance,
            )

            rospy.loginfo(
                "Forward distance along local +Z: %.6f m",
                forward_distance,
            )

            rospy.loginfo(
                "Lateral error: %.6f m",
                lateral_error,
            )

            rospy.loginfo(
                "Orientation difference: %.6f rad",
                orientation_error,
            )

            if distance > self.max_distance:
                raise ValueError(
                    "target is too far from the current pose"
                )

            if forward_distance <= 0.0:
                raise ValueError(
                    "target is not in the local +Z direction"
                )

            if lateral_error > self.max_lateral_error:
                raise ValueError(
                    "target has excessive lateral offset"
                )

            if (
                orientation_error
                > self.orientation_tolerance
            ):
                raise ValueError(
                    "current and target orientations differ too much"
                )

            target_pose = copy.deepcopy(
                message.pose
            )

            target_pose.orientation.x = (
                target_quaternion[0]
            )
            target_pose.orientation.y = (
                target_quaternion[1]
            )
            target_pose.orientation.z = (
                target_quaternion[2]
            )
            target_pose.orientation.w = (
                target_quaternion[3]
            )

            self.arm_group.set_start_state_to_current_state()

            waypoints = [
                target_pose,
            ]

            plan, fraction = (
                self.arm_group.compute_cartesian_path(
                    waypoints,
                    self.eef_step,
                    True,
                )
            )

            rospy.loginfo(
                "Cartesian path fraction: %.3f",
                fraction,
            )

            if fraction < 0.999:
                raise RuntimeError(
                    "Cartesian path is incomplete: "
                    "fraction={:.3f}".format(
                        fraction
                    )
                )

            if not plan.joint_trajectory.points:
                raise RuntimeError(
                    "Cartesian trajectory contains no points"
                )

            retimed_plan = (
                self.arm_group.retime_trajectory(
                    self.robot.get_current_state(),
                    plan,
                    self.velocity,
                    self.acceleration,
                )
            )

            rospy.loginfo(
                "Executing Cartesian approach..."
            )

            success = self.arm_group.execute(
                retimed_plan,
                wait=True,
            )

            self.arm_group.stop()

            if not success:
                raise RuntimeError(
                    "Cartesian trajectory execution failed"
                )

            rospy.loginfo(
                "Cartesian grasp approach completed successfully."
            )

            rospy.sleep(0.3)

            gripper_request = JointMoveitCtrlRequest()

            gripper_request.joint_states = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            gripper_request.gripper = self.gripper_close

            gripper_request.joint_endpose = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]

            gripper_request.max_velocity = 0.1
            gripper_request.max_acceleration = 0.1

            rospy.loginfo(
                "Closing gripper to %.4f m...",
                self.gripper_close,
            )

            gripper_response = self.gripper_service(
                gripper_request
            )

            if not gripper_response.status:
                raise RuntimeError(
                    "Gripper close failed, error_code={}".format(
                        gripper_response.error_code
                    )
                )

            self.completed = True

            rospy.loginfo(
                "Gripper closed successfully."
            )

            rospy.loginfo(
                "Cartesian grasp and close sequence completed."
            )

            rospy.signal_shutdown(
                "One Cartesian approach and close executed."
            )

        except Exception as error:
            self.arm_group.stop()

            rospy.logerr(
                "Cartesian approach failed: %s",
                str(error),
            )

        finally:
            self.executing = False


def main():
    moveit_commander.roscpp_initialize(
        sys.argv
    )

    rospy.init_node(
        "cartesian_grasp_and_close_node"
    )

    try:
        CartesianGraspApproach()
        rospy.spin()

    except Exception as error:
        rospy.logerr(
            "Node initialization failed: %s",
            str(error),
        )

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
