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


class FullGraspSequence:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            DEFAULT_INPUT_TOPIC,
        )

        self.planning_frame = rospy.get_param(
            "~planning_frame",
            "dummy_link",
        )

        self.approach_distance = float(
            rospy.get_param(
                "~approach_distance",
                0.01,
            )
        )

        self.max_approach_distance = float(
            rospy.get_param(
                "~max_approach_distance",
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

        self.gripper_open = float(
            rospy.get_param(
                "~gripper_open",
                0.02,
            )
        )

        self.gripper_close = float(
            rospy.get_param(
                "~gripper_close",
                0.0,
            )
        )

        self.validate_parameters()

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
            "Waiting for transformed grasp pose: %s",
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

        rospy.loginfo(
            "Approach distance: %.4f m",
            self.approach_distance,
        )

    def validate_parameters(self):
        if not 0.0 < self.approach_distance <= 0.20:
            raise ValueError(
                "~approach_distance must be in (0, 0.20]."
            )

        if not 0.0 < self.max_approach_distance <= 0.20:
            raise ValueError(
                "~max_approach_distance must be in (0, 0.20]."
            )

        if not 0.0 < self.max_lateral_error <= 0.05:
            raise ValueError(
                "~max_lateral_error must be in (0, 0.05]."
            )

        if not 0.0 < self.eef_step <= 0.01:
            raise ValueError(
                "~eef_step must be in (0, 0.01]."
            )

        if not 0.0 < self.velocity <= 1.0:
            raise ValueError(
                "~velocity must be in (0, 1]."
            )

        if not 0.0 < self.acceleration <= 1.0:
            raise ValueError(
                "~acceleration must be in (0, 1]."
            )

        for name, value in [
            ("~gripper_open", self.gripper_open),
            ("~gripper_close", self.gripper_close),
        ]:
            if not 0.0 <= value <= 0.035:
                raise ValueError(
                    "{} must be in [0, 0.035].".format(
                        name
                    )
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
                "Quaternion contains NaN or infinity."
            )

        norm = math.sqrt(
            sum(value * value for value in values)
        )

        if norm < 1e-8:
            raise ValueError(
                "Quaternion norm is zero."
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

        if norm < 1e-8:
            raise ValueError(
                "Computed approach axis is invalid."
            )

        return [
            value / norm
            for value in axis
        ]

    @staticmethod
    def quaternion_angle(first, second):
        dot = abs(
            sum(
                a * b
                for a, b in zip(first, second)
            )
        )

        dot = max(0.0, min(1.0, dot))

        return 2.0 * math.acos(dot)

    def command_gripper(self, opening, stage_name):
        request = JointMoveitCtrlRequest()

        request.joint_states = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        request.gripper = opening

        request.joint_endpose = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        request.max_velocity = 0.1
        request.max_acceleration = 0.1

        rospy.loginfo(
            "%s: gripper target %.4f m",
            stage_name,
            opening,
        )

        response = self.gripper_service(request)

        if not response.status:
            raise RuntimeError(
                "{} failed, error_code={}".format(
                    stage_name,
                    response.error_code,
                )
            )

        rospy.loginfo(
            "%s completed.",
            stage_name,
        )

    def move_to_pose(self, target_pose, stage_name):
        self.arm_group.set_start_state_to_current_state()

        self.arm_group.set_max_velocity_scaling_factor(
            self.velocity
        )

        self.arm_group.set_max_acceleration_scaling_factor(
            self.acceleration
        )

        self.arm_group.set_pose_target(
            target_pose,
            self.end_effector_link,
        )

        rospy.loginfo(
            "%s: planning and executing...",
            stage_name,
        )

        success = self.arm_group.go(wait=True)

        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if not success:
            raise RuntimeError(
                "{} failed.".format(stage_name)
            )

        rospy.loginfo(
            "%s completed.",
            stage_name,
        )

    def execute_cartesian(
        self,
        target_pose,
        stage_name,
    ):
        self.arm_group.set_start_state_to_current_state()

        plan, fraction = (
            self.arm_group.compute_cartesian_path(
                [target_pose],
                self.eef_step,
                True,
            )
        )

        rospy.loginfo(
            "%s path fraction: %.3f",
            stage_name,
            fraction,
        )

        if fraction < 0.999:
            raise RuntimeError(
                "{} path is incomplete: "
                "fraction={:.3f}".format(
                    stage_name,
                    fraction,
                )
            )

        if not plan.joint_trajectory.points:
            raise RuntimeError(
                "{} trajectory contains no points.".format(
                    stage_name
                )
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
            "%s: executing...",
            stage_name,
        )

        success = self.arm_group.execute(
            retimed_plan,
            wait=True,
        )

        self.arm_group.stop()

        if not success:
            raise RuntimeError(
                "{} execution failed.".format(
                    stage_name
                )
            )

        rospy.loginfo(
            "%s completed.",
            stage_name,
        )

    def pose_callback(self, message):
        if self.executing or self.completed:
            return

        self.executing = True

        try:
            frame_id = message.header.frame_id.strip()

            if frame_id != self.planning_frame:
                raise ValueError(
                    "Expected frame '{}', received '{}'.".format(
                        self.planning_frame,
                        frame_id,
                    )
                )

            position = message.pose.position

            position_values = [
                position.x,
                position.y,
                position.z,
            ]

            if not all(
                math.isfinite(value)
                for value in position_values
            ):
                raise ValueError(
                    "Position contains NaN or infinity."
                )

            quaternion = self.normalize_quaternion(
                message.pose.orientation
            )

            approach_axis = (
                self.local_positive_z_axis(
                    quaternion
                )
            )

            grasp_pose = copy.deepcopy(
                message.pose
            )

            grasp_pose.orientation.x = quaternion[0]
            grasp_pose.orientation.y = quaternion[1]
            grasp_pose.orientation.z = quaternion[2]
            grasp_pose.orientation.w = quaternion[3]

            pregrasp_pose = copy.deepcopy(
                grasp_pose
            )

            pregrasp_pose.position.x -= (
                self.approach_distance
                * approach_axis[0]
            )

            pregrasp_pose.position.y -= (
                self.approach_distance
                * approach_axis[1]
            )

            pregrasp_pose.position.z -= (
                self.approach_distance
                * approach_axis[2]
            )

            rospy.loginfo(
                "===== Full grasp sequence ====="
            )

            rospy.loginfo(
                "Grasp position: "
                "[%.6f, %.6f, %.6f]",
                grasp_pose.position.x,
                grasp_pose.position.y,
                grasp_pose.position.z,
            )

            rospy.loginfo(
                "Local +Z axis: "
                "[%.6f, %.6f, %.6f]",
                approach_axis[0],
                approach_axis[1],
                approach_axis[2],
            )

            rospy.loginfo(
                "Pregrasp position: "
                "[%.6f, %.6f, %.6f]",
                pregrasp_pose.position.x,
                pregrasp_pose.position.y,
                pregrasp_pose.position.z,
            )

            self.command_gripper(
                self.gripper_open,
                "GRIPPER OPEN",
            )

            self.move_to_pose(
                pregrasp_pose,
                "MOVE TO PREGRASP",
            )

            rospy.sleep(0.3)

            actual_pregrasp = (
                self.arm_group.get_current_pose(
                    self.end_effector_link
                ).pose
            )

            actual_quaternion = (
                self.normalize_quaternion(
                    actual_pregrasp.orientation
                )
            )

            delta = [
                grasp_pose.position.x
                - actual_pregrasp.position.x,

                grasp_pose.position.y
                - actual_pregrasp.position.y,

                grasp_pose.position.z
                - actual_pregrasp.position.z,
            ]

            total_distance = math.sqrt(
                sum(value * value for value in delta)
            )

            forward_distance = sum(
                delta[index]
                * approach_axis[index]
                for index in range(3)
            )

            lateral_squared = max(
                0.0,
                total_distance * total_distance
                - forward_distance * forward_distance,
            )

            lateral_error = math.sqrt(
                lateral_squared
            )

            orientation_error = (
                self.quaternion_angle(
                    actual_quaternion,
                    quaternion,
                )
            )

            rospy.loginfo(
                "Actual pregrasp position: "
                "[%.6f, %.6f, %.6f]",
                actual_pregrasp.position.x,
                actual_pregrasp.position.y,
                actual_pregrasp.position.z,
            )

            rospy.loginfo(
                "Approach distance: %.6f m",
                total_distance,
            )

            rospy.loginfo(
                "Forward distance: %.6f m",
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

            if total_distance > self.max_approach_distance:
                raise RuntimeError(
                    "Approach target is too far away."
                )

            if forward_distance <= 0.0:
                raise RuntimeError(
                    "Grasp target is not along local +Z."
                )

            if lateral_error > self.max_lateral_error:
                raise RuntimeError(
                    "Approach has excessive lateral error."
                )

            if (
                orientation_error
                > self.orientation_tolerance
            ):
                raise RuntimeError(
                    "Pregrasp orientation differs too much."
                )

            self.execute_cartesian(
                grasp_pose,
                "CARTESIAN APPROACH",
            )

            self.command_gripper(
                self.gripper_close,
                "GRIPPER CLOSE",
            )

            rospy.sleep(0.3)

            self.execute_cartesian(
                actual_pregrasp,
                "CARTESIAN RETREAT",
            )

            self.completed = True

            rospy.loginfo(
                "Full grasp sequence completed successfully."
            )

            rospy.signal_shutdown(
                "One full grasp sequence executed."
            )

        except Exception as error:
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()

            rospy.logerr(
                "Full grasp sequence failed: %s",
                str(error),
            )

        finally:
            self.executing = False


def main():
    moveit_commander.roscpp_initialize(
        sys.argv
    )

    rospy.init_node(
        "full_grasp_sequence_node"
    )

    try:
        FullGraspSequence()
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
