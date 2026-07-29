#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import threading

import moveit_commander
import numpy as np
import rospy
import tf2_geometry_msgs  # 注册 PoseStamped 的 TF2 转换支持
import tf2_ros

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import UInt32
from tf.transformations import quaternion_matrix

from piper_anygrasp_demo.msg import GraspCandidateArray


class TopKGraspPlanner:
    """Plan Top-K grasp candidates without executing robot motion."""

    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/filtered_grasp_candidates",
        )
        self.output_pose_topic = rospy.get_param(
            "~output_pose_topic",
            "/planned_grasp_tcp_pose",
        )
        self.output_id_topic = rospy.get_param(
            "~output_id_topic",
            "/selected_grasp_candidate_id",
        )

        self.arm_group = rospy.get_param(
            "~arm_group",
            "arm",
        )

        # 已经在当前 Piper 抓放流程中验证稳定的参数。
        self.tcp_offset = float(
            rospy.get_param("~tcp_offset", 0.09755)
        )
        self.approach_distance = float(
            rospy.get_param("~approach_distance", 0.045)
        )

        self.planning_time = float(
            rospy.get_param("~planning_time", 5.0)
        )
        self.planning_attempts = max(
            1,
            int(rospy.get_param("~planning_attempts", 5)),
        )

        self.velocity = float(
            rospy.get_param("~velocity", 0.03)
        )
        self.acceleration = float(
            rospy.get_param("~acceleration", 0.03)
        )

        self.tf_timeout = float(
            rospy.get_param("~tf_timeout", 2.0)
        )

        # -1 表示不注入失败。
        # 测试时设置为 2，可以验证 Candidate 2 失败后尝试 Candidate 3。
        self.forced_failure_id = int(
            rospy.get_param("~forced_failure_id", -1)
        )

        self.robot = moveit_commander.RobotCommander()

        self.arm = moveit_commander.MoveGroupCommander(
            self.arm_group
        )

        self.planning_frame = (
            self.arm.get_planning_frame()
        )
        self.end_effector_link = (
            self.arm.get_end_effector_link()
        )

        self.arm.set_planning_time(
            self.planning_time
        )
        self.arm.set_num_planning_attempts(
            self.planning_attempts
        )
        self.arm.set_max_velocity_scaling_factor(
            self.velocity
        )
        self.arm.set_max_acceleration_scaling_factor(
            self.acceleration
        )
        self.arm.allow_replanning(False)

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(10.0)
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        self.pose_publisher = rospy.Publisher(
            self.output_pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )
        self.id_publisher = rospy.Publisher(
            self.output_id_topic,
            UInt32,
            queue_size=1,
            latch=True,
        )

        self.processing_lock = threading.Lock()

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            GraspCandidateArray,
            self.candidates_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Top-K grasp planner ready: %s",
            self.input_topic,
        )
        rospy.loginfo(
            "MoveIt arm group: %s",
            self.arm_group,
        )
        rospy.loginfo(
            "Planning frame: %s",
            self.planning_frame,
        )
        rospy.loginfo(
            "MoveIt end-effector link: %s",
            self.end_effector_link,
        )
        rospy.loginfo(
            "TCP offset: %.5f m, approach distance: %.4f m",
            self.tcp_offset,
            self.approach_distance,
        )

        if self.forced_failure_id >= 0:
            rospy.logwarn(
                "Failure injection enabled for Candidate %d.",
                self.forced_failure_id,
            )

    @staticmethod
    def normalize_quaternion(quaternion):
        norm = math.sqrt(
            sum(value * value for value in quaternion)
        )

        if norm < 1.0e-8:
            raise ValueError(
                "Quaternion norm is nearly zero."
            )

        return [
            value / norm
            for value in quaternion
        ]

    @staticmethod
    def local_positive_z_axis(quaternion):
        rotation = quaternion_matrix(
            quaternion
        )

        return rotation[:3, 2]

    def transform_candidate_pose(
        self,
        candidate,
        source_frame,
    ):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.header.frame_id = source_frame
        pose_stamped.pose = candidate.pose

        if source_frame == self.planning_frame:
            return pose_stamped

        return self.tf_buffer.transform(
            pose_stamped,
            self.planning_frame,
            rospy.Duration(self.tf_timeout),
        )

    def calculate_pregrasp_tcp(
        self,
        grasp_tcp,
    ):
        orientation = grasp_tcp.pose.orientation

        quaternion = self.normalize_quaternion(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ]
        )

        local_z = self.local_positive_z_axis(
            quaternion
        )

        pregrasp_tcp = PoseStamped()
        pregrasp_tcp.header.stamp = rospy.Time(0)
        pregrasp_tcp.header.frame_id = (
            self.planning_frame
        )

        pregrasp_tcp.pose.position.x = (
            grasp_tcp.pose.position.x
            - self.approach_distance * local_z[0]
        )
        pregrasp_tcp.pose.position.y = (
            grasp_tcp.pose.position.y
            - self.approach_distance * local_z[1]
        )
        pregrasp_tcp.pose.position.z = (
            grasp_tcp.pose.position.z
            - self.approach_distance * local_z[2]
        )

        pregrasp_tcp.pose.orientation.x = quaternion[0]
        pregrasp_tcp.pose.orientation.y = quaternion[1]
        pregrasp_tcp.pose.orientation.z = quaternion[2]
        pregrasp_tcp.pose.orientation.w = quaternion[3]

        return pregrasp_tcp, local_z

    def tcp_pose_to_link6_pose(
        self,
        tcp_pose,
    ):
        orientation = tcp_pose.pose.orientation

        quaternion = self.normalize_quaternion(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ]
        )

        local_z = self.local_positive_z_axis(
            quaternion
        )

        link6_pose = Pose()

        link6_pose.position.x = (
            tcp_pose.pose.position.x
            - self.tcp_offset * local_z[0]
        )
        link6_pose.position.y = (
            tcp_pose.pose.position.y
            - self.tcp_offset * local_z[1]
        )
        link6_pose.position.z = (
            tcp_pose.pose.position.z
            - self.tcp_offset * local_z[2]
        )

        link6_pose.orientation.x = quaternion[0]
        link6_pose.orientation.y = quaternion[1]
        link6_pose.orientation.z = quaternion[2]
        link6_pose.orientation.w = quaternion[3]

        return link6_pose

    @staticmethod
    def parse_plan_result(result):
        """
        Compatible with two common ROS1 MoveIt Python APIs:

        1. plan() -> RobotTrajectory
        2. plan() -> (
               success,
               RobotTrajectory,
               planning_time,
               MoveItErrorCodes,
           )
        """
        success_flag = None
        trajectory = None
        error_code_value = None

        if isinstance(result, tuple):
            if len(result) >= 1:
                success_flag = bool(result[0])

            if len(result) >= 2:
                trajectory = result[1]

            if len(result) >= 4:
                error_code = result[3]
                error_code_value = getattr(
                    error_code,
                    "val",
                    None,
                )
        else:
            trajectory = result

        points = []

        if trajectory is not None:
            joint_trajectory = getattr(
                trajectory,
                "joint_trajectory",
                None,
            )

            if joint_trajectory is not None:
                points = joint_trajectory.points

        has_trajectory = len(points) > 0

        if success_flag is None:
            success = has_trajectory
        else:
            success = success_flag and has_trajectory

        return (
            success,
            trajectory,
            len(points),
            error_code_value,
        )

    def plan_pregrasp(
        self,
        target_pose,
    ):
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(
            target_pose,
            self.end_effector_link,
        )

        try:
            result = self.arm.plan()

        finally:
            self.arm.clear_pose_targets()

        return self.parse_plan_result(result)

    def process_candidates(self, message):
        source_frame = message.header.frame_id.strip()

        if not source_frame:
            rospy.logerr(
                "Filtered candidate array has empty frame_id."
            )
            return

        if not message.candidates:
            rospy.logerr(
                "Filtered candidate array contains no candidates."
            )
            return

        candidates = sorted(
            message.candidates,
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        rospy.loginfo(
            "Received %d filtered candidates. "
            "Planning order: %s",
            len(candidates),
            [candidate.id for candidate in candidates],
        )

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            rospy.loginfo(
                "========== PLAN CANDIDATE %d (%d/%d) ==========",
                candidate.id,
                index,
                len(candidates),
            )

            if candidate.id == self.forced_failure_id:
                rospy.logwarn(
                    "Candidate %d rejected by injected failure.",
                    candidate.id,
                )
                continue

            try:
                grasp_tcp = self.transform_candidate_pose(
                    candidate,
                    source_frame,
                )

                pregrasp_tcp, local_z = (
                    self.calculate_pregrasp_tcp(
                        grasp_tcp
                    )
                )

                pregrasp_link6 = (
                    self.tcp_pose_to_link6_pose(
                        pregrasp_tcp
                    )
                )

            except (
                ValueError,
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as error:
                rospy.logwarn(
                    "Candidate %d preparation failed: %s",
                    candidate.id,
                    error,
                )
                continue

            rospy.loginfo(
                "Candidate %d grasp TCP in %s: "
                "[%.4f, %.4f, %.4f]",
                candidate.id,
                self.planning_frame,
                grasp_tcp.pose.position.x,
                grasp_tcp.pose.position.y,
                grasp_tcp.pose.position.z,
            )
            rospy.loginfo(
                "Candidate %d local +Z: "
                "[%.3f, %.3f, %.3f]",
                candidate.id,
                local_z[0],
                local_z[1],
                local_z[2],
            )
            rospy.loginfo(
                "Candidate %d pregrasp TCP: "
                "[%.4f, %.4f, %.4f]",
                candidate.id,
                pregrasp_tcp.pose.position.x,
                pregrasp_tcp.pose.position.y,
                pregrasp_tcp.pose.position.z,
            )
            rospy.loginfo(
                "Candidate %d: MoveIt planning to pregrasp...",
                candidate.id,
            )

            try:
                (
                    success,
                    _trajectory,
                    trajectory_points,
                    error_code,
                ) = self.plan_pregrasp(
                    pregrasp_link6
                )

            except Exception as error:
                rospy.logwarn(
                    "Candidate %d MoveIt planning raised: %s",
                    candidate.id,
                    error,
                )
                continue

            if not success:
                rospy.logwarn(
                    "Candidate %d planning failed: "
                    "trajectory_points=%d, error_code=%s",
                    candidate.id,
                    trajectory_points,
                    str(error_code),
                )
                continue

            rospy.loginfo(
                "Candidate %d planning succeeded: "
                "%d trajectory points.",
                candidate.id,
                trajectory_points,
            )

            # 发布经过 TF 转换、并且已通过规划检查的抓取 TCP。
            selected_pose = PoseStamped()
            selected_pose.header.stamp = rospy.Time(0)
            selected_pose.header.frame_id = (
                self.planning_frame
            )
            selected_pose.pose = grasp_tcp.pose

            self.pose_publisher.publish(
                selected_pose
            )
            self.id_publisher.publish(
                UInt32(data=candidate.id)
            )

            rospy.loginfo(
                "Selected executable Candidate %d.",
                candidate.id,
            )
            rospy.loginfo(
                "Published selected pose on %s.",
                self.output_pose_topic,
            )
            rospy.loginfo(
                "No robot trajectory was executed."
            )

            return

        rospy.logerr(
            "No MoveIt-plannable grasp candidate remains."
        )

    def candidates_callback(self, message):
        if not self.processing_lock.acquire(
            blocking=False
        ):
            rospy.logwarn(
                "Planner is already processing candidates; "
                "new message ignored."
            )
            return

        try:
            self.process_candidates(message)

        finally:
            self.processing_lock.release()


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("topk_grasp_planner")

    try:
        TopKGraspPlanner()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "Top-K grasp planner failed: %s",
            error,
        )
        sys.exit(1)

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()