#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys

import numpy as np
import rospy
import tf2_geometry_msgs  # 注册 PoseStamped 的 TF2 转换支持
import tf2_ros

from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_matrix

from piper_anygrasp_demo.msg import (
    GraspCandidate,
    GraspCandidateArray,
)


class GraspCandidateFilter:
    """Filter, rank and publish grasp candidates."""

    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/grasp_candidates",
        )
        self.filtered_topic = rospy.get_param(
            "~filtered_topic",
            "/filtered_grasp_candidates",
        )
        self.selected_pose_topic = rospy.get_param(
            "~selected_pose_topic",
            "/grasp_tcp_pose",
        )
        self.target_frame = rospy.get_param(
            "~target_frame",
            "base_link",
        )

        # 当前仿真抓取工作空间。
        self.x_min = float(rospy.get_param("~x_min", 0.20))
        self.x_max = float(rospy.get_param("~x_max", 0.32))
        self.y_min = float(rospy.get_param("~y_min", -0.08))
        self.y_max = float(rospy.get_param("~y_max", 0.08))
        self.z_min = float(rospy.get_param("~z_min", 0.030))
        self.z_max = float(rospy.get_param("~z_max", 0.080))

        # width 表示两根手指之间的总开口宽度。
        self.width_min = float(
            rospy.get_param("~width_min", 0.010)
        )
        self.width_max = float(
            rospy.get_param("~width_max", 0.070)
        )

        self.min_score = float(
            rospy.get_param("~min_score", 0.0)
        )
        self.top_k = max(
            1,
            int(rospy.get_param("~top_k", 5)),
        )

        # 抓取器局部 +Z 是接近方向。
        # 正上方抓取时，转换到 base_link 后应大致指向 -Z。
        self.max_approach_z = float(
            rospy.get_param("~max_approach_z", -0.80)
        )

        self.tf_timeout = float(
            rospy.get_param("~tf_timeout", 2.0)
        )

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(10.0)
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        self.filtered_publisher = rospy.Publisher(
            self.filtered_topic,
            GraspCandidateArray,
            queue_size=1,
            latch=True,
        )

        # 暂时继续兼容现有单候选抓放执行节点。
        self.selected_pose_publisher = rospy.Publisher(
            self.selected_pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            GraspCandidateArray,
            self.candidates_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Grasp candidate filter ready: %s -> %s",
            self.input_topic,
            self.filtered_topic,
        )
        rospy.loginfo(
            "Target frame: %s, Top-K: %d",
            self.target_frame,
            self.top_k,
        )

    @staticmethod
    def candidate_numeric_values(candidate):
        pose = candidate.pose

        return [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            candidate.score,
            candidate.width,
            candidate.height,
            candidate.depth,
        ]

    def transform_pose(
        self,
        candidate,
        source_frame,
    ):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.header.frame_id = source_frame
        pose_stamped.pose = candidate.pose

        if source_frame == self.target_frame:
            return pose_stamped

        rospy.loginfo(
            "Transforming candidate %d from %s to %s.",
            candidate.id,
            source_frame,
            self.target_frame,
        )

        return self.tf_buffer.transform(
            pose_stamped,
            self.target_frame,
            rospy.Duration(self.tf_timeout),
        )

    def validate_candidate(
        self,
        candidate,
        source_frame,
    ):
        values = self.candidate_numeric_values(
            candidate
        )

        if not all(math.isfinite(value) for value in values):
            return None, "contains NaN or Inf"

        if candidate.score < self.min_score:
            return (
                None,
                "score {:.3f} below {:.3f}".format(
                    candidate.score,
                    self.min_score,
                ),
            )

        if not (
            self.width_min
            <= candidate.width
            <= self.width_max
        ):
            return (
                None,
                "width {:.4f} m outside "
                "[{:.4f}, {:.4f}]".format(
                    candidate.width,
                    self.width_min,
                    self.width_max,
                ),
            )

        if candidate.height <= 0.0:
            return None, "height must be positive"

        if candidate.depth <= 0.0:
            return None, "depth must be positive"

        try:
            transformed = self.transform_pose(
                candidate,
                source_frame,
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            return None, "TF transform failed: {}".format(error)

        position = transformed.pose.position

        if not self.x_min <= position.x <= self.x_max:
            return (
                None,
                "x {:.4f} outside [{:.4f}, {:.4f}]".format(
                    position.x,
                    self.x_min,
                    self.x_max,
                ),
            )

        if not self.y_min <= position.y <= self.y_max:
            return (
                None,
                "y {:.4f} outside [{:.4f}, {:.4f}]".format(
                    position.y,
                    self.y_min,
                    self.y_max,
                ),
            )

        if not self.z_min <= position.z <= self.z_max:
            return (
                None,
                "z {:.4f} outside [{:.4f}, {:.4f}]".format(
                    position.z,
                    self.z_min,
                    self.z_max,
                ),
            )

        orientation = transformed.pose.orientation

        quaternion = np.array(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ],
            dtype=np.float64,
        )

        quaternion_norm = float(
            np.linalg.norm(quaternion)
        )

        if quaternion_norm < 1.0e-6:
            return None, "quaternion norm is nearly zero"

        # 归一化，避免网络输出的四元数有微小数值误差。
        quaternion /= quaternion_norm

        rotation = quaternion_matrix(quaternion)

        # 旋转矩阵第三列：夹爪局部 +Z 在目标坐标系中的方向。
        approach = rotation[:3, 2]

        if approach[2] > self.max_approach_z:
            return (
                None,
                "approach direction "
                "[{:.3f}, {:.3f}, {:.3f}] "
                "is not sufficiently downward".format(
                    approach[0],
                    approach[1],
                    approach[2],
                ),
            )

        accepted = GraspCandidate()
        accepted.id = candidate.id
        accepted.pose = transformed.pose

        accepted.pose.orientation.x = float(
            quaternion[0]
        )
        accepted.pose.orientation.y = float(
            quaternion[1]
        )
        accepted.pose.orientation.z = float(
            quaternion[2]
        )
        accepted.pose.orientation.w = float(
            quaternion[3]
        )

        accepted.score = candidate.score
        accepted.width = candidate.width
        accepted.height = candidate.height
        accepted.depth = candidate.depth

        reason = (
            "accepted: score={:.3f}, "
            "approach=[{:.3f}, {:.3f}, {:.3f}]"
        ).format(
            accepted.score,
            approach[0],
            approach[1],
            approach[2],
        )

        return accepted, reason

    def candidates_callback(self, message):
        source_frame = message.header.frame_id.strip()

        if not source_frame:
            rospy.logerr(
                "Received grasp candidate array "
                "with an empty frame_id."
            )
            return

        rospy.loginfo(
            "Received %d candidates in frame '%s'.",
            len(message.candidates),
            source_frame,
        )

        accepted_candidates = []

        for candidate in message.candidates:
            accepted, reason = self.validate_candidate(
                candidate,
                source_frame,
            )

            if accepted is None:
                rospy.logwarn(
                    "Candidate %d rejected: %s",
                    candidate.id,
                    reason,
                )
                continue

            accepted_candidates.append(accepted)

            rospy.loginfo(
                "Candidate %d %s",
                candidate.id,
                reason,
            )

        accepted_candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        accepted_candidates = accepted_candidates[
            : self.top_k
        ]

        filtered_message = GraspCandidateArray()
        filtered_message.header.stamp = rospy.Time.now()
        filtered_message.header.frame_id = self.target_frame
        filtered_message.candidates = accepted_candidates

        self.filtered_publisher.publish(
            filtered_message
        )

        accepted_ids = [
            candidate.id
            for candidate in accepted_candidates
        ]

        rospy.loginfo(
            "Published %d filtered candidates. IDs: %s",
            len(accepted_candidates),
            accepted_ids,
        )

        if not accepted_candidates:
            rospy.logerr(
                "No valid grasp candidate remains "
                "after filtering."
            )
            return

        selected = accepted_candidates[0]

        selected_pose = PoseStamped()
        selected_pose.header.stamp = rospy.Time(0)
        selected_pose.header.frame_id = self.target_frame
        selected_pose.pose = selected.pose

        self.selected_pose_publisher.publish(
            selected_pose
        )

        rospy.loginfo(
            "Selected candidate %d: "
            "score=%.3f, width=%.4f m, "
            "TCP=[%.4f, %.4f, %.4f]",
            selected.id,
            selected.score,
            selected.width,
            selected.pose.position.x,
            selected.pose.position.y,
            selected.pose.position.z,
        )
        rospy.loginfo(
            "Published selected pose on %s.",
            self.selected_pose_topic,
        )


def main():
    rospy.init_node("grasp_candidate_filter")

    try:
        GraspCandidateFilter()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "Grasp candidate filter failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()