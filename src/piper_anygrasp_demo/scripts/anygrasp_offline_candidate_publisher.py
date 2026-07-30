#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import rospy

from tf.transformations import quaternion_from_matrix

from piper_anygrasp_demo.msg import (
    GraspCandidate,
    GraspCandidateArray,
)


class AnyGraspOfflineCandidatePublisher:
    """Publish saved real AnyGrasp results as ROS grasp candidates."""

    def __init__(self):
        self.input_path = os.path.expanduser(
            rospy.get_param(
                "~input_path",
                "~/anygrasp_ws/captures/"
                "anygrasp_offline_results_latest.npz",
            )
        )

        self.capture_path = os.path.expanduser(
            rospy.get_param(
                "~capture_path",
                "~/anygrasp_ws/captures/"
                "hand_camera_xyzrgb_latest.npz",
            )
        )

        # 只沿夹爪开合轴，把候选中心对齐到分割物体中心。
        self.align_opening_axis = bool(
            rospy.get_param(
                "~align_opening_axis",
                True,
            )
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/anygrasp_raw_candidates",
        )

        self.top_n = max(
            1,
            int(rospy.get_param("~top_n", 5)),
        )

        # AnyGrasp抓取中心到当前Piper TCP定义的固定补偿。
        # 沿适配后的Piper局部+Z接近方向反向移动。
        self.tcp_backoff = float(
            rospy.get_param(
                "~tcp_backoff",
                0.015,
            )
        )

        self.apply_tcp_alignment = bool(
            rospy.get_param(
                "~apply_tcp_alignment",
                True,
            )
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            GraspCandidateArray,
            queue_size=1,
            latch=True,
        )

    @staticmethod
    def rotation_to_quaternion(rotation):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation

        quaternion = np.asarray(
            quaternion_from_matrix(transform),
            dtype=np.float64,
        )

        norm = np.linalg.norm(quaternion)

        if not np.isfinite(norm) or norm < 1.0e-8:
            raise RuntimeError(
                "Rotation produced an invalid quaternion."
            )

        return quaternion / norm

    def run(self):
        if not os.path.isfile(self.input_path):
            raise RuntimeError(
                "Inference result does not exist: {}".format(
                    self.input_path
                )
            )

        data = np.load(self.input_path)

        frame_id = str(data["frame_id"])
        translations = data["translations"]
        rotations = data["piper_rotation_matrices"]
        scores = data["scores"]
        widths = data["widths"]
        heights = data["heights"]
        depths = data["depths"]

        if not os.path.isfile(self.capture_path):
            raise RuntimeError(
                "XYZRGB capture does not exist: {}".format(
                    self.capture_path
                )
            )

        capture_data = np.load(
            self.capture_path
        )

        red_points = np.asarray(
            capture_data["red_points"],
            dtype=np.float64,
        )

        if red_points.shape[0] == 0:
            raise RuntimeError(
                "XYZRGB capture contains no red target points."
            )

        object_center = np.median(
            red_points,
            axis=0,
        )

        rospy.loginfo(
            "Segmented object center in %s: "
            "[%.4f, %.4f, %.4f]",
            frame_id,
            object_center[0],
            object_center[1],
            object_center[2],
        )

        available_count = len(scores)
        publish_count = min(
            self.top_n,
            available_count,
        )

        if publish_count == 0:
            raise RuntimeError(
                "Inference result contains no candidates."
            )

        message = GraspCandidateArray()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = frame_id

        rospy.loginfo(
            "Publishing %d/%d real AnyGrasp candidates "
            "in frame '%s'.",
            publish_count,
            available_count,
            frame_id,
        )

        for index in range(publish_count):
            quaternion = self.rotation_to_quaternion(
                rotations[index]
            )

            candidate = GraspCandidate()
            candidate.id = index

            raw_position = np.asarray(
                translations[index],
                dtype=np.float64,
            )

            # Piper局部+Y：夹爪开合方向。
            local_y = np.asarray(
                rotations[index][:, 1],
                dtype=np.float64,
            )

            # Piper局部+Z：抓取接近方向。
            local_z = np.asarray(
                rotations[index][:, 2],
                dtype=np.float64,
            )

            lateral_shift = float(
                np.dot(
                    object_center - raw_position,
                    local_y,
                )
            )

            if self.align_opening_axis:
                aligned_position = (
                    raw_position
                    + lateral_shift * local_y
                )
            else:
                aligned_position = raw_position

            if self.apply_tcp_alignment:
                tcp_position = (
                    aligned_position
                    - self.tcp_backoff * local_z
                )
            else:
                tcp_position = aligned_position

            candidate.pose.position.x = float(
                tcp_position[0]
            )
            candidate.pose.position.y = float(
                tcp_position[1]
            )
            candidate.pose.position.z = float(
                tcp_position[2]
            )

            candidate.pose.orientation.x = float(
                quaternion[0]
            )
            candidate.pose.orientation.y = float(
                quaternion[1]
            )
            candidate.pose.orientation.z = float(
                quaternion[2]
            )
            candidate.pose.orientation.w = float(
                quaternion[3]
            )

            candidate.score = float(scores[index])
            candidate.width = float(widths[index])
            candidate.height = float(heights[index])
            candidate.depth = float(depths[index])

            message.candidates.append(candidate)


            rospy.loginfo(
                "Candidate %d: score=%.3f, width=%.4f m, "
                "raw=[%.4f, %.4f, %.4f], "
                "opening-axis shift=%+.2f mm, "
                "aligned=[%.4f, %.4f, %.4f], "
                "Piper TCP=[%.4f, %.4f, %.4f]",
                index,
                candidate.score,
                candidate.width,
                raw_position[0],
                raw_position[1],
                raw_position[2],
                lateral_shift * 1000.0,
                aligned_position[0],
                aligned_position[1],
                aligned_position[2],
                tcp_position[0],
                tcp_position[1],
                tcp_position[2],
            )

        self.publisher.publish(message)

        rospy.loginfo(
            "Published real AnyGrasp candidates on %s.",
            self.output_topic,
        )

        rospy.loginfo(
            "Node remains alive for latched subscribers."
        )

        rospy.spin()


def main():
    rospy.init_node(
        "anygrasp_offline_candidate_publisher"
    )

    try:
        AnyGraspOfflineCandidatePublisher().run()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "Offline AnyGrasp candidate publisher failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()