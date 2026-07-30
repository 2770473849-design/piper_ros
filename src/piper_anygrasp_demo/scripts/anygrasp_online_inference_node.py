#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from argparse import Namespace

import numpy as np
import rospy

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as point_cloud2

from tf.transformations import quaternion_from_matrix

from piper_anygrasp_demo.msg import (
    GraspCandidate,
    GraspCandidateArray,
)


class AnyGraspOnlineInference:
    """
    Capture one colored PointCloud2 frame, run one AnyGrasp inference,
    adapt the grasp poses to the Piper TCP convention, and publish a
    latched GraspCandidateArray.

    This node intentionally performs one static inference per launch.
    """

    def __init__(self):
        # ---------- Point-cloud input ----------
        self.point_cloud_topic = rospy.get_param(
            "~point_cloud_topic",
            "/hand_camera/depth/points",
        )

        self.point_cloud_timeout = float(
            rospy.get_param(
                "~point_cloud_timeout",
                30.0,
            )
        )

        self.min_depth = float(
            rospy.get_param(
                "~min_depth",
                0.03,
            )
        )

        self.max_depth = float(
            rospy.get_param(
                "~max_depth",
                2.0,
            )
        )

        self.stride = max(
            1,
            int(
                rospy.get_param(
                    "~stride",
                    1,
                )
            ),
        )

        # ---------- Red-target segmentation ----------
        self.red_min = int(
            rospy.get_param(
                "~red_min",
                120,
            )
        )

        self.red_dominance = int(
            rospy.get_param(
                "~red_dominance",
                50,
            )
        )

        self.min_target_points = max(
            1,
            int(
                rospy.get_param(
                    "~min_target_points",
                    20,
                )
            ),
        )

        # Gazebo/OpenNI currently needs this enabled.
        # Set false when the real D435i color order is verified.
        self.swap_red_blue = bool(
            rospy.get_param(
                "~swap_red_blue",
                True,
            )
        )

        # ---------- AnyGrasp detector ----------
        self.checkpoint_path = os.path.expanduser(
            rospy.get_param(
                "~checkpoint_path",
                "~/anygrasp_ws/anygrasp_sdk/"
                "grasp_detection/log/"
                "checkpoint_detection.tar",
            )
        )

        self.max_gripper_width = float(
            rospy.get_param(
                "~max_gripper_width",
                0.07,
            )
        )

        self.gripper_height = float(
            rospy.get_param(
                "~gripper_height",
                0.03,
            )
        )

        self.collision_detection = bool(
            rospy.get_param(
                "~collision_detection",
                True,
            )
        )

        self.top_n = max(
            1,
            int(
                rospy.get_param(
                    "~top_n",
                    5,
                )
            ),
        )

        # ---------- Piper pose adaptation ----------
        self.align_opening_axis = bool(
            rospy.get_param(
                "~align_opening_axis",
                True,
            )
        )

        self.apply_tcp_alignment = bool(
            rospy.get_param(
                "~apply_tcp_alignment",
                True,
            )
        )

        self.tcp_backoff = float(
            rospy.get_param(
                "~tcp_backoff",
                0.015,
            )
        )

        # ---------- ROS output ----------
        self.output_topic = rospy.get_param(
            "~output_topic",
            "/anygrasp_raw_candidates",
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            GraspCandidateArray,
            queue_size=1,
            latch=True,
        )

        self.detector = None

    @staticmethod
    def find_color_field(message):
        fields = {
            field.name: field
            for field in message.fields
        }

        if "rgb" in fields:
            return fields["rgb"]

        if "rgba" in fields:
            return fields["rgba"]

        raise RuntimeError(
            "PointCloud2 has no rgb or rgba field. "
            "Available fields: {}".format(
                sorted(fields.keys())
            )
        )

    @staticmethod
    def decode_packed_color(values, datatype):
        if datatype == PointField.FLOAT32:
            packed = np.asarray(
                values,
                dtype=np.float32,
            ).view(np.uint32)

        elif datatype in (
            PointField.UINT32,
            PointField.INT32,
        ):
            packed = np.asarray(
                values,
                dtype=np.uint32,
            )

        else:
            raise RuntimeError(
                "Unsupported RGB field datatype: {}".format(
                    datatype
                )
            )

        red = (
            (packed >> 16) & 0xFF
        ).astype(np.uint8)

        green = (
            (packed >> 8) & 0xFF
        ).astype(np.uint8)

        blue = (
            packed & 0xFF
        ).astype(np.uint8)

        return np.column_stack(
            (red, green, blue)
        )

    @staticmethod
    def load_anygrasp_api():
        """
        The installed PointNet2 extension is pointnet2._ext, while
        AnyGrasp imports pointnet2_ext. Register the compatibility alias
        before importing gsnet.
        """
        try:
            from pointnet2 import _ext
        except ImportError as error:
            raise RuntimeError(
                "Failed to import pointnet2._ext. "
                "Start this node from the configured AnyGrasp "
                "Conda environment. Original error: {}".format(
                    error
                )
            )

        sys.modules.setdefault(
            "pointnet2_ext",
            _ext,
        )

        try:
            from gsnet import create_detector
        except ImportError as error:
            raise RuntimeError(
                "Failed to import gsnet. Source the AnyGrasp "
                "environment before roslaunch. Original error: {}".format(
                    error
                )
            )

        return create_detector

    @staticmethod
    def adapt_rotation_to_piper(anygrasp_rotation):
        """
        AnyGrasp:
          local +X = approach
          local +Y = gripper opening direction

        Piper execution convention:
          local +Z = approach
          local +Y = gripper opening direction
        """
        return np.column_stack(
            (
                -anygrasp_rotation[:, 2],
                anygrasp_rotation[:, 1],
                anygrasp_rotation[:, 0],
            )
        )

    @staticmethod
    def validate_rotation(rotation, name):
        determinant = float(
            np.linalg.det(rotation)
        )

        orthogonality_error = float(
            np.linalg.norm(
                rotation.T @ rotation
                - np.eye(3)
            )
        )

        if abs(determinant - 1.0) > 1.0e-4:
            raise RuntimeError(
                "{} determinant is invalid: {:.6f}".format(
                    name,
                    determinant,
                )
            )

        if orthogonality_error > 1.0e-4:
            raise RuntimeError(
                "{} is not orthonormal: {:.9f}".format(
                    name,
                    orthogonality_error,
                )
            )

    @staticmethod
    def rotation_to_quaternion(rotation):
        transform = np.eye(
            4,
            dtype=np.float64,
        )
        transform[:3, :3] = rotation

        quaternion = np.asarray(
            quaternion_from_matrix(transform),
            dtype=np.float64,
        )

        norm = np.linalg.norm(quaternion)

        if (
            not np.isfinite(norm)
            or norm < 1.0e-8
        ):
            raise RuntimeError(
                "Rotation produced an invalid quaternion."
            )

        return quaternion / norm

    def load_detector(self):
        if not os.path.isfile(
            self.checkpoint_path
        ):
            raise RuntimeError(
                "AnyGrasp checkpoint does not exist: {}".format(
                    self.checkpoint_path
                )
            )

        rospy.loginfo(
            "Loading AnyGrasp detector from: %s",
            self.checkpoint_path,
        )

        create_detector = (
            self.load_anygrasp_api()
        )

        config = Namespace(
            checkpoint_path=self.checkpoint_path,
            max_gripper_width=self.max_gripper_width,
            gripper_height=self.gripper_height,
        )

        self.detector = create_detector(
            config
        )

        if self.detector is None:
            raise RuntimeError(
                "create_detector returned None."
            )

        rospy.loginfo(
            "AnyGrasp detector loaded: %s",
            type(self.detector),
        )

    def capture_cloud(self):
        rospy.loginfo(
            "Waiting for one colored PointCloud2 on: %s",
            self.point_cloud_topic,
        )

        try:
            message = rospy.wait_for_message(
                self.point_cloud_topic,
                PointCloud2,
                timeout=self.point_cloud_timeout,
            )

        except rospy.ROSException:
            raise RuntimeError(
                "Timed out waiting for '{}'.".format(
                    self.point_cloud_topic
                )
            )

        color_field = self.find_color_field(
            message
        )

        rospy.loginfo(
            "Received cloud: frame=%s, width=%d, "
            "height=%d, color_field=%s, datatype=%d",
            message.header.frame_id,
            message.width,
            message.height,
            color_field.name,
            color_field.datatype,
        )

        rows = list(
            point_cloud2.read_points(
                message,
                field_names=(
                    "x",
                    "y",
                    "z",
                    color_field.name,
                ),
                skip_nans=False,
            )
        )

        if not rows:
            raise RuntimeError(
                "Received an empty point cloud."
            )

        points = np.asarray(
            [
                row[:3]
                for row in rows
            ],
            dtype=np.float32,
        )

        packed_color_values = [
            row[3]
            for row in rows
        ]

        colors = self.decode_packed_color(
            packed_color_values,
            color_field.datatype,
        )

        if self.swap_red_blue:
            colors = colors[:, [2, 1, 0]]

            rospy.loginfo(
                "Swapped red and blue channels for "
                "the current point-cloud source."
            )

        finite_mask = np.isfinite(
            points
        ).all(axis=1)

        depth_mask = (
            (points[:, 2] >= self.min_depth)
            & (points[:, 2] <= self.max_depth)
        )

        valid_mask = (
            finite_mask
            & depth_mask
        )

        points = np.ascontiguousarray(
            points[valid_mask],
            dtype=np.float32,
        )

        colors = np.ascontiguousarray(
            colors[valid_mask],
            dtype=np.uint8,
        )

        red = colors[:, 0].astype(
            np.int16
        )
        green = colors[:, 1].astype(
            np.int16
        )
        blue = colors[:, 2].astype(
            np.int16
        )

        region_mask = (
            (red >= self.red_min)
            & (
                (red - green)
                >= self.red_dominance
            )
            & (
                (red - blue)
                >= self.red_dominance
            )
        )

        target_points = points[
            region_mask
        ]

        if (
            target_points.shape[0]
            < self.min_target_points
        ):
            raise RuntimeError(
                "Detected only {} red target points; "
                "minimum required is {}. Check the RGB "
                "stream and color thresholds.".format(
                    target_points.shape[0],
                    self.min_target_points,
                )
            )

        object_center = np.median(
            target_points,
            axis=0,
        ).astype(np.float64)

        rospy.loginfo(
            "Valid cloud points: %d",
            points.shape[0],
        )

        rospy.loginfo(
            "Red target points: %d",
            target_points.shape[0],
        )

        rospy.loginfo(
            "Segmented object center in %s: "
            "[%.4f, %.4f, %.4f]",
            message.header.frame_id,
            object_center[0],
            object_center[1],
            object_center[2],
        )

        if self.stride > 1:
            points = np.ascontiguousarray(
                points[::self.stride],
                dtype=np.float32,
            )

            region_mask = np.ascontiguousarray(
                region_mask[::self.stride],
                dtype=bool,
            )

        else:
            points = np.ascontiguousarray(
                points,
                dtype=np.float32,
            )

            region_mask = np.ascontiguousarray(
                region_mask,
                dtype=bool,
            )

        if int(region_mask.sum()) == 0:
            raise RuntimeError(
                "Point-cloud stride removed all target "
                "region points."
            )

        rospy.loginfo(
            "Inference points after stride=%d: %d",
            self.stride,
            points.shape[0],
        )

        rospy.loginfo(
            "Region-steering points after stride: %d",
            int(region_mask.sum()),
        )

        return (
            points,
            region_mask,
            object_center,
            message.header.frame_id,
            message.header.stamp,
        )

    def infer(self, points, region_mask):
        optional_params = {
            "dense_grasp": False,
            "collision_detection": (
                self.collision_detection
            ),
            "region_steering": region_mask,
            "approach_steering": None,
            "approach_thresh": np.pi,
        }

        rospy.loginfo(
            "Starting detector.get_grasp()..."
        )

        start_time = time.time()

        grasp_group = self.detector.get_grasp(
            points,
            optional_params,
        )

        elapsed = (
            time.time()
            - start_time
        )

        rospy.loginfo(
            "AnyGrasp inference time: %.3f s",
            elapsed,
        )

        if grasp_group is None:
            raise RuntimeError(
                "AnyGrasp returned None."
            )

        if len(grasp_group) == 0:
            raise RuntimeError(
                "AnyGrasp returned an empty GraspGroup."
            )

        pre_nms_count = len(
            grasp_group
        )

        grasp_group = grasp_group.nms()
        grasp_group = (
            grasp_group.sort_by_score()
        )

        rospy.loginfo(
            "AnyGrasp candidates: pre-NMS=%d, post-NMS=%d",
            pre_nms_count,
            len(grasp_group),
        )

        if len(grasp_group) == 0:
            raise RuntimeError(
                "No candidates remained after NMS."
            )

        return grasp_group

    def build_candidate_message(
        self,
        grasp_group,
        object_center,
        frame_id,
        stamp,
    ):
        publish_count = min(
            self.top_n,
            len(grasp_group),
        )

        output = GraspCandidateArray()

        if stamp.to_sec() > 0.0:
            output.header.stamp = stamp
        else:
            output.header.stamp = (
                rospy.Time.now()
            )

        output.header.frame_id = frame_id

        rospy.loginfo(
            "Preparing %d/%d online AnyGrasp candidates "
            "in frame '%s'.",
            publish_count,
            len(grasp_group),
            frame_id,
        )

        for index in range(
            publish_count
        ):
            grasp = grasp_group[
                index
            ]

            raw_position = np.asarray(
                grasp.translation,
                dtype=np.float64,
            )

            anygrasp_rotation = np.asarray(
                grasp.rotation_matrix,
                dtype=np.float64,
            )

            self.validate_rotation(
                anygrasp_rotation,
                "AnyGrasp Candidate {}".format(
                    index
                ),
            )

            piper_rotation = (
                self.adapt_rotation_to_piper(
                    anygrasp_rotation
                )
            )

            self.validate_rotation(
                piper_rotation,
                "Piper Candidate {}".format(
                    index
                ),
            )

            local_y = np.asarray(
                piper_rotation[:, 1],
                dtype=np.float64,
            )

            local_z = np.asarray(
                piper_rotation[:, 2],
                dtype=np.float64,
            )

            lateral_shift = float(
                np.dot(
                    object_center
                    - raw_position,
                    local_y,
                )
            )

            if self.align_opening_axis:
                aligned_position = (
                    raw_position
                    + lateral_shift
                    * local_y
                )
            else:
                aligned_position = (
                    raw_position
                )

            if self.apply_tcp_alignment:
                tcp_position = (
                    aligned_position
                    - self.tcp_backoff
                    * local_z
                )
            else:
                tcp_position = (
                    aligned_position
                )

            quaternion = (
                self.rotation_to_quaternion(
                    piper_rotation
                )
            )

            candidate = GraspCandidate()
            candidate.id = index

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

            candidate.score = float(
                grasp.score
            )
            candidate.width = float(
                grasp.width
            )
            candidate.height = float(
                grasp.height
            )
            candidate.depth = float(
                grasp.depth
            )

            output.candidates.append(
                candidate
            )

            rospy.loginfo(
                "Candidate %d: score=%.3f, width=%.4f m, "
                "raw=[%.4f, %.4f, %.4f], "
                "opening-axis shift=%+.2f mm, "
                "Piper TCP=[%.4f, %.4f, %.4f]",
                index,
                candidate.score,
                candidate.width,
                raw_position[0],
                raw_position[1],
                raw_position[2],
                lateral_shift * 1000.0,
                tcp_position[0],
                tcp_position[1],
                tcp_position[2],
            )

        return output

    def run(self):
        rospy.loginfo(
            "========== ANYGRASP ONLINE SINGLE-SHOT =========="
        )

        self.load_detector()

        (
            points,
            region_mask,
            object_center,
            frame_id,
            stamp,
        ) = self.capture_cloud()

        grasp_group = self.infer(
            points,
            region_mask,
        )

        output = self.build_candidate_message(
            grasp_group,
            object_center,
            frame_id,
            stamp,
        )

        self.publisher.publish(
            output
        )

        rospy.loginfo(
            "Published %d online AnyGrasp candidates on %s.",
            len(output.candidates),
            self.output_topic,
        )

        rospy.loginfo(
            "Node remains alive for latched subscribers."
        )

        rospy.loginfo(
            "========== ONLINE INFERENCE COMPLETED =========="
        )

        rospy.spin()


def main():
    rospy.init_node(
        "anygrasp_online_inference_node"
    )

    try:
        AnyGraspOnlineInference().run()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "Online AnyGrasp inference failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
