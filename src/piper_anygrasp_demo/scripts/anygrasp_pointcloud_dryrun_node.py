#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import rospy

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as point_cloud2


class AnyGraspPointCloudDryRun:
    """
    读取一帧 Gazebo RGB-D 点云，转换为 AnyGrasp 所需的
    Nx3 float32 数组，但暂时不进行神经网络推理。
    """

    def __init__(self):
        self.point_cloud_topic = rospy.get_param(
            "~point_cloud_topic",
            "/hand_camera/depth/points",
        )

        self.timeout = float(
            rospy.get_param(
                "~timeout",
                30.0,
            )
        )

        # ROS optical frame 中 +Z 是相机前方。
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

        self.sample_stride = max(
            1,
            int(
                rospy.get_param(
                    "~sample_stride",
                    1,
                )
            ),
        )

        default_output_path = os.path.expanduser(
            "~/anygrasp_ws/captures/"
            "hand_camera_points_latest.npz"
        )

        self.output_path = os.path.expanduser(
            rospy.get_param(
                "~output_path",
                default_output_path,
            )
        )

    @staticmethod
    def pointcloud_to_numpy(message):
        available_fields = {
            field.name for field in message.fields
        }

        required_fields = {"x", "y", "z"}

        if not required_fields.issubset(
            available_fields
        ):
            raise RuntimeError(
                "PointCloud2 lacks XYZ fields. "
                "Available fields: {}".format(
                    sorted(available_fields)
                )
            )

        expected_points = (
            int(message.width)
            * int(message.height)
        )

        point_iterator = point_cloud2.read_points(
            message,
            field_names=("x", "y", "z"),
            skip_nans=False,
        )

        flat_points = np.fromiter(
            (
                coordinate
                for point in point_iterator
                for coordinate in point
            ),
            dtype=np.float32,
            count=expected_points * 3,
        )

        if flat_points.size == 0:
            raise RuntimeError(
                "Received an empty PointCloud2 message."
            )

        if flat_points.size % 3 != 0:
            raise RuntimeError(
                "PointCloud2 conversion produced an "
                "invalid number of coordinates: {}.".format(
                    flat_points.size
                )
            )

        return flat_points.reshape((-1, 3))

    def run(self):
        rospy.loginfo(
            "========== ANYGRASP POINT CLOUD DRY RUN =========="
        )

        rospy.loginfo(
            "Waiting for PointCloud2 on: %s",
            self.point_cloud_topic,
        )

        try:
            message = rospy.wait_for_message(
                self.point_cloud_topic,
                PointCloud2,
                timeout=self.timeout,
            )

        except rospy.ROSException:
            raise RuntimeError(
                "Timed out waiting for point cloud "
                "on '{}'.".format(
                    self.point_cloud_topic
                )
            )

        rospy.loginfo(
            "Received cloud: frame=%s, width=%d, "
            "height=%d, point_step=%d, dense=%s",
            message.header.frame_id,
            message.width,
            message.height,
            message.point_step,
            message.is_dense,
        )

        rospy.loginfo(
            "Point fields: %s",
            [field.name for field in message.fields],
        )

        raw_points = self.pointcloud_to_numpy(
            message
        )

        raw_count = raw_points.shape[0]

        finite_mask = np.isfinite(
            raw_points
        ).all(axis=1)

        finite_points = raw_points[finite_mask]

        depth_mask = (
            (finite_points[:, 2] >= self.min_depth)
            & (finite_points[:, 2] <= self.max_depth)
        )

        valid_points = finite_points[depth_mask]

        if self.sample_stride > 1:
            valid_points = valid_points[
                ::self.sample_stride
            ]

        valid_points = np.ascontiguousarray(
            valid_points,
            dtype=np.float32,
        )

        if valid_points.shape[0] == 0:
            raise RuntimeError(
                "No valid points remain after filtering."
            )

        minimum = valid_points.min(axis=0)
        maximum = valid_points.max(axis=0)
        mean = valid_points.mean(axis=0)

        rospy.loginfo(
            "Raw points: %d",
            raw_count,
        )

        rospy.loginfo(
            "Finite points: %d",
            finite_points.shape[0],
        )

        rospy.loginfo(
            "Valid depth-filtered points: %d",
            valid_points.shape[0],
        )

        rospy.loginfo(
            "XYZ minimum: [%.4f, %.4f, %.4f] m",
            minimum[0],
            minimum[1],
            minimum[2],
        )

        rospy.loginfo(
            "XYZ maximum: [%.4f, %.4f, %.4f] m",
            maximum[0],
            maximum[1],
            maximum[2],
        )

        rospy.loginfo(
            "XYZ mean: [%.4f, %.4f, %.4f] m",
            mean[0],
            mean[1],
            mean[2],
        )

        output_directory = os.path.dirname(
            self.output_path
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        np.savez_compressed(
            self.output_path,
            points=valid_points,
            frame_id=np.array(
                message.header.frame_id
            ),
            stamp=np.array(
                message.header.stamp.to_sec(),
                dtype=np.float64,
            ),
            source_topic=np.array(
                self.point_cloud_topic
            ),
            min_depth=np.array(
                self.min_depth,
                dtype=np.float32,
            ),
            max_depth=np.array(
                self.max_depth,
                dtype=np.float32,
            ),
        )

        rospy.loginfo(
            "Saved AnyGrasp-ready point cloud to: %s",
            self.output_path,
        )

        rospy.loginfo(
            "Saved points shape=%s, dtype=%s",
            valid_points.shape,
            valid_points.dtype,
        )

        rospy.loginfo(
            "========== POINT CLOUD DRY RUN COMPLETED =========="
        )


def main():
    rospy.init_node(
        "anygrasp_pointcloud_dryrun_node"
    )

    try:
        AnyGraspPointCloudDryRun().run()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "AnyGrasp point-cloud dry run failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()