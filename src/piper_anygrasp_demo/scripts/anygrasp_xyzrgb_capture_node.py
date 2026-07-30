#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import rospy

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as point_cloud2


class AnyGraspXYZRGBCapture:
    """Capture one XYZRGB point cloud and segment the red target."""

    def __init__(self):
        self.point_cloud_topic = rospy.get_param(
            "~point_cloud_topic",
            "/hand_camera/depth/points",
        )

        self.timeout = float(
            rospy.get_param("~timeout", 30.0)
        )

        self.min_depth = float(
            rospy.get_param("~min_depth", 0.03)
        )

        self.max_depth = float(
            rospy.get_param("~max_depth", 2.0)
        )

        # 红色阈值：
        # R必须足够大，同时明显大于G、B。
        self.red_min = int(
            rospy.get_param("~red_min", 120)
        )

        self.red_dominance = int(
            rospy.get_param("~red_dominance", 50)
        )


        # Gazebo OpenNI点云中的红蓝通道与当前解码顺序相反。
        # 真实D435i接入后可通过参数设为false。
        self.swap_red_blue = bool(
            rospy.get_param(
                "~swap_red_blue",
                True,
            )
        )

        self.output_path = os.path.expanduser(
            rospy.get_param(
                "~output_path",
                "~/anygrasp_ws/captures/"
                "hand_camera_xyzrgb_latest.npz",
            )
        )

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
        """
        Decode packed RGB/RGBA values from PointCloud2.

        Gazebo/OpenNI usually stores rgb as FLOAT32 whose binary bits
        contain an RGB uint32 value.
        """

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

    def run(self):
        rospy.loginfo(
            "========== ANYGRASP XYZRGB CAPTURE =========="
        )

        rospy.loginfo(
            "Waiting for colored PointCloud2 on: %s",
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

        valid_mask = finite_mask & depth_mask

        points = np.ascontiguousarray(
            points[valid_mask],
            dtype=np.float32,
        )

        colors = np.ascontiguousarray(
            colors[valid_mask],
            dtype=np.uint8,
        )

        red = colors[:, 0].astype(np.int16)
        green = colors[:, 1].astype(np.int16)
        blue = colors[:, 2].astype(np.int16)

        red_mask = (
            (red >= self.red_min)
            & ((red - green) >= self.red_dominance)
            & ((red - blue) >= self.red_dominance)
        )

        red_points = points[red_mask]
        red_colors = colors[red_mask]

        rospy.loginfo(
            "Valid XYZRGB points: %d",
            points.shape[0],
        )

        rospy.loginfo(
            "Red target points: %d",
            red_points.shape[0],
        )

        if red_points.shape[0] == 0:
            raise RuntimeError(
                "No red target points were detected. "
                "Check the RGB image and color thresholds."
            )

        red_minimum = red_points.min(axis=0)
        red_maximum = red_points.max(axis=0)
        red_mean = red_points.mean(axis=0)
        red_median = np.median(
            red_points,
            axis=0,
        )

        mean_rgb = red_colors.mean(axis=0)

        rospy.loginfo(
            "Red XYZ minimum: [%.4f, %.4f, %.4f] m",
            red_minimum[0],
            red_minimum[1],
            red_minimum[2],
        )

        rospy.loginfo(
            "Red XYZ maximum: [%.4f, %.4f, %.4f] m",
            red_maximum[0],
            red_maximum[1],
            red_maximum[2],
        )

        rospy.loginfo(
            "Red XYZ mean: [%.4f, %.4f, %.4f] m",
            red_mean[0],
            red_mean[1],
            red_mean[2],
        )

        rospy.loginfo(
            "Red XYZ median: [%.4f, %.4f, %.4f] m",
            red_median[0],
            red_median[1],
            red_median[2],
        )

        rospy.loginfo(
            "Detected target mean RGB: "
            "[%.1f, %.1f, %.1f]",
            mean_rgb[0],
            mean_rgb[1],
            mean_rgb[2],
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
            points=points,
            colors=colors,
            red_mask=red_mask,
            red_points=red_points,
            frame_id=np.array(
                message.header.frame_id
            ),
            stamp=np.array(
                message.header.stamp.to_sec(),
                dtype=np.float64,
            ),
            image_height=np.array(
                message.height,
                dtype=np.int32,
            ),
            image_width=np.array(
                message.width,
                dtype=np.int32,
            ),
            source_topic=np.array(
                self.point_cloud_topic
            ),
        )

        rospy.loginfo(
            "Saved XYZRGB capture to: %s",
            self.output_path,
        )

        rospy.loginfo(
            "points=%s, colors=%s, red_mask=%s",
            points.shape,
            colors.shape,
            red_mask.shape,
        )

        rospy.loginfo(
            "========== XYZRGB CAPTURE COMPLETED =========="
        )


def main():
    rospy.init_node(
        "anygrasp_xyzrgb_capture_node"
    )

    try:
        AnyGraspXYZRGBCapture().run()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "AnyGrasp XYZRGB capture failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()