#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped


class AnyGraspTcpAlignmentInspect:
    """Compare AnyGrasp centers/tips with the validated Piper TCP."""

    def __init__(self):
        self.input_path = os.path.expanduser(
            rospy.get_param(
                "~input_path",
                "~/anygrasp_ws/captures/"
                "anygrasp_offline_results_latest.npz",
            )
        )

        self.target_frame = rospy.get_param(
            "~target_frame",
            "base_link",
        )

        self.reference_tcp = np.array(
            [
                float(rospy.get_param("~reference_x", 0.25)),
                float(rospy.get_param("~reference_y", 0.00)),
                float(rospy.get_param("~reference_z", 0.045)),
            ],
            dtype=np.float64,
        )

        self.top_n = max(
            1,
            int(rospy.get_param("~top_n", 5)),
        )

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(10.0)
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

    def transform_point(
        self,
        point,
        source_frame,
    ):
        message = PointStamped()
        message.header.stamp = rospy.Time(0)
        message.header.frame_id = source_frame

        message.point.x = float(point[0])
        message.point.y = float(point[1])
        message.point.z = float(point[2])

        transformed = self.tf_buffer.transform(
            message,
            self.target_frame,
            rospy.Duration(3.0),
        )

        return np.array(
            [
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            ],
            dtype=np.float64,
        )

    def run(self):
        if not os.path.isfile(self.input_path):
            raise RuntimeError(
                "Inference result does not exist: {}".format(
                    self.input_path
                )
            )

        data = np.load(self.input_path)

        source_frame = str(data["frame_id"])
        translations = data["translations"]
        rotations = data["anygrasp_rotation_matrices"]
        depths = data["depths"]
        scores = data["scores"]

        count = min(
            self.top_n,
            len(scores),
        )

        rospy.loginfo(
            "========== ANYGRASP TCP ALIGNMENT =========="
        )

        rospy.loginfo(
            "Source frame: %s, target frame: %s",
            source_frame,
            self.target_frame,
        )

        rospy.loginfo(
            "Validated Piper TCP reference: "
            "[%.4f, %.4f, %.4f]",
            self.reference_tcp[0],
            self.reference_tcp[1],
            self.reference_tcp[2],
        )

        rospy.sleep(1.0)

        for index in range(count):
            center_camera = np.asarray(
                translations[index],
                dtype=np.float64,
            )

            approach_camera = np.asarray(
                rotations[index][:, 0],
                dtype=np.float64,
            )

            depth = float(depths[index])

            # AnyGrasp文档定义的夹爪尖端。
            tip_camera = (
                center_camera
                + depth * approach_camera
            )

            center_base = self.transform_point(
                center_camera,
                source_frame,
            )

            tip_base = self.transform_point(
                tip_camera,
                source_frame,
            )

            # 从AnyGrasp中心沿接近方向反向移动15 mm。
            offset_15_camera = (
                center_camera
                - 0.015 * approach_camera
            )

            offset_15_base = self.transform_point(
                offset_15_camera,
                source_frame,
            )

            center_error = float(
                np.linalg.norm(
                    center_base
                    - self.reference_tcp
                )
            )

            tip_error = float(
                np.linalg.norm(
                    tip_base
                    - self.reference_tcp
                )
            )

            offset_error = float(
                np.linalg.norm(
                    offset_15_base
                    - self.reference_tcp
                )
            )

            rospy.loginfo(
                "---------- Candidate %d, score=%.3f ----------",
                index,
                float(scores[index]),
            )

            rospy.loginfo(
                "AnyGrasp center in base_link: "
                "[%.4f, %.4f, %.4f], error=%.4f m",
                center_base[0],
                center_base[1],
                center_base[2],
                center_error,
            )

            rospy.loginfo(
                "AnyGrasp tip in base_link: "
                "[%.4f, %.4f, %.4f], error=%.4f m",
                tip_base[0],
                tip_base[1],
                tip_base[2],
                tip_error,
            )

            rospy.loginfo(
                "Center - 15mm approach in base_link: "
                "[%.4f, %.4f, %.4f], error=%.4f m",
                offset_15_base[0],
                offset_15_base[1],
                offset_15_base[2],
                offset_error,
            )

        rospy.loginfo(
            "========== TCP ALIGNMENT INSPECTION COMPLETED =========="
        )


def main():
    rospy.init_node(
        "anygrasp_tcp_alignment_inspect_node"
    )

    try:
        AnyGraspTcpAlignmentInspect().run()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "AnyGrasp TCP alignment inspection failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()