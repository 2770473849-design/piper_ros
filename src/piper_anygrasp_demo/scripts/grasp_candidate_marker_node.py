#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys

import numpy as np
import rospy

from geometry_msgs.msg import Point
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray

from piper_anygrasp_demo.msg import GraspCandidateArray


class GraspCandidateMarkerNode:
    """Visualize filtered grasp candidates as coordinate axes in RViz."""

    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/filtered_grasp_candidates",
        )
        self.output_topic = rospy.get_param(
            "~output_topic",
            "/grasp_candidate_markers",
        )

        self.axis_length = float(
            rospy.get_param("~axis_length", 0.045)
        )
        self.shaft_diameter = float(
            rospy.get_param("~shaft_diameter", 0.003)
        )
        self.head_diameter = float(
            rospy.get_param("~head_diameter", 0.007)
        )
        self.head_length = float(
            rospy.get_param("~head_length", 0.012)
        )
        self.center_size = float(
            rospy.get_param("~center_size", 0.009)
        )
        self.text_height = float(
            rospy.get_param("~text_height", 0.014)
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            MarkerArray,
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
            "Grasp candidate marker node ready: %s -> %s",
            self.input_topic,
            self.output_topic,
        )

    @staticmethod
    def normalize_quaternion(candidate):
        orientation = candidate.pose.orientation

        quaternion = np.array(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ],
            dtype=np.float64,
        )

        norm = float(np.linalg.norm(quaternion))

        if not math.isfinite(norm) or norm < 1.0e-8:
            raise ValueError(
                "Candidate {} has invalid quaternion.".format(
                    candidate.id
                )
            )

        return quaternion / norm

    @staticmethod
    def make_point(x, y, z):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point

    @staticmethod
    def set_color(marker, red, green, blue, alpha=1.0):
        marker.color.r = float(red)
        marker.color.g = float(green)
        marker.color.b = float(blue)
        marker.color.a = float(alpha)

    def make_axis_marker(
        self,
        frame_id,
        marker_id,
        namespace,
        origin,
        direction,
        color,
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time(0)

        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.points.append(
            self.make_point(
                origin[0],
                origin[1],
                origin[2],
            )
        )
        marker.points.append(
            self.make_point(
                origin[0]
                + self.axis_length * direction[0],
                origin[1]
                + self.axis_length * direction[1],
                origin[2]
                + self.axis_length * direction[2],
            )
        )

        # 当 ARROW 使用 points 定义时：
        # scale.x 是杆直径，scale.y 是箭头直径，
        # scale.z 是箭头长度。
        marker.scale.x = self.shaft_diameter
        marker.scale.y = self.head_diameter
        marker.scale.z = self.head_length

        self.set_color(
            marker,
            color[0],
            color[1],
            color[2],
        )

        marker.lifetime = rospy.Duration(0.0)

        return marker

    def make_center_marker(
        self,
        frame_id,
        marker_id,
        origin,
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time(0)

        marker.ns = "grasp_candidate_centers"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(origin[0])
        marker.pose.position.y = float(origin[1])
        marker.pose.position.z = float(origin[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.center_size
        marker.scale.y = self.center_size
        marker.scale.z = self.center_size

        # 黄色抓取中心。
        self.set_color(marker, 1.0, 0.85, 0.0)

        marker.lifetime = rospy.Duration(0.0)

        return marker

    def make_text_marker(
        self,
        frame_id,
        marker_id,
        candidate,
        rank,
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time(0)

        marker.ns = "grasp_candidate_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = (
            candidate.pose.position.x
        )
        marker.pose.position.y = (
            candidate.pose.position.y
        )

        # 候选位姿可能重合，文字按排名向上错开。
        marker.pose.position.z = (
            candidate.pose.position.z
            + 0.030
            + 0.018 * rank
        )
        marker.pose.orientation.w = 1.0

        marker.scale.z = self.text_height

        marker.text = (
            "Rank {} | ID {} | score {:.2f} | width {:.1f} mm"
        ).format(
            rank + 1,
            candidate.id,
            candidate.score,
            candidate.width * 1000.0,
        )

        self.set_color(marker, 1.0, 1.0, 1.0)

        marker.lifetime = rospy.Duration(0.0)

        return marker

    @staticmethod
    def make_delete_all_marker(frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time(0)
        marker.action = Marker.DELETEALL
        return marker

    def candidates_callback(self, message):
        frame_id = message.header.frame_id.strip()

        if not frame_id:
            rospy.logerr(
                "Cannot visualize candidates: empty frame_id."
            )
            return

        marker_array = MarkerArray()
        marker_array.markers.append(
            self.make_delete_all_marker(frame_id)
        )

        rospy.loginfo(
            "Visualizing %d grasp candidates in frame '%s'.",
            len(message.candidates),
            frame_id,
        )

        for rank, candidate in enumerate(
            message.candidates
        ):
            try:
                quaternion = self.normalize_quaternion(
                    candidate
                )

            except ValueError as error:
                rospy.logwarn("%s", error)
                continue

            rotation = quaternion_matrix(
                quaternion
            )[:3, :3]

            # 旋转矩阵三列分别是：
            # 局部 +X、局部 +Y、局部 +Z 在 frame_id 下的方向。
            local_x = rotation[:, 0]
            local_y = rotation[:, 1]
            local_z = rotation[:, 2]

            origin = np.array(
                [
                    candidate.pose.position.x,
                    candidate.pose.position.y,
                    candidate.pose.position.z,
                ],
                dtype=np.float64,
            )

            marker_base_id = rank * 10

            marker_array.markers.append(
                self.make_center_marker(
                    frame_id,
                    marker_base_id,
                    origin,
                )
            )

            # 红：局部 +X。
            marker_array.markers.append(
                self.make_axis_marker(
                    frame_id,
                    marker_base_id + 1,
                    "grasp_candidate_x_axes",
                    origin,
                    local_x,
                    (1.0, 0.0, 0.0),
                )
            )

            # 绿：局部 +Y。
            marker_array.markers.append(
                self.make_axis_marker(
                    frame_id,
                    marker_base_id + 2,
                    "grasp_candidate_y_axes",
                    origin,
                    local_y,
                    (0.0, 1.0, 0.0),
                )
            )

            # 蓝：局部 +Z，即接近方向。
            marker_array.markers.append(
                self.make_axis_marker(
                    frame_id,
                    marker_base_id + 3,
                    "grasp_candidate_z_axes",
                    origin,
                    local_z,
                    (0.0, 0.35, 1.0),
                )
            )

            marker_array.markers.append(
                self.make_text_marker(
                    frame_id,
                    marker_base_id + 4,
                    candidate,
                    rank,
                )
            )

            rospy.loginfo(
                "Candidate %d marker: "
                "score=%.3f, local +Z=[%.3f, %.3f, %.3f]",
                candidate.id,
                candidate.score,
                local_z[0],
                local_z[1],
                local_z[2],
            )

        self.publisher.publish(marker_array)

        rospy.loginfo(
            "Published %d RViz markers on %s.",
            len(marker_array.markers),
            self.output_topic,
        )


def main():
    rospy.init_node(
        "grasp_candidate_marker_node"
    )

    try:
        GraspCandidateMarkerNode()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "Grasp candidate marker node failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()