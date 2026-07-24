#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from tf.transformations import quaternion_matrix


class PointCloudGraspPosePublisher:
    def __init__(self):
        self.cloud_topic = rospy.get_param(
            "~cloud_topic",
            "/hand_camera/depth/points",
        )
        self.output_topic = rospy.get_param(
            "~output_topic",
            "/grasp_tcp_pose",
        )
        self.target_frame = rospy.get_param(
            "~target_frame",
            "base_link",
        )

        self.sample_step = max(
            1,
            int(rospy.get_param("~sample_step", 2)),
        )

        # 当前红色方块所在的抓取区域。
        self.x_min = float(rospy.get_param("~x_min", 0.20))
        self.x_max = float(rospy.get_param("~x_max", 0.30))
        self.y_min = float(rospy.get_param("~y_min", -0.06))
        self.y_max = float(rospy.get_param("~y_max", 0.06))
        self.z_min = float(rospy.get_param("~z_min", 0.008))
        self.z_max = float(rospy.get_param("~z_max", 0.060))

        self.top_percentile = float(
            rospy.get_param("~top_percentile", 90.0)
        )
        self.top_band = float(
            rospy.get_param("~top_band", 0.006)
        )

        # 方块边长为 0.03 m，因此半高为 0.015 m。
        self.cube_half_height = float(
            rospy.get_param("~cube_half_height", 0.015)
        )

        # TCP 位于方块顶面上方 0.015 m。
        self.tcp_clearance = float(
            rospy.get_param("~tcp_clearance", 0.015)
        )

        self.min_roi_points = int(
            rospy.get_param("~min_roi_points", 50)
        )
        self.min_top_points = int(
            rospy.get_param("~min_top_points", 20)
        )

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(10.0)
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        self.pose_publisher = rospy.Publisher(
            self.output_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )

    def wait_for_cloud(self):
        rospy.loginfo(
            "Waiting for point cloud: %s",
            self.cloud_topic,
        )

        return rospy.wait_for_message(
            self.cloud_topic,
            PointCloud2,
            timeout=15.0,
        )

    def transform_points(self, cloud, points):
        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            cloud.header.frame_id,
            rospy.Time(0),
            rospy.Duration(3.0),
        )

        rotation = transform.transform.rotation

        matrix = quaternion_matrix(
            [
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ]
        )

        translation = transform.transform.translation
        matrix[:3, 3] = [
            translation.x,
            translation.y,
            translation.z,
        ]

        homogeneous_points = np.column_stack(
            (
                points,
                np.ones(points.shape[0], dtype=np.float64),
            )
        )

        transformed = (
            matrix @ homogeneous_points.T
        ).T[:, :3]

        return transformed

    def detect_grasp_pose(self, cloud):
        uvs = [
            (u, v)
            for v in range(
                0,
                cloud.height,
                self.sample_step,
            )
            for u in range(
                0,
                cloud.width,
                self.sample_step,
            )
        ]

        points = np.asarray(
            list(
                pc2.read_points(
                    cloud,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                    uvs=uvs,
                )
            ),
            dtype=np.float64,
        )

        if points.size == 0:
            raise RuntimeError(
                "Point cloud contains no valid XYZ points."
            )

        base_points = self.transform_points(
            cloud,
            points,
        )

        roi_mask = (
            (base_points[:, 0] >= self.x_min)
            & (base_points[:, 0] <= self.x_max)
            & (base_points[:, 1] >= self.y_min)
            & (base_points[:, 1] <= self.y_max)
            & (base_points[:, 2] >= self.z_min)
            & (base_points[:, 2] <= self.z_max)
        )

        roi_points = base_points[roi_mask]

        if roi_points.shape[0] < self.min_roi_points:
            raise RuntimeError(
                "Too few ROI points: {} < {}".format(
                    roi_points.shape[0],
                    self.min_roi_points,
                )
            )

        top_z = float(
            np.percentile(
                roi_points[:, 2],
                self.top_percentile,
            )
        )

        top_points = roi_points[
            roi_points[:, 2]
            >= top_z - self.top_band
        ]

        if top_points.shape[0] < self.min_top_points:
            raise RuntimeError(
                "Too few top-surface points: {} < {}".format(
                    top_points.shape[0],
                    self.min_top_points,
                )
            )

        center_xy = np.median(
            top_points[:, :2],
            axis=0,
        )

        center_z = (
            top_z - self.cube_half_height
        )
        grasp_z = (
            top_z + self.tcp_clearance
        )

        grasp_pose = PoseStamped()
        grasp_pose.header.stamp = rospy.Time(0)
        grasp_pose.header.frame_id = self.target_frame

        grasp_pose.pose.position.x = float(
            center_xy[0]
        )
        grasp_pose.pose.position.y = float(
            center_xy[1]
        )
        grasp_pose.pose.position.z = float(
            grasp_z
        )

        # 保持已经稳定验证过的俯视抓取姿态。
        grasp_pose.pose.orientation.x = 0.0
        grasp_pose.pose.orientation.y = 1.0
        grasp_pose.pose.orientation.z = 0.0
        grasp_pose.pose.orientation.w = 0.0

        rospy.loginfo(
            "Valid sampled points: %d",
            points.shape[0],
        )
        rospy.loginfo(
            "ROI candidate points: %d",
            roi_points.shape[0],
        )
        rospy.loginfo(
            "Top-surface points: %d",
            top_points.shape[0],
        )
        rospy.loginfo(
            "Detected top height: %.4f m",
            top_z,
        )
        rospy.loginfo(
            "Estimated cube center: "
            "[%.4f, %.4f, %.4f]",
            center_xy[0],
            center_xy[1],
            center_z,
        )
        rospy.loginfo(
            "Publishing grasp TCP in %s: "
            "[%.4f, %.4f, %.4f]",
            self.target_frame,
            grasp_pose.pose.position.x,
            grasp_pose.pose.position.y,
            grasp_pose.pose.position.z,
        )

        return grasp_pose

    def run(self):
        # 给 TF Listener 一点时间建立缓存。
        rospy.sleep(1.0)

        cloud = self.wait_for_cloud()
        grasp_pose = self.detect_grasp_pose(cloud)

        self.pose_publisher.publish(grasp_pose)

        rospy.loginfo(
            "Published latched grasp pose on %s.",
            self.output_topic,
        )
        rospy.loginfo(
            "Node will stay alive for future subscribers."
        )

        rospy.spin()


def main():
    rospy.init_node(
        "pointcloud_grasp_pose_publisher"
    )

    try:
        node = PointCloudGraspPosePublisher()
        node.run()

    except (
        rospy.ROSException,
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
        RuntimeError,
    ) as error:
        rospy.logerr(
            "Point-cloud grasp detection failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()