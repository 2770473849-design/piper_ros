#!/usr/bin/env python3

import copy
import math

import rospy
import tf2_ros
import tf2_geometry_msgs  # 注册 PoseStamped 的 TF2 转换支持

from geometry_msgs.msg import PoseStamped


class GraspPoseTransformer:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/detected_grasp_pose",
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/detected_grasp_pose_in_planning_frame",
        )

        self.target_frame = rospy.get_param(
            "~target_frame",
            "dummy_link",
        )

        self.once = rospy.get_param(
            "~once",
            True,
        )

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(10.0)
        )

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for grasp pose: %s",
            self.input_topic,
        )
        rospy.loginfo(
            "Target frame: %s",
            self.target_frame,
        )
        rospy.loginfo(
            "Transformed output topic: %s",
            self.output_topic,
        )

    @staticmethod
    def validate_and_normalize(message):
        if not message.header.frame_id.strip():
            raise ValueError(
                "header.frame_id is empty"
            )

        position = message.pose.position
        orientation = message.pose.orientation

        values = [
            position.x,
            position.y,
            position.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "pose contains NaN or infinity"
            )

        quaternion_norm = math.sqrt(
            orientation.x ** 2
            + orientation.y ** 2
            + orientation.z ** 2
            + orientation.w ** 2
        )

        if quaternion_norm < 1e-8:
            raise ValueError(
                "quaternion norm is zero"
            )

        normalized = copy.deepcopy(message)

        normalized.pose.orientation.x /= quaternion_norm
        normalized.pose.orientation.y /= quaternion_norm
        normalized.pose.orientation.z /= quaternion_norm
        normalized.pose.orientation.w /= quaternion_norm

        return normalized

    def pose_callback(self, message):
        try:
            normalized_pose = self.validate_and_normalize(
                message
            )
        except ValueError as error:
            rospy.logerr(
                "Rejected grasp pose: %s",
                str(error),
            )
            return

        source_frame = normalized_pose.header.frame_id

        rospy.loginfo(
            "Received grasp pose in frame '%s': "
            "position=[%.6f, %.6f, %.6f]",
            source_frame,
            normalized_pose.pose.position.x,
            normalized_pose.pose.position.y,
            normalized_pose.pose.position.z,
        )

        try:
            transformed_pose = self.tf_buffer.transform(
                normalized_pose,
                self.target_frame,
                rospy.Duration(1.0),
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logerr(
                "TF transform failed: %s -> %s: %s",
                source_frame,
                self.target_frame,
                str(error),
            )
            return

        self.publisher.publish(
            transformed_pose
        )

        rospy.loginfo(
            "Transformed grasp pose in frame '%s': "
            "position=[%.6f, %.6f, %.6f]",
            transformed_pose.header.frame_id,
            transformed_pose.pose.position.x,
            transformed_pose.pose.position.y,
            transformed_pose.pose.position.z,
        )

        rospy.loginfo(
            "Transformed quaternion: "
            "[%.6f, %.6f, %.6f, %.6f]",
            transformed_pose.pose.orientation.x,
            transformed_pose.pose.orientation.y,
            transformed_pose.pose.orientation.z,
            transformed_pose.pose.orientation.w,
        )

        rospy.loginfo(
            "Grasp pose TF conversion completed successfully."
        )

        if self.once:
            rospy.signal_shutdown(
                "One grasp pose transformed."
            )


def main():
    rospy.init_node(
        "grasp_pose_transform_node"
    )

    GraspPoseTransformer()

    rospy.spin()


if __name__ == "__main__":
    main()
