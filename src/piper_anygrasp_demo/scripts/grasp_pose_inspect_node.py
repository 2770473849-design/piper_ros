#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseStamped


DEFAULT_TOPIC = "/detected_grasp_pose"


class GraspPoseInspector:
    def __init__(self):
        self.topic = rospy.get_param(
            "~topic",
            DEFAULT_TOPIC,
        )
        self.once = rospy.get_param(
            "~once",
            True,
        )

        self.received_count = 0

        self.subscriber = rospy.Subscriber(
            self.topic,
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for grasp pose on topic: %s",
            self.topic,
        )

    def pose_callback(self, message):
        self.received_count += 1

        frame_id = message.header.frame_id.strip()

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

        if not frame_id:
            rospy.logerr(
                "Rejected grasp pose: header.frame_id is empty."
            )
            return

        if not all(math.isfinite(value) for value in values):
            rospy.logerr(
                "Rejected grasp pose: pose contains NaN or infinity."
            )
            return

        quaternion_norm = math.sqrt(
            orientation.x ** 2
            + orientation.y ** 2
            + orientation.z ** 2
            + orientation.w ** 2
        )

        if quaternion_norm < 1e-8:
            rospy.logerr(
                "Rejected grasp pose: quaternion norm is zero."
            )
            return

        normalized_quaternion = [
            orientation.x / quaternion_norm,
            orientation.y / quaternion_norm,
            orientation.z / quaternion_norm,
            orientation.w / quaternion_norm,
        ]

        rospy.loginfo(
            "===== Received grasp pose #%d =====",
            self.received_count,
        )

        rospy.loginfo(
            "Frame: %s",
            frame_id,
        )

        rospy.loginfo(
            "Timestamp: %.9f",
            message.header.stamp.to_sec(),
        )

        rospy.loginfo(
            "Position: x=%.6f, y=%.6f, z=%.6f",
            position.x,
            position.y,
            position.z,
        )

        rospy.loginfo(
            "Original quaternion: "
            "qx=%.6f, qy=%.6f, qz=%.6f, qw=%.6f",
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

        rospy.loginfo(
            "Quaternion norm: %.9f",
            quaternion_norm,
        )

        rospy.loginfo(
            "Normalized quaternion: "
            "qx=%.6f, qy=%.6f, qz=%.6f, qw=%.6f",
            normalized_quaternion[0],
            normalized_quaternion[1],
            normalized_quaternion[2],
            normalized_quaternion[3],
        )

        rospy.loginfo(
            "Grasp pose passed basic validation."
        )

        if self.once:
            rospy.signal_shutdown(
                "One valid grasp pose received."
            )


def main():
    rospy.init_node("grasp_pose_inspect_node")

    GraspPoseInspector()

    rospy.spin()


if __name__ == "__main__":
    main()
