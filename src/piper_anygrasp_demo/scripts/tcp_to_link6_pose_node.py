#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseStamped


class TcpToLink6PoseConverter:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/detected_grasp_tcp_pose_in_planning_frame",
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/detected_link6_grasp_pose_in_planning_frame",
        )

        self.planning_frame = rospy.get_param(
            "~planning_frame",
            "dummy_link",
        )

        self.tcp_offset = float(
            rospy.get_param(
                "~tcp_offset",
                0.09755,
            )
        )

        self.once = rospy.get_param(
            "~once",
            True,
        )

        if not 0.0 < self.tcp_offset <= 0.30:
            raise ValueError(
                "~tcp_offset must be in (0, 0.30]."
            )

        self.publisher = rospy.Publisher(
            self.output_topic,
            PoseStamped,
            queue_size=1,
            latch=False,
        )

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for grasp TCP pose: %s",
            self.input_topic,
        )

        rospy.loginfo(
            "Output link6 target topic: %s",
            self.output_topic,
        )

        rospy.loginfo(
            "TCP offset along link6 local +Z: %.5f m",
            self.tcp_offset,
        )

    @staticmethod
    def normalize_quaternion(orientation):
        quaternion = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]

        if not all(
            math.isfinite(value)
            for value in quaternion
        ):
            raise ValueError(
                "Quaternion contains NaN or infinity."
            )

        norm = math.sqrt(
            sum(value * value for value in quaternion)
        )

        if norm < 1e-8:
            raise ValueError(
                "Quaternion norm is zero."
            )

        return [
            value / norm
            for value in quaternion
        ]

    @staticmethod
    def local_positive_z_axis(quaternion):
        qx, qy, qz, qw = quaternion

        # 旋转矩阵第三列：
        # TCP/夹爪局部 +Z 轴在规划坐标系中的方向。
        axis = [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (
                qx * qx + qy * qy
            ),
        ]

        norm = math.sqrt(
            sum(value * value for value in axis)
        )

        if norm < 1e-8:
            raise ValueError(
                "Computed local +Z axis is invalid."
            )

        return [
            value / norm
            for value in axis
        ]

    def pose_callback(self, message):
        try:
            frame_id = message.header.frame_id.strip()

            if frame_id != self.planning_frame:
                raise ValueError(
                    "Expected frame '{}', received '{}'.".format(
                        self.planning_frame,
                        frame_id,
                    )
                )

            position = message.pose.position

            if not all(
                math.isfinite(value)
                for value in [
                    position.x,
                    position.y,
                    position.z,
                ]
            ):
                raise ValueError(
                    "Position contains NaN or infinity."
                )

            quaternion = self.normalize_quaternion(
                message.pose.orientation
            )

            local_z = self.local_positive_z_axis(
                quaternion
            )

            # T_world_link6 =
            # T_world_tcp × inverse(T_link6_tcp)
            #
            # 当前 T_link6_tcp 只有沿局部 +Z 的固定平移，
            # 所以 link6 位置 = TCP位置 - offset × 局部+Z。
            link6_pose = PoseStamped()

            link6_pose.header.stamp = rospy.Time.now()
            link6_pose.header.frame_id = self.planning_frame

            link6_pose.pose.position.x = (
                position.x
                - self.tcp_offset * local_z[0]
            )

            link6_pose.pose.position.y = (
                position.y
                - self.tcp_offset * local_z[1]
            )

            link6_pose.pose.position.z = (
                position.z
                - self.tcp_offset * local_z[2]
            )

            link6_pose.pose.orientation.x = quaternion[0]
            link6_pose.pose.orientation.y = quaternion[1]
            link6_pose.pose.orientation.z = quaternion[2]
            link6_pose.pose.orientation.w = quaternion[3]

            rospy.loginfo(
                "===== TCP target to link6 target ====="
            )

            rospy.loginfo(
                "TCP target position: "
                "[%.6f, %.6f, %.6f]",
                position.x,
                position.y,
                position.z,
            )

            rospy.loginfo(
                "TCP local +Z axis: "
                "[%.6f, %.6f, %.6f]",
                local_z[0],
                local_z[1],
                local_z[2],
            )

            rospy.loginfo(
                "TCP offset: %.6f m",
                self.tcp_offset,
            )

            rospy.loginfo(
                "Converted link6 target: "
                "[%.6f, %.6f, %.6f]",
                link6_pose.pose.position.x,
                link6_pose.pose.position.y,
                link6_pose.pose.position.z,
            )

            rospy.loginfo(
                "Converted quaternion: "
                "[%.6f, %.6f, %.6f, %.6f]",
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],
            )

            self.publisher.publish(
                link6_pose
            )

            rospy.loginfo(
                "TCP to link6 conversion completed successfully."
            )

            if self.once:
                rospy.sleep(0.2)
                rospy.signal_shutdown(
                    "One TCP target converted."
                )

        except ValueError as error:
            rospy.logerr(
                "TCP to link6 conversion failed: %s",
                str(error),
            )


def main():
    rospy.init_node(
        "tcp_to_link6_pose_node"
    )

    try:
        TcpToLink6PoseConverter()
        rospy.spin()

    except Exception as error:
        rospy.logerr(
            "Node initialization failed: %s",
            str(error),
        )


if __name__ == "__main__":
    main()
