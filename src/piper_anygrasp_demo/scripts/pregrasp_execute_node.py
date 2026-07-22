#!/usr/bin/env python3

import math

import rospy

from geometry_msgs.msg import PoseStamped
from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


DEFAULT_INPUT_TOPIC = (
    "/detected_grasp_pose_in_planning_frame"
)
ENDPOSE_SERVICE = "/joint_moveit_ctrl_endpose"


class PregraspExecutor:
    def __init__(self):
        self.input_topic = rospy.get_param(
            "~input_topic",
            DEFAULT_INPUT_TOPIC,
        )

        self.planning_frame = rospy.get_param(
            "~planning_frame",
            "dummy_link",
        )

        self.approach_distance = float(
            rospy.get_param(
                "~approach_distance",
                0.01,
            )
        )

        self.velocity = float(
            rospy.get_param("~velocity", 0.05)
        )

        self.acceleration = float(
            rospy.get_param("~acceleration", 0.05)
        )

        if not 0.0 < self.approach_distance <= 0.20:
            raise ValueError(
                "~approach_distance must be in (0, 0.20]."
            )

        if not 0.0 < self.velocity <= 1.0:
            raise ValueError(
                "~velocity must be in (0, 1]."
            )

        if not 0.0 < self.acceleration <= 1.0:
            raise ValueError(
                "~acceleration must be in (0, 1]."
            )

        self.executing = False
        self.completed = False

        rospy.loginfo(
            "Waiting for service: %s",
            ENDPOSE_SERVICE,
        )

        rospy.wait_for_service(
            ENDPOSE_SERVICE,
            timeout=10.0,
        )

        self.endpose_service = rospy.ServiceProxy(
            ENDPOSE_SERVICE,
            JointMoveitCtrl,
        )

        self.subscriber = rospy.Subscriber(
            self.input_topic,
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Waiting for transformed grasp pose: %s",
            self.input_topic,
        )

        rospy.loginfo(
            "Planning frame: %s",
            self.planning_frame,
        )

        rospy.loginfo(
            "Pregrasp offset along local -Z: %.4f m",
            self.approach_distance,
        )

    @staticmethod
    def normalize_quaternion(orientation):
        values = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "quaternion contains NaN or infinity"
            )

        norm = math.sqrt(
            sum(value * value for value in values)
        )

        if norm < 1e-8:
            raise ValueError(
                "quaternion norm is zero"
            )

        return [
            value / norm
            for value in values
        ]

    @staticmethod
    def local_positive_z_axis(quaternion):
        qx, qy, qz, qw = quaternion

        # 四元数旋转矩阵的第三列：
        # 表示夹爪局部 +Z 轴在规划坐标系中的方向。
        axis_x = 2.0 * (
            qx * qz + qw * qy
        )

        axis_y = 2.0 * (
            qy * qz - qw * qx
        )

        axis_z = (
            1.0
            - 2.0 * (
                qx * qx
                + qy * qy
            )
        )

        axis_norm = math.sqrt(
            axis_x * axis_x
            + axis_y * axis_y
            + axis_z * axis_z
        )

        if axis_norm < 1e-8:
            raise ValueError(
                "computed approach axis is invalid"
            )

        return [
            axis_x / axis_norm,
            axis_y / axis_norm,
            axis_z / axis_norm,
        ]

    def pose_callback(self, message):
        if self.executing or self.completed:
            return

        self.executing = True

        try:
            frame_id = message.header.frame_id.strip()

            if frame_id != self.planning_frame:
                raise ValueError(
                    "expected frame '{}', received '{}'".format(
                        self.planning_frame,
                        frame_id,
                    )
                )

            position = message.pose.position

            position_values = [
                position.x,
                position.y,
                position.z,
            ]

            if not all(
                math.isfinite(value)
                for value in position_values
            ):
                raise ValueError(
                    "position contains NaN or infinity"
                )

            quaternion = self.normalize_quaternion(
                message.pose.orientation
            )

            approach_axis = (
                self.local_positive_z_axis(
                    quaternion
                )
            )

            pregrasp_position = [
                position.x
                - self.approach_distance
                * approach_axis[0],

                position.y
                - self.approach_distance
                * approach_axis[1],

                position.z
                - self.approach_distance
                * approach_axis[2],
            ]

            rospy.loginfo(
                "===== Grasp to pregrasp ====="
            )

            rospy.loginfo(
                "Grasp position: "
                "[%.6f, %.6f, %.6f]",
                position.x,
                position.y,
                position.z,
            )

            rospy.loginfo(
                "Local +Z approach axis: "
                "[%.6f, %.6f, %.6f]",
                approach_axis[0],
                approach_axis[1],
                approach_axis[2],
            )

            rospy.loginfo(
                "Pregrasp position: "
                "[%.6f, %.6f, %.6f]",
                pregrasp_position[0],
                pregrasp_position[1],
                pregrasp_position[2],
            )

            request = JointMoveitCtrlRequest()

            request.joint_states = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            request.gripper = 0.0

            request.joint_endpose = (
                pregrasp_position
                + quaternion
            )

            request.max_velocity = self.velocity
            request.max_acceleration = (
                self.acceleration
            )

            rospy.loginfo(
                "Sending pregrasp target..."
            )

            response = self.endpose_service(
                request
            )

            if not response.status:
                rospy.logerr(
                    "Pregrasp movement failed, "
                    "error_code=%d",
                    response.error_code,
                )
                return

            self.completed = True

            rospy.loginfo(
                "Pregrasp movement completed successfully."
            )

            rospy.signal_shutdown(
                "One pregrasp target executed."
            )

        except (
            ValueError,
            rospy.ServiceException,
        ) as error:
            rospy.logerr(
                "Pregrasp execution failed: %s",
                str(error),
            )

        finally:
            self.executing = False


def main():
    rospy.init_node(
        "pregrasp_execute_node"
    )

    try:
        PregraspExecutor()
        rospy.spin()

    except (
        ValueError,
        rospy.ROSException,
    ) as error:
        rospy.logerr(
            "Node initialization failed: %s",
            str(error),
        )


if __name__ == "__main__":
    main()
