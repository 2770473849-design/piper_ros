#!/usr/bin/env python3

import math

import rospy

from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


ENDPOSE_SERVICE = "/joint_moveit_ctrl_endpose"


def parse_numeric_list(value, expected_length, parameter_name):
    if not isinstance(value, list):
        raise ValueError(
            "{} must be a list.".format(parameter_name)
        )

    if len(value) != expected_length:
        raise ValueError(
            "{} must contain exactly {} values.".format(
                parameter_name,
                expected_length,
            )
        )

    result = [float(item) for item in value]

    if not all(math.isfinite(item) for item in result):
        raise ValueError(
            "{} contains a non-finite value.".format(
                parameter_name
            )
        )

    return result


def main():
    rospy.init_node("arm_absolute_pose_node")

    try:
        position = parse_numeric_list(
            rospy.get_param("~position"),
            3,
            "~position",
        )

        orientation = parse_numeric_list(
            rospy.get_param("~orientation"),
            4,
            "~orientation",
        )

        velocity = float(
            rospy.get_param("~velocity", 0.05)
        )
        acceleration = float(
            rospy.get_param("~acceleration", 0.05)
        )

    except (KeyError, TypeError, ValueError) as error:
        rospy.logerr(
            "Invalid parameter: %s",
            str(error),
        )
        rospy.logerr(
            "Required format: "
            "_position:='[x, y, z]' "
            "_orientation:='[qx, qy, qz, qw]'"
        )
        return

    quaternion_norm = math.sqrt(
        sum(value * value for value in orientation)
    )

    if quaternion_norm < 1e-8:
        rospy.logerr(
            "Quaternion norm is zero."
        )
        return

    # 在客户端先归一化一次。
    orientation = [
        value / quaternion_norm
        for value in orientation
    ]

    if not 0.0 < velocity <= 1.0:
        rospy.logerr(
            "~velocity must be in the range (0, 1]."
        )
        return

    if not 0.0 < acceleration <= 1.0:
        rospy.logerr(
            "~acceleration must be in the range (0, 1]."
        )
        return

    rospy.loginfo(
        "Target frame: dummy_link"
    )

    rospy.loginfo(
        "Target position: "
        "x=%.6f, y=%.6f, z=%.6f",
        position[0],
        position[1],
        position[2],
    )

    rospy.loginfo(
        "Target orientation: "
        "qx=%.6f, qy=%.6f, qz=%.6f, qw=%.6f",
        orientation[0],
        orientation[1],
        orientation[2],
        orientation[3],
    )

    rospy.loginfo(
        "Waiting for service: %s",
        ENDPOSE_SERVICE,
    )

    try:
        rospy.wait_for_service(
            ENDPOSE_SERVICE,
            timeout=10.0,
        )

        service = rospy.ServiceProxy(
            ENDPOSE_SERVICE,
            JointMoveitCtrl,
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
            position + orientation
        )

        request.max_velocity = velocity
        request.max_acceleration = acceleration

        rospy.loginfo(
            "Sending absolute pose target..."
        )

        response = service(request)

    except rospy.ROSException as error:
        rospy.logerr(
            "Service unavailable: %s",
            str(error),
        )
        return

    except rospy.ServiceException as error:
        rospy.logerr(
            "Service call failed: %s",
            str(error),
        )
        return

    if not response.status:
        rospy.logerr(
            "Absolute pose movement failed, "
            "error_code=%d",
            response.error_code,
        )
        return

    rospy.loginfo(
        "Absolute pose movement completed successfully."
    )


if __name__ == "__main__":
    main()
