#!/usr/bin/env python3

import rospy

from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


GRIPPER_SERVICE = "/joint_moveit_ctrl_gripper"


def main():
    rospy.init_node("gripper_command_node")

    # 单侧夹爪开合量，合法范围为 0～0.035 m。
    opening = rospy.get_param("~opening", 0.02)

    if opening < 0.0 or opening > 0.035:
        rospy.logerr(
            "Invalid opening %.4f m; expected 0.0～0.035 m.",
            opening,
        )
        return

    rospy.loginfo(
        "Waiting for service: %s",
        GRIPPER_SERVICE,
    )

    try:
        rospy.wait_for_service(
            GRIPPER_SERVICE,
            timeout=10.0,
        )
    except rospy.ROSException:
        rospy.logerr(
            "Service is unavailable: %s",
            GRIPPER_SERVICE,
        )
        return

    request = JointMoveitCtrlRequest()

    # gripper 服务只使用这个字段。
    request.gripper = opening

    # 以下字段属于服务消息的固定结构，
    # 在 gripper 服务中不会实际使用。
    request.joint_states = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    request.joint_endpose = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    request.max_velocity = 0.1
    request.max_acceleration = 0.1

    try:
        gripper_service = rospy.ServiceProxy(
            GRIPPER_SERVICE,
            JointMoveitCtrl,
        )

        rospy.loginfo(
            "Sending gripper opening target: %.4f m",
            opening,
        )

        response = gripper_service(request)

    except rospy.ServiceException as error:
        rospy.logerr(
            "Service call failed: %s",
            str(error),
        )
        return

    if response.status:
        rospy.loginfo(
            "Gripper command completed successfully."
        )
    else:
        rospy.logerr(
            "Gripper command failed, error_code=%d",
            response.error_code,
        )


if __name__ == "__main__":
    main()
