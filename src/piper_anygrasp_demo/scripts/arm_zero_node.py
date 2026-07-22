#!/usr/bin/env python3

import rospy

from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


ARM_SERVICE = "/joint_moveit_ctrl_arm"


def main():
    rospy.init_node("arm_zero_node")

    rospy.loginfo(
        "Waiting for service: %s",
        ARM_SERVICE,
    )

    try:
        rospy.wait_for_service(
            ARM_SERVICE,
            timeout=10.0,
        )
    except rospy.ROSException:
        rospy.logerr(
            "Service is unavailable: %s",
            ARM_SERVICE,
        )
        return

    request = JointMoveitCtrlRequest()

    request.joint_states = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # 下面两个字段在 arm 服务中不会实际使用，
    # 但属于服务请求的固定字段，仍需填写。
    request.gripper = 0.0
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
        arm_service = rospy.ServiceProxy(
            ARM_SERVICE,
            JointMoveitCtrl,
        )

        rospy.loginfo(
            "Sending arm zero target..."
        )

        response = arm_service(request)

    except rospy.ServiceException as error:
        rospy.logerr(
            "Service call failed: %s",
            str(error),
        )
        return

    if response.status:
        rospy.loginfo(
            "Arm returned to zero successfully."
        )
    else:
        rospy.logerr(
            "Arm zero failed, error_code=%d",
            response.error_code,
        )


if __name__ == "__main__":
    main()
