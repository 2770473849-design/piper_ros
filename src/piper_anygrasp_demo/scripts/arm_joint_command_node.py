#!/usr/bin/env python3

import rospy

from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


ARM_SERVICE = "/joint_moveit_ctrl_arm"


def main():
    rospy.init_node("arm_joint_command_node")

    joints = rospy.get_param(
        "~joints",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    velocity = float(rospy.get_param("~velocity", 0.1))
    acceleration = float(rospy.get_param("~acceleration", 0.1))

    if not isinstance(joints, list) or len(joints) != 6:
        rospy.logerr(
            "Parameter ~joints must contain exactly 6 values: %s",
            joints,
        )
        return

    try:
        joints = [float(value) for value in joints]
    except (TypeError, ValueError):
        rospy.logerr(
            "All values in ~joints must be numbers: %s",
            joints,
        )
        return

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
    request.joint_states = joints
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
    request.max_velocity = velocity
    request.max_acceleration = acceleration

    try:
        arm_service = rospy.ServiceProxy(
            ARM_SERVICE,
            JointMoveitCtrl,
        )

        rospy.loginfo(
            "Sending joint target: %s",
            joints,
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
            "Arm joint command completed successfully."
        )
    else:
        rospy.logerr(
            "Arm command failed, error_code=%d",
            response.error_code,
        )


if __name__ == "__main__":
    main()
