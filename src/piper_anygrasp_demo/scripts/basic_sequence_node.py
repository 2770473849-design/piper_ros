#!/usr/bin/env python3

import rospy

from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


ARM_SERVICE = "/joint_moveit_ctrl_arm"
GRIPPER_SERVICE = "/joint_moveit_ctrl_gripper"

ZERO_JOINTS = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

SAFE_JOINTS = [
    0.15,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


def make_request():
    request = JointMoveitCtrlRequest()

    request.joint_states = list(ZERO_JOINTS)
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

    return request


def call_arm(arm_service, joints, stage_name):
    request = make_request()
    request.joint_states = list(joints)

    rospy.loginfo(
        "[%s] Sending arm target: %s",
        stage_name,
        joints,
    )

    try:
        response = arm_service(request)
    except rospy.ServiceException as error:
        rospy.logerr(
            "[%s] Arm service call failed: %s",
            stage_name,
            str(error),
        )
        return False

    if not response.status:
        rospy.logerr(
            "[%s] Arm command failed, error_code=%d",
            stage_name,
            response.error_code,
        )
        return False

    rospy.loginfo(
        "[%s] Arm command completed.",
        stage_name,
    )
    return True


def call_gripper(gripper_service, opening, stage_name):
    request = make_request()
    request.gripper = opening

    rospy.loginfo(
        "[%s] Sending gripper opening: %.4f m",
        stage_name,
        opening,
    )

    try:
        response = gripper_service(request)
    except rospy.ServiceException as error:
        rospy.logerr(
            "[%s] Gripper service call failed: %s",
            stage_name,
            str(error),
        )
        return False

    if not response.status:
        rospy.logerr(
            "[%s] Gripper command failed, error_code=%d",
            stage_name,
            response.error_code,
        )
        return False

    rospy.loginfo(
        "[%s] Gripper command completed.",
        stage_name,
    )
    return True


def main():
    rospy.init_node("basic_sequence_node")

    rospy.loginfo("Waiting for Piper MoveIt services...")

    try:
        rospy.wait_for_service(
            ARM_SERVICE,
            timeout=10.0,
        )
        rospy.wait_for_service(
            GRIPPER_SERVICE,
            timeout=10.0,
        )
    except rospy.ROSException as error:
        rospy.logerr(
            "Required service is unavailable: %s",
            str(error),
        )
        return

    arm_service = rospy.ServiceProxy(
        ARM_SERVICE,
        JointMoveitCtrl,
    )
    gripper_service = rospy.ServiceProxy(
        GRIPPER_SERVICE,
        JointMoveitCtrl,
    )

    if not call_arm(
        arm_service,
        ZERO_JOINTS,
        "1/5 ARM ZERO",
    ):
        return

    rospy.sleep(0.5)

    if not call_gripper(
        gripper_service,
        0.02,
        "2/5 GRIPPER OPEN",
    ):
        return

    rospy.sleep(0.5)

    if not call_arm(
        arm_service,
        SAFE_JOINTS,
        "3/5 ARM SAFE TARGET",
    ):
        return

    rospy.sleep(0.5)

    if not call_gripper(
        gripper_service,
        0.0,
        "4/5 GRIPPER CLOSE",
    ):
        return

    rospy.sleep(0.5)

    if not call_arm(
        arm_service,
        ZERO_JOINTS,
        "5/5 ARM ZERO",
    ):
        return

    rospy.loginfo(
        "Basic Piper sequence completed successfully."
    )


if __name__ == "__main__":
    main()
