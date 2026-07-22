#!/usr/bin/env python3

import sys

import rospy
import moveit_commander

from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)


ARM_GROUP = "arm"
ENDPOSE_SERVICE = "/joint_moveit_ctrl_endpose"


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("arm_relative_pose_node")

    delta_x = float(rospy.get_param("~delta_x", 0.0))
    delta_y = float(rospy.get_param("~delta_y", 0.0))
    delta_z = float(rospy.get_param("~delta_z", 0.01))

    try:
        arm_group = moveit_commander.MoveGroupCommander(
            ARM_GROUP
        )

        rospy.sleep(1.0)

        planning_frame = arm_group.get_planning_frame()
        end_effector_link = arm_group.get_end_effector_link()

        current_pose = arm_group.get_current_pose(
            end_effector_link
        ).pose

        target_x = current_pose.position.x + delta_x
        target_y = current_pose.position.y + delta_y
        target_z = current_pose.position.z + delta_z

        rospy.loginfo(
            "Planning frame: %s",
            planning_frame,
        )
        rospy.loginfo(
            "End-effector link: %s",
            end_effector_link,
        )

        rospy.loginfo(
            "Current position: "
            "x=%.6f, y=%.6f, z=%.6f",
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        )

        rospy.loginfo(
            "Relative displacement: "
            "dx=%.6f, dy=%.6f, dz=%.6f",
            delta_x,
            delta_y,
            delta_z,
        )

        rospy.loginfo(
            "Target position: "
            "x=%.6f, y=%.6f, z=%.6f",
            target_x,
            target_y,
            target_z,
        )

        rospy.loginfo(
            "Waiting for service: %s",
            ENDPOSE_SERVICE,
        )

        rospy.wait_for_service(
            ENDPOSE_SERVICE,
            timeout=10.0,
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

        request.joint_endpose = [
            target_x,
            target_y,
            target_z,
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w,
        ]

        request.max_velocity = 0.05
        request.max_acceleration = 0.05

        endpose_service = rospy.ServiceProxy(
            ENDPOSE_SERVICE,
            JointMoveitCtrl,
        )

        rospy.loginfo(
            "Sending relative pose target..."
        )

        response = endpose_service(request)

        if not response.status:
            rospy.logerr(
                "Relative pose movement failed, "
                "error_code=%d",
                response.error_code,
            )
            return

        rospy.loginfo(
            "Relative pose movement completed successfully."
        )

    except rospy.ROSException as error:
        rospy.logerr(
            "ROS error: %s",
            str(error),
        )

    except rospy.ServiceException as error:
        rospy.logerr(
            "Service call failed: %s",
            str(error),
        )

    except Exception as error:
        rospy.logerr(
            "Unexpected error: %s",
            str(error),
        )

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
