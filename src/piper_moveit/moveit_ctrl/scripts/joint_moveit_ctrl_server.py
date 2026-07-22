#!/usr/bin/env python

import rospy
from moveit_commander import *
from moveit_ctrl.srv import JointMoveitCtrl, JointMoveitCtrlResponse
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

class JointMoveitCtrlServer:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('joint_moveit_ctrl_server')

        # 初始化 MoveIt
        roscpp_initialize([])
        self.robot = RobotCommander()

        # 获取 MoveIt 规划组列表
        available_groups = self.robot.get_group_names()
        rospy.loginfo(f"Available MoveIt groups: {available_groups}")

        # 仅实例化存在的规划组
        self.arm_move_group = None
        self.gripper_move_group = None
        self.piper_move_group = None

        if "arm" in available_groups:
            self.arm_move_group = MoveGroupCommander("arm")
            rospy.loginfo("Initialized arm move group.")
        
        if "gripper" in available_groups:
            self.gripper_move_group = MoveGroupCommander("gripper")
            rospy.loginfo("Initialized gripper move group.")
        
        if "piper" in available_groups:
            self.piper_move_group = MoveGroupCommander("piper")
            rospy.loginfo("Initialized piper move group.")

        # 创建关节运动控制服务
        self.arm_srv = rospy.Service('joint_moveit_ctrl_arm', JointMoveitCtrl, self.handle_joint_moveit_ctrl_arm)
        self.gripper_srv = rospy.Service('joint_moveit_ctrl_gripper', JointMoveitCtrl, self.handle_joint_moveit_ctrl_gripper)
        self.piper_srv = rospy.Service('joint_moveit_ctrl_piper', JointMoveitCtrl, self.handle_joint_moveit_ctrl_piper)
        self.endpose_srv = rospy.Service('joint_moveit_ctrl_endpose', JointMoveitCtrl, self.handle_joint_moveit_ctrl_endpose)

        rospy.loginfo("Joint MoveIt Control Services Ready.")

    def handle_joint_moveit_ctrl_arm(self, request):
        rospy.loginfo("Received arm joint movement request.")

        if not self.arm_move_group:
            rospy.logerr("Arm move group is not initialized.")
            return JointMoveitCtrlResponse(status=False, error_code=1)

        try:
            arm_joint_goal = list(request.joint_states[:6])

            max_velocity = max(
                1e-6,
                min(1.0 - 1e-6, request.max_velocity),
            )
            max_acceleration = max(
                1e-6,
                min(1.0 - 1e-6, request.max_acceleration),
            )

            self.arm_move_group.set_max_velocity_scaling_factor(
                max_velocity
            )
            self.arm_move_group.set_max_acceleration_scaling_factor(
                max_acceleration
            )
            self.arm_move_group.set_joint_value_target(
                arm_joint_goal
            )

            rospy.loginfo(
                "Arm target: %s, velocity=%.3f, acceleration=%.3f",
                arm_joint_goal,
                max_velocity,
                max_acceleration,
            )

            success = self.arm_move_group.go(wait=True)
            self.arm_move_group.stop()

            if not success:
                rospy.logerr(
                    "Arm planning or execution failed."
                )
                return JointMoveitCtrlResponse(
                    status=False,
                    error_code=2,
                )

            rospy.loginfo(
                "Arm movement executed successfully."
            )
            return JointMoveitCtrlResponse(
                status=True,
                error_code=0,
            )

        except Exception as e:
            rospy.logerr(
                "Exception during arm movement: %s",
                str(e),
            )
            return JointMoveitCtrlResponse(
                status=False,
                error_code=3,
            )

    def handle_joint_moveit_ctrl_gripper(self, request):
        rospy.loginfo("Received gripper joint movement request.")

        if not self.gripper_move_group:
            rospy.logerr("Gripper move group is not initialized.")
            return JointMoveitCtrlResponse(status=False, error_code=1)

        try:
            # Piper 单侧夹爪行程：0～0.035 m
            opening = max(0.0, min(0.035, request.gripper))

            # 两个夹爪关节方向相反，使用关节名避免顺序歧义
            gripper_goal = {
                "joint7": opening,
                "joint8": -opening,
            }

            self.gripper_move_group.set_joint_value_target(gripper_goal)
            success = self.gripper_move_group.go(wait=True)
            self.gripper_move_group.stop()

            if not success:
                rospy.logerr("Gripper planning or execution failed.")
                return JointMoveitCtrlResponse(status=False, error_code=2)

            rospy.loginfo(
                "Gripper executed successfully: joint7=%.4f, joint8=%.4f",
                opening,
                -opening,
            )
            return JointMoveitCtrlResponse(status=True, error_code=0)

        except Exception as e:
            rospy.logerr(f"Exception during gripper movement: {str(e)}")
            return JointMoveitCtrlResponse(status=False, error_code=3)

    def handle_joint_moveit_ctrl_piper(self, request):
        rospy.loginfo("Received piper joint movement request.")

        try:
            if self.piper_move_group:
                piper_joint_goal = list(request.joint_states[:6]) + [request.gripper]
                self.piper_move_group.set_joint_value_target(piper_joint_goal)
                max_velocity = max(1e-6, min(1-1e-6, request.max_velocity))
                max_acceleration = max(1e-6, min(1-1e-6, request.max_acceleration))
                self.piper_move_group.set_max_velocity_scaling_factor(max_velocity)
                self.piper_move_group.set_max_acceleration_scaling_factor(max_acceleration)
                rospy.loginfo(f"max_velocity: {max_velocity} max_acceleration: {max_acceleration}")
                self.piper_move_group.go(wait=True)
                rospy.loginfo("Piper movement executed successfully.")
            else:
                rospy.logerr("Piper move group is not initialized.")
        except Exception as e:
            rospy.logerr(f"Exception during piper movement: {str(e)}")
        
        return JointMoveitCtrlResponse(status=True, error_code=0)

    def handle_joint_moveit_ctrl_endpose(self, request):
        rospy.loginfo("Received endpose movement request.")

        if not self.arm_move_group:
            rospy.logerr("Arm move group is not initialized.")
            return JointMoveitCtrlResponse(
                status=False,
                error_code=1,
            )

        try:
            if len(request.joint_endpose) != 7:
                rospy.logerr(
                    "joint_endpose must contain exactly 7 values: "
                    "[x, y, z, qx, qy, qz, qw]"
                )
                return JointMoveitCtrlResponse(
                    status=False,
                    error_code=2,
                )

            position = list(request.joint_endpose[:3])
            quaternion = list(request.joint_endpose[3:7])

            quaternion_norm = sum(
                value * value
                for value in quaternion
            ) ** 0.5

            if quaternion_norm < 1e-8:
                rospy.logerr(
                    "Invalid quaternion: norm is zero."
                )
                return JointMoveitCtrlResponse(
                    status=False,
                    error_code=3,
                )

            quaternion = [
                value / quaternion_norm
                for value in quaternion
            ]

            target_pose = Pose()

            target_pose.position.x = position[0]
            target_pose.position.y = position[1]
            target_pose.position.z = position[2]

            target_pose.orientation.x = quaternion[0]
            target_pose.orientation.y = quaternion[1]
            target_pose.orientation.z = quaternion[2]
            target_pose.orientation.w = quaternion[3]

            max_velocity = max(
                1e-6,
                min(1.0 - 1e-6, request.max_velocity),
            )
            max_acceleration = max(
                1e-6,
                min(1.0 - 1e-6, request.max_acceleration),
            )

            self.arm_move_group.set_max_velocity_scaling_factor(
                max_velocity
            )
            self.arm_move_group.set_max_acceleration_scaling_factor(
                max_acceleration
            )

            self.arm_move_group.set_start_state_to_current_state()

            end_effector_link = (
                self.arm_move_group.get_end_effector_link()
            )

            self.arm_move_group.set_pose_target(
                target_pose,
                end_effector_link,
            )

            rospy.loginfo(
                "Endpose target in frame '%s': "
                "position=[%.6f, %.6f, %.6f], "
                "quaternion=[%.6f, %.6f, %.6f, %.6f]",
                self.arm_move_group.get_planning_frame(),
                position[0],
                position[1],
                position[2],
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],
            )

            rospy.loginfo(
                "Velocity=%.3f, acceleration=%.3f",
                max_velocity,
                max_acceleration,
            )

            success = self.arm_move_group.go(wait=True)

            self.arm_move_group.stop()
            self.arm_move_group.clear_pose_targets()

            if not success:
                rospy.logerr(
                    "Endpose planning or execution failed."
                )
                return JointMoveitCtrlResponse(
                    status=False,
                    error_code=4,
                )

            rospy.loginfo(
                "Endpose movement executed successfully."
            )

            return JointMoveitCtrlResponse(
                status=True,
                error_code=0,
            )

        except Exception as error:
            self.arm_move_group.stop()
            self.arm_move_group.clear_pose_targets()

            rospy.logerr(
                "Exception during endpose movement: %s",
                str(error),
            )

            return JointMoveitCtrlResponse(
                status=False,
                error_code=5,
            )


if __name__ == '__main__':
    JointMoveitCtrlServer()
    rospy.spin()
