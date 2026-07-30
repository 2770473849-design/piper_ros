#!/usr/bin/env python3

import copy
import math
import sys

import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import Pose, PoseStamped

from piper_anygrasp_demo.msg import GraspCandidateArray

from moveit_ctrl.srv import (
    JointMoveitCtrl,
    JointMoveitCtrlRequest,
)

from gazebo_ros_link_attacher.srv import (
    Attach,
    AttachRequest,
)

from moveit_msgs.msg import (
    AllowedCollisionEntry,
    PlanningScene,
    PlanningSceneComponents,
)

from moveit_msgs.srv import (
    ApplyPlanningScene,
    ApplyPlanningSceneRequest,
    GetPlanningScene,
    GetPlanningSceneRequest,
)


ARM_GROUP = "arm"

GRIPPER_SERVICE = "/joint_moveit_ctrl_gripper"
GAZEBO_ATTACH_SERVICE = "/link_attacher_node/attach"
GAZEBO_DETACH_SERVICE = "/link_attacher_node/detach"


class GazeboPickPlace:
    def __init__(self):
        self.object_name = rospy.get_param(
            "~object_name",
            "anygrasp_test_cube",
        )

        self.gazebo_robot_model = rospy.get_param(
            "~gazebo_robot_model",
            "arm",
        )

        # Gazebo 会合并 link6 与 gripper_base 的固定连接，
        # 所以 Gazebo 中使用实际存在的 link6。
        self.gazebo_robot_link = rospy.get_param(
            "~gazebo_robot_link",
            "link6",
        )

        self.gazebo_object_link = rospy.get_param(
            "~gazebo_object_link",
            "cube_link",
        )

        # MoveIt 中仍然存在 gripper_base。
        self.moveit_attach_link = rospy.get_param(
            "~moveit_attach_link",
            "gripper_base",
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

        # 方块中心上方的最终夹持中心 TCP 位姿。
        self.grasp_tcp_position = self.read_list_parameter(
            "~grasp_tcp_position",
            [0.25, 0.0, 0.045],
            3,
        )

        # 夹爪朝下。
        self.grasp_orientation = self.normalize_quaternion(
            self.read_list_parameter(
                "~grasp_orientation",
                [0.0, 1.0, 0.0, 0.0],
                4,
            )
        )

        # 从 ROS 话题接收抓取中心 TCP 位姿。
        self.grasp_pose_topic = rospy.get_param(
            "~grasp_pose_topic",
            "/grasp_tcp_pose",
        )

        # 非空时优先从完整候选消息读取：
        # pose + id + score + width + height + depth。
        self.grasp_candidate_topic = rospy.get_param(
            "~grasp_candidate_topic",
            "",
        ).strip()

        self.grasp_pose_timeout = float(
            rospy.get_param(
                "~grasp_pose_timeout",
                30.0,
            )
        )

        # 从 TCP z=0.090 下降到 TCP z=0.045。
        self.approach_distance = float(
            rospy.get_param(
                "~approach_distance",
                0.045,
            )
        )

        self.lift_distance = float(
            rospy.get_param(
                "~lift_distance",
                0.02,
            )
        )

        self.transport_y = float(
            rospy.get_param(
                "~transport_y",
                0.08,
            )
        )

        self.place_descent = float(
            rospy.get_param(
                "~place_descent",
                0.02,
            )
        )

        self.final_retreat = float(
            rospy.get_param(
                "~final_retreat",
                0.04,
            )
        )

        self.gripper_open = float(
            rospy.get_param(
                "~gripper_open",
                0.02,
            )
        )

        self.gripper_close = float(
            rospy.get_param(
                "~gripper_close",
                0.0170,
            )
        )

        # 候选 width 是两根手指之间的总开口宽度。
        # 服务中的 opening 是单侧关节目标。
        self.gripper_close_margin = float(
            rospy.get_param(
                "~gripper_close_margin",
                0.0005,
            )
        )

        self.gripper_open_margin = float(
            rospy.get_param(
                "~gripper_open_margin",
                0.0025,
            )
        )

        # Piper 仿真夹爪单侧关节范围约为 0～0.035 m，
        # 稍微避开极限位置。
        self.gripper_target_min = float(
            rospy.get_param(
                "~gripper_target_min",
                0.0010,
            )
        )

        self.gripper_target_max = float(
            rospy.get_param(
                "~gripper_target_max",
                0.0340,
            )
        )

        self.gripper_min_open_close_gap = float(
            rospy.get_param(
                "~gripper_min_open_close_gap",
                0.0010,
            )
        )

        self.velocity = float(
            rospy.get_param(
                "~velocity",
                0.03,
            )
        )

        self.acceleration = float(
            rospy.get_param(
                "~acceleration",
                0.03,
            )
        )

        self.eef_step = float(
            rospy.get_param(
                "~eef_step",
                0.001,
            )
        )

        self.ready_joints = self.read_list_parameter(
            "~ready_joints",
            [0.0, 0.4, -0.8, 0.0, 0.4, 0.0],
            6,
        )

        self.robot = moveit_commander.RobotCommander()

        self.scene = moveit_commander.PlanningSceneInterface(
            synchronous=True
        )

        self.arm = moveit_commander.MoveGroupCommander(
            ARM_GROUP
        )

        self.end_effector_link = (
            self.arm.get_end_effector_link()
        )

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(10.0)
        )

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        rospy.loginfo(
            "Planning frame: %s",
            self.arm.get_planning_frame(),
        )

        rospy.loginfo(
            "MoveIt end-effector link: %s",
            self.end_effector_link,
        )

        self.wait_for_services()

        self.gripper_service = rospy.ServiceProxy(
            GRIPPER_SERVICE,
            JointMoveitCtrl,
        )

        self.gazebo_attach_service = rospy.ServiceProxy(
            GAZEBO_ATTACH_SERVICE,
            Attach,
        )

        self.gazebo_detach_service = rospy.ServiceProxy(
            GAZEBO_DETACH_SERVICE,
            Attach,
        )

        # 仅在夹爪闭合阶段，允许两根手指接触目标物体。
        self.target_touch_links = rospy.get_param(
            "~target_touch_links",
            [
                "link7",
                "link8",
            ],
        )

        self.get_planning_scene_service_name = rospy.get_param(
            "~get_planning_scene_service",
            "/get_planning_scene",
        )

        self.apply_planning_scene_service_name = rospy.get_param(
            "~apply_planning_scene_service",
            "/apply_planning_scene",
        )

        self.get_planning_scene_client = rospy.ServiceProxy(
            self.get_planning_scene_service_name,
            GetPlanningScene,
        )

        self.apply_planning_scene_client = rospy.ServiceProxy(
            self.apply_planning_scene_service_name,
            ApplyPlanningScene,
        )

        self.saved_touch_permissions = {}

    @staticmethod
    def read_list_parameter(name, default, expected_length):
        value = rospy.get_param(name, default)

        if not isinstance(value, list):
            raise ValueError(
                "{} must be a list.".format(name)
            )

        if len(value) != expected_length:
            raise ValueError(
                "{} must contain {} values.".format(
                    name,
                    expected_length,
                )
            )

        return [float(item) for item in value]

    @staticmethod
    def normalize_quaternion(quaternion):
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

        return [
            value / norm
            for value in axis
        ]

    def wait_for_services(self):
        for service_name in [
            GRIPPER_SERVICE,
            GAZEBO_ATTACH_SERVICE,
            GAZEBO_DETACH_SERVICE,
        ]:
            rospy.loginfo(
                "Waiting for service: %s",
                service_name,
            )

            rospy.wait_for_service(
                service_name,
                timeout=15.0,
            )

    def command_gripper(self, opening, stage_name):
        request = JointMoveitCtrlRequest()

        request.joint_states = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        request.gripper = opening

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

        rospy.loginfo(
            "%s: target opening %.4f m",
            stage_name,
            opening,
        )

        response = self.gripper_service(request)

        if not response.status:
            raise RuntimeError(
                "{} failed, error_code={}".format(
                    stage_name,
                    response.error_code,
                )
            )

        rospy.loginfo(
            "%s completed.",
            stage_name,
        )

    def move_ready(self):
        rospy.loginfo(
            "MOVE TO READY_A: %s",
            self.ready_joints,
        )

        self.arm.set_start_state_to_current_state()

        self.arm.set_max_velocity_scaling_factor(
            self.velocity
        )

        self.arm.set_max_acceleration_scaling_factor(
            self.acceleration
        )

        self.arm.set_joint_value_target(
            self.ready_joints
        )

        success = self.arm.go(wait=True)

        self.arm.stop()

        if not success:
            raise RuntimeError(
                "MOVE TO READY_A failed."
            )

        rospy.loginfo(
            "MOVE TO READY_A completed."
        )

    def move_to_pose(self, target_pose, stage_name):
        self.arm.set_start_state_to_current_state()

        self.arm.set_max_velocity_scaling_factor(
            self.velocity
        )

        self.arm.set_max_acceleration_scaling_factor(
            self.acceleration
        )

        self.arm.set_pose_target(
            target_pose,
            self.end_effector_link,
        )

        rospy.loginfo(
            "%s: planning and executing...",
            stage_name,
        )

        success = self.arm.go(wait=True)

        self.arm.stop()
        self.arm.clear_pose_targets()

        if not success:
            raise RuntimeError(
                "{} failed.".format(stage_name)
            )

        rospy.loginfo(
            "%s completed.",
            stage_name,
        )

    def execute_cartesian(
        self,
        target_pose,
        stage_name,
    ):
        self.arm.set_start_state_to_current_state()

        plan, fraction = self.arm.compute_cartesian_path(
            [target_pose],
            self.eef_step,
            True,
        )

        rospy.loginfo(
            "%s path fraction: %.3f",
            stage_name,
            fraction,
        )

        if fraction < 0.999:
            raise RuntimeError(
                "{} path incomplete: fraction={:.3f}".format(
                    stage_name,
                    fraction,
                )
            )

        if not plan.joint_trajectory.points:
            raise RuntimeError(
                "{} trajectory has no points.".format(
                    stage_name
                )
            )

        retimed_plan = self.arm.retime_trajectory(
            self.robot.get_current_state(),
            plan,
            self.velocity,
            self.acceleration,
        )

        rospy.loginfo(
            "%s: executing...",
            stage_name,
        )

        success = self.arm.execute(
            retimed_plan,
            wait=True,
        )

        self.arm.stop()

        if not success:
            raise RuntimeError(
                "{} execution failed.".format(
                    stage_name
                )
            )

        rospy.loginfo(
            "%s completed.",
            stage_name,
        )

    def tcp_pose_to_link6_pose(
        self,
        tcp_position,
        quaternion,
    ):
        local_z = self.local_positive_z_axis(
            quaternion
        )

        pose = Pose()

        pose.position.x = (
            tcp_position[0]
            - self.tcp_offset * local_z[0]
        )

        pose.position.y = (
            tcp_position[1]
            - self.tcp_offset * local_z[1]
        )

        pose.position.z = (
            tcp_position[2]
            - self.tcp_offset * local_z[2]
        )

        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose

    def wait_for_scene_state(
        self,
        attached=None,
        known=None,
        timeout=5.0,
    ):
        start = rospy.Time.now()

        while (
            rospy.Time.now() - start
            < rospy.Duration(timeout)
            and not rospy.is_shutdown()
        ):
            attached_objects = (
                self.scene.get_attached_objects(
                    [self.object_name]
                )
            )

            is_attached = (
                self.object_name
                in attached_objects
            )

            is_known = (
                self.object_name
                in self.scene.get_known_object_names()
            )

            attached_ok = (
                attached is None
                or is_attached == attached
            )

            known_ok = (
                known is None
                or is_known == known
            )

            if attached_ok and known_ok:
                return True

            rospy.sleep(0.1)

        return False

    def attach_moveit(self):
        known_objects = (
            self.scene.get_known_object_names()
        )

        if self.object_name not in known_objects:
            raise RuntimeError(
                "MoveIt world does not contain '{}'.".format(
                    self.object_name
                )
            )

        touch_links = self.robot.get_link_names(
            group="gripper"
        )

        for link_name in [
            "link6",
            "gripper_base",
            "link7",
            "link8",
        ]:
            if link_name not in touch_links:
                touch_links.append(link_name)

        rospy.loginfo(
            "MOVEIT ATTACH: %s -> %s",
            self.object_name,
            self.moveit_attach_link,
        )

        self.scene.attach_box(
            self.moveit_attach_link,
            self.object_name,
            touch_links=touch_links,
        )

        if not self.wait_for_scene_state(
            attached=True,
            known=False,
        ):
            raise RuntimeError(
                "MoveIt object attach failed."
            )

        rospy.loginfo(
            "MOVEIT ATTACH completed."
        )

    def detach_moveit(self):
        rospy.loginfo(
            "MOVEIT DETACH: %s",
            self.object_name,
        )

        self.scene.remove_attached_object(
            self.moveit_attach_link,
            name=self.object_name,
        )

        if not self.wait_for_scene_state(
            attached=False,
            known=True,
        ):
            raise RuntimeError(
                "MoveIt object detach failed."
            )

        rospy.loginfo(
            "MOVEIT DETACH completed."
        )

    def gazebo_attachment_request(self):
        request = AttachRequest()

        request.model_name_1 = (
            self.gazebo_robot_model
        )

        request.link_name_1 = (
            self.gazebo_robot_link
        )

        request.model_name_2 = (
            self.object_name
        )

        request.link_name_2 = (
            self.gazebo_object_link
        )

        return request

    def attach_gazebo(self):
        rospy.loginfo(
            "GAZEBO ATTACH: %s::%s <-> %s::%s",
            self.gazebo_robot_model,
            self.gazebo_robot_link,
            self.object_name,
            self.gazebo_object_link,
        )

        response = self.gazebo_attach_service(
            self.gazebo_attachment_request()
        )

        if not response.ok:
            raise RuntimeError(
                "Gazebo physical attach failed."
            )

        rospy.loginfo(
            "GAZEBO ATTACH completed."
        )

    def detach_gazebo(self):
        rospy.loginfo(
            "GAZEBO DETACH: %s",
            self.object_name,
        )

        response = self.gazebo_detach_service(
            self.gazebo_attachment_request()
        )

        if not response.ok:
            raise RuntimeError(
                "Gazebo physical detach failed."
            )

        rospy.loginfo(
            "GAZEBO DETACH completed."
        )

    def current_pose_offset(
        self,
        dx=0.0,
        dy=0.0,
        dz=0.0,
    ):
        pose = copy.deepcopy(
            self.arm.get_current_pose(
                self.end_effector_link
            ).pose
        )

        pose.position.x += dx
        pose.position.y += dy
        pose.position.z += dz

        return pose

    def configure_gripper_from_candidate_width(
        self,
        candidate_width,
    ):
        if (
            not math.isfinite(candidate_width)
            or candidate_width <= 0.0
        ):
            raise RuntimeError(
                "Candidate width must be a positive finite value."
            )

        half_width = 0.5 * candidate_width

        calculated_close = (
            half_width
            - self.gripper_close_margin
        )

        calculated_open = (
            half_width
            + self.gripper_open_margin
        )

        # 保留原 gripper_open 作为最低张开目标，
        # 防止候选较窄时夹爪张开不足。
        calculated_open = max(
            calculated_open,
            self.gripper_open,
        )

        calculated_close = min(
            max(
                calculated_close,
                self.gripper_target_min,
            ),
            self.gripper_target_max,
        )

        calculated_open = min(
            max(
                calculated_open,
                self.gripper_target_min,
            ),
            self.gripper_target_max,
        )

        if (
            calculated_open
            - calculated_close
            < self.gripper_min_open_close_gap
        ):
            raise RuntimeError(
                "Candidate width {:.4f} m produces invalid "
                "gripper targets: open={:.4f}, close={:.4f}.".format(
                    candidate_width,
                    calculated_open,
                    calculated_close,
                )
            )

        self.gripper_open = calculated_open
        self.gripper_close = calculated_close

        rospy.loginfo(
            "Candidate width %.4f m -> "
            "gripper open %.4f m, close %.4f m",
            candidate_width,
            self.gripper_open,
            self.gripper_close,
        )

    def wait_for_grasp_pose(self):
        if self.grasp_candidate_topic:
            rospy.loginfo(
                "Waiting for planned grasp candidate on topic: %s",
                self.grasp_candidate_topic,
            )

            try:
                candidate_message = rospy.wait_for_message(
                    self.grasp_candidate_topic,
                    GraspCandidateArray,
                    timeout=self.grasp_pose_timeout,
                )

            except rospy.ROSException:
                raise RuntimeError(
                    "Timed out waiting for grasp candidate "
                    "on '{}'.".format(
                        self.grasp_candidate_topic
                    )
                )

            if len(candidate_message.candidates) != 1:
                raise RuntimeError(
                    "Expected exactly one planned grasp candidate, "
                    "but received {}.".format(
                        len(candidate_message.candidates)
                    )
                )

            candidate = candidate_message.candidates[0]

            message = PoseStamped()
            message.header = copy.deepcopy(
                candidate_message.header
            )
            message.pose = copy.deepcopy(
                candidate.pose
            )

            rospy.loginfo(
                "Received planned Candidate %d: "
                "score=%.3f, width=%.4f m, "
                "height=%.4f m, depth=%.4f m",
                candidate.id,
                candidate.score,
                candidate.width,
                candidate.height,
                candidate.depth,
            )

            self.configure_gripper_from_candidate_width(
                candidate.width
            )

        else:
            rospy.loginfo(
                "Waiting for grasp TCP pose on topic: %s",
                self.grasp_pose_topic,
            )

            try:
                message = rospy.wait_for_message(
                    self.grasp_pose_topic,
                    PoseStamped,
                    timeout=self.grasp_pose_timeout,
                )

            except rospy.ROSException:
                raise RuntimeError(
                    "Timed out waiting for grasp pose "
                    "on '{}'.".format(
                        self.grasp_pose_topic
                    )
                )

            rospy.loginfo(
                "Using legacy PoseStamped input; "
                "gripper open=%.4f m, close=%.4f m.",
                self.gripper_open,
                self.gripper_close,
            )

        received_frame = (
            message.header.frame_id.lstrip("/")
        )
        expected_frame = (
            self.planning_frame.lstrip("/")
        )

        if not received_frame:
            raise RuntimeError(
                "Received grasp pose has an empty frame_id."
            )

        if received_frame != expected_frame:
            source_frame = message.header.frame_id

            rospy.loginfo(
                "Transforming grasp pose from %s to %s...",
                source_frame,
                self.planning_frame,
            )

            try:
                message = self.tf_buffer.transform(
                    message,
                    self.planning_frame,
                    rospy.Duration(2.0),
                )

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as error:
                raise RuntimeError(
                    "Failed to transform grasp pose "
                    "from '{}' to '{}': {}".format(
                        source_frame,
                        self.planning_frame,
                        str(error),
                    )
                )

            rospy.loginfo(
                "Grasp pose TF transform completed."
            )

        self.grasp_tcp_position = [
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ]

        self.grasp_orientation = (
            self.normalize_quaternion(
                [
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                    message.pose.orientation.w,
                ]
            )
        )

        rospy.loginfo(
            "Received grasp TCP position: "
            "[%.4f, %.4f, %.4f]",
            self.grasp_tcp_position[0],
            self.grasp_tcp_position[1],
            self.grasp_tcp_position[2],
        )

        rospy.loginfo(
            "Received grasp orientation: "
            "[%.4f, %.4f, %.4f, %.4f]",
            self.grasp_orientation[0],
            self.grasp_orientation[1],
            self.grasp_orientation[2],
            self.grasp_orientation[3],
        )

    @staticmethod
    def ensure_acm_name(
        allowed_collision_matrix,
        name,
    ):
        """Ensure a name exists as both a row and column in the ACM."""

        if name in allowed_collision_matrix.entry_names:
            return allowed_collision_matrix.entry_names.index(
                name
            )

        old_size = len(
            allowed_collision_matrix.entry_names
        )

        allowed_collision_matrix.entry_names.append(
            name
        )

        for entry in allowed_collision_matrix.entry_values:
            while len(entry.enabled) < old_size:
                entry.enabled.append(False)

            entry.enabled.append(False)

        new_entry = AllowedCollisionEntry()
        new_entry.enabled = [
            False
            for _ in range(old_size + 1)
        ]

        allowed_collision_matrix.entry_values.append(
            new_entry
        )

        return old_size

    @classmethod
    def get_acm_pair(
        cls,
        allowed_collision_matrix,
        first_name,
        second_name,
    ):
        first_index = cls.ensure_acm_name(
            allowed_collision_matrix,
            first_name,
        )

        second_index = cls.ensure_acm_name(
            allowed_collision_matrix,
            second_name,
        )

        return bool(
            allowed_collision_matrix
            .entry_values[first_index]
            .enabled[second_index]
        )

    @classmethod
    def set_acm_pair(
        cls,
        allowed_collision_matrix,
        first_name,
        second_name,
        enabled,
    ):
        first_index = cls.ensure_acm_name(
            allowed_collision_matrix,
            first_name,
        )

        second_index = cls.ensure_acm_name(
            allowed_collision_matrix,
            second_name,
        )

        allowed_collision_matrix.entry_values[
            first_index
        ].enabled[second_index] = bool(enabled)

        allowed_collision_matrix.entry_values[
            second_index
        ].enabled[first_index] = bool(enabled)

    def get_allowed_collision_matrix(self):
        request = GetPlanningSceneRequest()

        request.components.components = (
            PlanningSceneComponents
            .ALLOWED_COLLISION_MATRIX
        )

        response = self.get_planning_scene_client(
            request
        )

        return response.scene.allowed_collision_matrix

    def apply_allowed_collision_matrix(
        self,
        allowed_collision_matrix,
    ):
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.allowed_collision_matrix = (
            allowed_collision_matrix
        )

        request = ApplyPlanningSceneRequest()
        request.scene = planning_scene

        response = self.apply_planning_scene_client(
            request
        )

        if not response.success:
            raise RuntimeError(
                "MoveIt rejected the Allowed Collision Matrix update."
            )

    def allow_target_touch(self):
        """Allow finger contact only after Cartesian approach completes."""

        rospy.loginfo(
            "Waiting for MoveIt planning-scene services..."
        )

        rospy.wait_for_service(
            self.get_planning_scene_service_name,
            timeout=10.0,
        )

        rospy.wait_for_service(
            self.apply_planning_scene_service_name,
            timeout=10.0,
        )

        allowed_collision_matrix = (
            self.get_allowed_collision_matrix()
        )

        self.saved_touch_permissions = {}

        for link_name in self.target_touch_links:
            previous_value = self.get_acm_pair(
                allowed_collision_matrix,
                self.object_name,
                link_name,
            )

            self.saved_touch_permissions[
                link_name
            ] = previous_value

            self.set_acm_pair(
                allowed_collision_matrix,
                self.object_name,
                link_name,
                True,
            )

            rospy.loginfo(
                "Temporarily allowing expected contact: "
                "%s <-> %s",
                self.object_name,
                link_name,
            )

        self.apply_allowed_collision_matrix(
            allowed_collision_matrix
        )

        rospy.sleep(0.2)

        rospy.loginfo(
            "Temporary target-touch permissions applied."
        )

    def restore_target_touch(self):
        """Restore the ACM values that existed before grasp closing."""

        if not self.saved_touch_permissions:
            return

        allowed_collision_matrix = (
            self.get_allowed_collision_matrix()
        )

        for (
            link_name,
            previous_value,
        ) in self.saved_touch_permissions.items():

            self.set_acm_pair(
                allowed_collision_matrix,
                self.object_name,
                link_name,
                previous_value,
            )

            rospy.loginfo(
                "Restoring collision permission: "
                "%s <-> %s = %s",
                self.object_name,
                link_name,
                previous_value,
            )

        self.apply_allowed_collision_matrix(
            allowed_collision_matrix
        )

        self.saved_touch_permissions = {}

        rospy.loginfo(
            "Original target-touch permissions restored."
        )

    def run(self):
        rospy.loginfo(
            "========== TOPIC-DRIVEN PICK AND PLACE =========="
        )

        self.wait_for_grasp_pose()

        local_z = self.local_positive_z_axis(
            self.grasp_orientation
        )

        grasp_tcp = list(
            self.grasp_tcp_position
        )

        pregrasp_tcp = [
            grasp_tcp[index]
            - self.approach_distance
            * local_z[index]
            for index in range(3)
        ]

        pregrasp_link6 = (
            self.tcp_pose_to_link6_pose(
                pregrasp_tcp,
                self.grasp_orientation,
            )
        )

        grasp_link6 = (
            self.tcp_pose_to_link6_pose(
                grasp_tcp,
                self.grasp_orientation,
            )
        )

        rospy.loginfo(
            "Grasp TCP: [%.4f, %.4f, %.4f]",
            grasp_tcp[0],
            grasp_tcp[1],
            grasp_tcp[2],
        )

        rospy.loginfo(
            "Pregrasp TCP: [%.4f, %.4f, %.4f]",
            pregrasp_tcp[0],
            pregrasp_tcp[1],
            pregrasp_tcp[2],
        )

        self.move_ready()

        self.command_gripper(
            self.gripper_open,
            "GRIPPER OPEN",
        )

        self.move_to_pose(
            pregrasp_link6,
            "MOVE TO PREGRASP",
        )

        self.execute_cartesian(
            grasp_link6,
            "CARTESIAN APPROACH",
        )

        # 到达方块两侧后，才允许两根手指接触目标。
        self.allow_target_touch()

        try:
            self.command_gripper(
                self.gripper_close,
                "GRIPPER CLOSE",
            )

            self.attach_moveit()
            self.attach_gazebo()

        finally:
            # attach完成后立即恢复原碰撞规则。
            self.restore_target_touch()

        rospy.sleep(0.3)

        lift_target = self.current_pose_offset(
            dx=-self.lift_distance * local_z[0],
            dy=-self.lift_distance * local_z[1],
            dz=-self.lift_distance * local_z[2],
        )

        self.execute_cartesian(
            lift_target,
            "CARTESIAN LIFT",
        )

        transport_target = self.current_pose_offset(
            dy=self.transport_y,
        )

        self.move_to_pose(
            transport_target,
            "TRANSPORT",
        )

        descent_target = self.current_pose_offset(
            dx=self.place_descent * local_z[0],
            dy=self.place_descent * local_z[1],
            dz=self.place_descent * local_z[2],
        )

        self.execute_cartesian(
            descent_target,
            "PLACE DESCENT",
        )

        self.detach_gazebo()

        rospy.sleep(0.5)

        self.command_gripper(
            self.gripper_open,
            "GRIPPER OPEN FOR RELEASE",
        )

        self.detach_moveit()

        retreat_target = self.current_pose_offset(
            dx=-self.final_retreat * local_z[0],
            dy=-self.final_retreat * local_z[1],
            dz=-self.final_retreat * local_z[2],
        )

        self.execute_cartesian(
            retreat_target,
            "FINAL RETREAT",
        )

        rospy.loginfo(
            "========== TOPIC PICK AND PLACE COMPLETED =========="
        )


def main():
    moveit_commander.roscpp_initialize(
        sys.argv
    )

    rospy.init_node(
        "gazebo_pick_place_from_topic_node"
    )

    try:
        node = GazeboPickPlace()
        node.run()

    except Exception as error:
        rospy.logerr(
            "Pick and place failed: %s",
            str(error),
        )

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
