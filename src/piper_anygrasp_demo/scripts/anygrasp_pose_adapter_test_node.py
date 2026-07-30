#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import rospy

from tf.transformations import quaternion_from_matrix

from piper_anygrasp_demo.msg import (
    GraspCandidate,
    GraspCandidateArray,
)


class AnyGraspPoseAdapterTest:
    """
    使用一个合成的AnyGrasp抓取姿态，验证：

    AnyGrasp +X（接近方向） -> Piper +Z
    AnyGrasp +Y（开合方向） -> Piper +Y
    AnyGrasp +Z             -> Piper -X
    """

    def __init__(self):
        self.output_topic = rospy.get_param(
            "~output_topic",
            "/anygrasp_adapter_test_candidates",
        )

        self.frame_id = rospy.get_param(
            "~frame_id",
            "base_link",
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            GraspCandidateArray,
            queue_size=1,
            latch=True,
        )

    @staticmethod
    def adapt_rotation_matrix(anygrasp_rotation):
        """
        AnyGrasp列向量：
          column 0 = 局部 +X，接近方向
          column 1 = 局部 +Y，开合方向
          column 2 = 局部 +Z

        Piper列向量：
          column 0 = 局部 +X = -AnyGrasp +Z
          column 1 = 局部 +Y =  AnyGrasp +Y
          column 2 = 局部 +Z =  AnyGrasp +X
        """
        piper_rotation = np.column_stack(
            (
                -anygrasp_rotation[:, 2],
                anygrasp_rotation[:, 1],
                anygrasp_rotation[:, 0],
            )
        )

        return piper_rotation

    @staticmethod
    def rotation_matrix_to_quaternion(rotation):
        transform = np.eye(
            4,
            dtype=np.float64,
        )

        transform[:3, :3] = rotation

        quaternion = quaternion_from_matrix(
            transform
        )

        quaternion = np.asarray(
            quaternion,
            dtype=np.float64,
        )

        quaternion /= np.linalg.norm(
            quaternion
        )

        return quaternion

    @staticmethod
    def validate_rotation(rotation, name):
        orthogonality_error = np.linalg.norm(
            rotation.T @ rotation
            - np.eye(3)
        )

        determinant = np.linalg.det(
            rotation
        )

        rospy.loginfo(
            "%s determinant = %.6f",
            name,
            determinant,
        )

        rospy.loginfo(
            "%s orthogonality error = %.9f",
            name,
            orthogonality_error,
        )

        if abs(determinant - 1.0) > 1.0e-6:
            raise RuntimeError(
                "{} is not a proper rotation matrix.".format(
                    name
                )
            )

        if orthogonality_error > 1.0e-6:
            raise RuntimeError(
                "{} is not orthonormal.".format(
                    name
                )
            )

    def run(self):
        rospy.loginfo(
            "========== ANYGRASP FRAME ADAPTER TEST =========="
        )

        # 构造一个竖直向下的AnyGrasp姿态：
        #
        # AnyGrasp +X（接近） = base_link -Z
        # AnyGrasp +Y（开合） = base_link +Y
        # AnyGrasp +Z         = base_link +X
        #
        # 三列分别代表局部 +X、+Y、+Z。
        anygrasp_rotation = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.validate_rotation(
            anygrasp_rotation,
            "AnyGrasp rotation",
        )

        piper_rotation = self.adapt_rotation_matrix(
            anygrasp_rotation
        )

        self.validate_rotation(
            piper_rotation,
            "Piper rotation",
        )

        anygrasp_approach = anygrasp_rotation[:, 0]
        anygrasp_open_close = anygrasp_rotation[:, 1]

        piper_local_x = piper_rotation[:, 0]
        piper_local_y = piper_rotation[:, 1]
        piper_local_z = piper_rotation[:, 2]

        rospy.loginfo(
            "AnyGrasp +X approach: [%.3f, %.3f, %.3f]",
            anygrasp_approach[0],
            anygrasp_approach[1],
            anygrasp_approach[2],
        )

        rospy.loginfo(
            "AnyGrasp +Y open/close: [%.3f, %.3f, %.3f]",
            anygrasp_open_close[0],
            anygrasp_open_close[1],
            anygrasp_open_close[2],
        )

        rospy.loginfo(
            "Piper local +X: [%.3f, %.3f, %.3f]",
            piper_local_x[0],
            piper_local_x[1],
            piper_local_x[2],
        )

        rospy.loginfo(
            "Piper local +Y: [%.3f, %.3f, %.3f]",
            piper_local_y[0],
            piper_local_y[1],
            piper_local_y[2],
        )

        rospy.loginfo(
            "Piper local +Z approach: [%.3f, %.3f, %.3f]",
            piper_local_z[0],
            piper_local_z[1],
            piper_local_z[2],
        )

        approach_error = np.linalg.norm(
            piper_local_z
            - anygrasp_approach
        )

        opening_axis_error = np.linalg.norm(
            piper_local_y
            - anygrasp_open_close
        )

        rospy.loginfo(
            "Approach-axis mapping error = %.9f",
            approach_error,
        )

        rospy.loginfo(
            "Opening-axis mapping error = %.9f",
            opening_axis_error,
        )

        if approach_error > 1.0e-6:
            raise RuntimeError(
                "AnyGrasp +X was not mapped to Piper +Z."
            )

        if opening_axis_error > 1.0e-6:
            raise RuntimeError(
                "AnyGrasp +Y was not mapped to Piper +Y."
            )

        quaternion = self.rotation_matrix_to_quaternion(
            piper_rotation
        )

        rospy.loginfo(
            "Adapted Piper quaternion: "
            "[%.6f, %.6f, %.6f, %.6f]",
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
        )

        # 用当前已经验证过的稳定抓取TCP位置做显示。
        translation = np.array(
            [0.25, 0.0, 0.045],
            dtype=np.float64,
        )

        depth = 0.03

        # 官方定义中的夹爪尖端位置。
        gripper_tip = (
            translation
            + depth * anygrasp_approach
        )

        rospy.loginfo(
            "Synthetic AnyGrasp translation: "
            "[%.4f, %.4f, %.4f]",
            translation[0],
            translation[1],
            translation[2],
        )

        rospy.loginfo(
            "Synthetic gripper tip: "
            "[%.4f, %.4f, %.4f]",
            gripper_tip[0],
            gripper_tip[1],
            gripper_tip[2],
        )

        candidate = GraspCandidate()
        candidate.id = 100

        candidate.pose.position.x = translation[0]
        candidate.pose.position.y = translation[1]
        candidate.pose.position.z = translation[2]

        candidate.pose.orientation.x = quaternion[0]
        candidate.pose.orientation.y = quaternion[1]
        candidate.pose.orientation.z = quaternion[2]
        candidate.pose.orientation.w = quaternion[3]

        candidate.score = 0.90
        candidate.width = 0.035
        candidate.height = 0.030
        candidate.depth = depth

        message = GraspCandidateArray()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.frame_id
        message.candidates = [candidate]

        self.publisher.publish(message)

        rospy.loginfo(
            "Published adapted synthetic candidate on %s",
            self.output_topic,
        )

        rospy.loginfo(
            "Node remains alive for latched-message subscribers."
        )

        rospy.loginfo(
            "========== FRAME ADAPTER TEST PASSED =========="
        )

        rospy.spin()


def main():
    rospy.init_node(
        "anygrasp_pose_adapter_test_node"
    )

    try:
        AnyGraspPoseAdapterTest().run()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "AnyGrasp pose adapter test failed: %s",
            error,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()