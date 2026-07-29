#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy

from piper_anygrasp_demo.msg import (
    GraspCandidate,
    GraspCandidateArray,
)


class MockGraspCandidatesPublisher:
    """Publish deterministic mock grasp candidates for filter testing."""

    def __init__(self):
        self.output_topic = rospy.get_param(
            "~output_topic",
            "/grasp_candidates",
        )
        self.frame_id = rospy.get_param(
            "~frame_id",
            "base_link",
        )

        # 当前已经验证成功的抓取 TCP。
        self.tcp_x = float(
            rospy.get_param("~tcp_x", 0.2500)
        )
        self.tcp_y = float(
            rospy.get_param("~tcp_y", 0.0000)
        )
        self.tcp_z = float(
            rospy.get_param("~tcp_z", 0.0450)
        )

        self.publisher = rospy.Publisher(
            self.output_topic,
            GraspCandidateArray,
            queue_size=1,
            latch=True,
        )

    @staticmethod
    def make_candidate(
        candidate_id,
        position,
        orientation,
        score,
        width,
        height=0.020,
        depth=0.030,
    ):
        candidate = GraspCandidate()

        candidate.id = int(candidate_id)

        candidate.pose.position.x = float(position[0])
        candidate.pose.position.y = float(position[1])
        candidate.pose.position.z = float(position[2])

        candidate.pose.orientation.x = float(
            orientation[0]
        )
        candidate.pose.orientation.y = float(
            orientation[1]
        )
        candidate.pose.orientation.z = float(
            orientation[2]
        )
        candidate.pose.orientation.w = float(
            orientation[3]
        )

        candidate.score = float(score)

        # width 表示两指之间的总开口宽度，单位为米。
        candidate.width = float(width)
        candidate.height = float(height)
        candidate.depth = float(depth)

        return candidate

    def build_message(self):
        message = GraspCandidateArray()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.frame_id

        top_down_orientation = (
            0.0,
            1.0,
            0.0,
            0.0,
        )

        wrong_orientation = (
            0.0,
            0.0,
            0.0,
            1.0,
        )

        # Candidate 0:
        # 分数最高，但 X 超出当前 Piper 抓取工作区域。
        message.candidates.append(
            self.make_candidate(
                candidate_id=0,
                position=(
                    0.420,
                    self.tcp_y,
                    self.tcp_z,
                ),
                orientation=top_down_orientation,
                score=0.99,
                width=0.035,
            )
        )

        # Candidate 1:
        # 位置正确，但要求的夹爪总开口宽度过大。
        message.candidates.append(
            self.make_candidate(
                candidate_id=1,
                position=(
                    self.tcp_x,
                    self.tcp_y,
                    self.tcp_z,
                ),
                orientation=top_down_orientation,
                score=0.98,
                width=0.090,
            )
        )

        # Candidate 4:
        # 位置和宽度正常，但姿态不是当前稳定的俯视抓取。
        message.candidates.append(
            self.make_candidate(
                candidate_id=4,
                position=(
                    self.tcp_x,
                    self.tcp_y,
                    self.tcp_z,
                ),
                orientation=wrong_orientation,
                score=0.96,
                width=0.035,
            )
        )

        # Candidate 2:
        # 当前已经完整验证成功的正上方抓取。
        message.candidates.append(
            self.make_candidate(
                candidate_id=2,
                position=(
                    self.tcp_x,
                    self.tcp_y,
                    self.tcp_z,
                ),
                orientation=top_down_orientation,
                score=0.92,
                width=0.035,
            )
        )

        # Candidate 3:
        # 略微偏移的备用合法候选。
        message.candidates.append(
            self.make_candidate(
                candidate_id=3,
                position=(
                    self.tcp_x,
                    self.tcp_y,
                    self.tcp_z,
                ),
                orientation=top_down_orientation,
                score=0.88,
                width=0.036,
            )
        )

        return message

    @staticmethod
    def log_candidate(candidate):
        rospy.loginfo(
            "Candidate %d: "
            "score=%.3f, width=%.3f m, "
            "position=[%.4f, %.4f, %.4f], "
            "quaternion=[%.3f, %.3f, %.3f, %.3f]",
            candidate.id,
            candidate.score,
            candidate.width,
            candidate.pose.position.x,
            candidate.pose.position.y,
            candidate.pose.position.z,
            candidate.pose.orientation.x,
            candidate.pose.orientation.y,
            candidate.pose.orientation.z,
            candidate.pose.orientation.w,
        )

    def run(self):
        # 等待订阅连接建立，不是必需，但方便查看日志。
        rospy.sleep(1.0)

        message = self.build_message()

        rospy.loginfo(
            "Publishing %d mock grasp candidates "
            "in frame '%s' on %s.",
            len(message.candidates),
            message.header.frame_id,
            self.output_topic,
        )

        for candidate in message.candidates:
            self.log_candidate(candidate)

        self.publisher.publish(message)

        rospy.loginfo(
            "Mock grasp candidates published as "
            "a latched message."
        )
        rospy.loginfo(
            "Node will remain alive for future subscribers."
        )

        rospy.spin()


def main():
    rospy.init_node(
        "mock_grasp_candidates_publisher"
    )

    node = MockGraspCandidatesPublisher()
    node.run()


if __name__ == "__main__":
    main()