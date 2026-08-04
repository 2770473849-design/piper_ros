#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import threading

import rospy

from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState


class DisplayTrajectoryAdapter:
    """
    Fill DisplayTrajectory.trajectory_start with the latest complete
    joint state so non-arm joints, especially joint7 and joint8,
    remain correctly displayed during arm-only trajectory animation.

    This node publishes visualization messages only.
    It sends no robot, MoveIt execution, or gripper command.
    """

    def __init__(self):
        self.joint_state_topic = rospy.get_param(
            "~joint_state_topic",
            "/joint_states_calibration",
        )
        self.input_topic = rospy.get_param(
            "~input_topic",
            "/move_group/display_planned_path",
        )
        self.output_topic = rospy.get_param(
            "~output_topic",
            "/safe_real_display_planned_path",
        )

        self.lock = threading.Lock()
        self.latest_joint_state = None

        self.publisher = rospy.Publisher(
            self.output_topic,
            DisplayTrajectory,
            queue_size=1,
            latch=True,
        )

        self.joint_subscriber = rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self.joint_state_callback,
            queue_size=1,
        )

        self.display_subscriber = rospy.Subscriber(
            self.input_topic,
            DisplayTrajectory,
            self.display_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Safe display adapter ready: %s -> %s",
            self.input_topic,
            self.output_topic,
        )
        rospy.loginfo(
            "Complete display start state source: %s",
            self.joint_state_topic,
        )
        rospy.logwarn(
            "VISUALIZATION ONLY: no robot or gripper command "
            "will be sent."
        )

    def joint_state_callback(self, message):
        if len(message.name) != len(message.position):
            rospy.logwarn_throttle(
                2.0,
                "Rejected incomplete JointState: "
                "name=%d, position=%d",
                len(message.name),
                len(message.position),
            )
            return

        with self.lock:
            self.latest_joint_state = copy.deepcopy(message)

    def display_callback(self, message):
        with self.lock:
            joint_state = copy.deepcopy(
                self.latest_joint_state
            )

        if joint_state is None:
            rospy.logwarn(
                "Display trajectory received before joint state; "
                "not republishing misleading visualization."
            )
            return

        values = dict(
            zip(
                joint_state.name,
                joint_state.position,
            )
        )

        missing = [
            name
            for name in ("joint7", "joint8")
            if name not in values
        ]

        if missing:
            rospy.logerr(
                "Cannot correct trajectory display; missing: %s",
                missing,
            )
            return

        corrected = copy.deepcopy(message)

        # Replace the trajectory start JointState with the latest
        # complete j1-j8 state. The arm trajectory will animate j1-j6;
        # joint7/joint8 then remain fixed at their real open positions.
        corrected.trajectory_start.joint_state = joint_state
        corrected.trajectory_start.joint_state.velocity = []
        corrected.trajectory_start.joint_state.effort = []
        corrected.trajectory_start.is_diff = False

        self.publisher.publish(corrected)

        opening_mm = abs(
            values["joint7"] - values["joint8"]
        ) * 1000.0

        rospy.loginfo(
            "Corrected trajectory display published: "
            "joint7=%.5f, joint8=%.5f, opening=%.1f mm, "
            "segments=%d",
            values["joint7"],
            values["joint8"],
            opening_mm,
            len(corrected.trajectory),
        )


def main():
    rospy.init_node(
        "safe_real_display_trajectory_adapter"
    )

    DisplayTrajectoryAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
