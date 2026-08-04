#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time

import rosgraph
import rospy

from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


JOINTS = [
    "joint1", "joint2", "joint3",
    "joint4", "joint5", "joint6",
]

DISPLAY_TOPIC = "/safe_real_display_planned_path"
FEEDBACK_TOPIC = "/joint_states_calibration"
COMMAND_TOPIC = "/joint_states"

SPEED_PERCENT = 3.0
SELECT_STEP_DEG = 2.0
MAX_SEGMENT_DEG = 5.0
REACHED_DEG = 1.0
TIMEOUT_SEC = 20.0


def positions(message, names):
    table = dict(zip(message.name, message.position))
    missing = [name for name in names if name not in table]

    if missing:
        raise RuntimeError(
            "反馈缺少关节：{}".format(missing)
        )

    return [float(table[name]) for name in names]


def read_feedback():
    message = rospy.wait_for_message(
        FEEDBACK_TOPIC,
        JointState,
        timeout=3.0,
    )
    return message, positions(message, JOINTS)


def trajectory_target(joint_names, point):
    table = dict(zip(joint_names, point.positions))
    missing = [name for name in JOINTS if name not in table]

    if missing:
        raise RuntimeError(
            "规划轨迹缺少关节：{}".format(missing)
        )

    return [float(table[name]) for name in JOINTS]


def max_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def send_target(publisher, target):
    command = JointState()
    command.name = list(JOINTS)
    command.position = list(target)
    command.velocity = [0.0] * 6 + [SPEED_PERCENT]

    for _ in range(3):
        command.header.stamp = rospy.Time.now()
        publisher.publish(command)
        rospy.sleep(0.05)


def wait_reached(target):
    deadline = time.time() + TIMEOUT_SEC
    tolerance = math.radians(REACHED_DEG)

    while not rospy.is_shutdown():
        _, actual = read_feedback()
        error = max_error(actual, target)

        if error <= tolerance:
            return math.degrees(error)

        if time.time() >= deadline:
            raise RuntimeError(
                "目标未按时到位，最大误差 {:.2f}°".format(
                    math.degrees(error)
                )
            )

        rospy.sleep(0.1)


def main():
    rospy.init_node("run_planned_pregrasp_once")

    if not os.path.exists("/sys/class/net/can0"):
        raise RuntimeError("can0 不存在，禁止运动")

    enabled = rospy.wait_for_message(
        "/enable_flag",
        Bool,
        timeout=3.0,
    )

    if not enabled.data:
        raise RuntimeError("机械臂未使能，禁止运动")

    master = rosgraph.Master(rospy.get_name())
    publishers, subscribers, _ = master.getSystemState()

    existing_publishers = dict(publishers).get(
        COMMAND_TOPIC,
        [],
    )

    if existing_publishers:
        raise RuntimeError(
            "{} 已有发布者：{}".format(
                COMMAND_TOPIC,
                existing_publishers,
            )
        )

    command_subscribers = dict(subscribers).get(
        COMMAND_TOPIC,
        [],
    )

    if "/piper_ctrl_single_node" not in command_subscribers:
        raise RuntimeError(
            "Piper 驱动没有订阅 /joint_states"
        )

    display = rospy.wait_for_message(
        DISPLAY_TOPIC,
        DisplayTrajectory,
        timeout=5.0,
    )

    if len(display.trajectory) != 1:
        raise RuntimeError(
            "轨迹段数量不是1：{}".format(
                len(display.trajectory)
            )
        )

    trajectory = display.trajectory[0].joint_trajectory

    if not trajectory.points:
        raise RuntimeError("规划轨迹为空")

    feedback_message, current = read_feedback()
    feedback_table = dict(
        zip(
            feedback_message.name,
            feedback_message.position,
        )
    )

    if (
        "joint7" not in feedback_table
        or "joint8" not in feedback_table
    ):
        raise RuntimeError("缺少夹爪反馈")

    opening = abs(
        float(feedback_table["joint7"])
        - float(feedback_table["joint8"])
    )

    if opening < 0.060:
        raise RuntimeError(
            "夹爪开口仅 {:.1f} mm，小于60 mm".format(
                opening * 1000.0
            )
        )

    targets = [
        trajectory_target(
            trajectory.joint_names,
            point,
        )
        for point in trajectory.points
    ]

    if max_error(current, targets[0]) > math.radians(3.0):
        raise RuntimeError(
            "规划起点与实机当前位置相差超过3°，"
            "可能读取到了旧轨迹"
        )

    selected = []
    previous = current
    threshold = math.radians(SELECT_STEP_DEG)

    for target in targets:
        if max_error(previous, target) >= threshold:
            selected.append(target)
            previous = target

    if (
        not selected
        or max_error(selected[-1], targets[-1]) > 1.0e-6
    ):
        selected.append(targets[-1])

    previous = current

    for index, target in enumerate(selected, start=1):
        segment = math.degrees(
            max_error(previous, target)
        )

        if segment > MAX_SEGMENT_DEG:
            raise RuntimeError(
                "第{}段跨度过大：{:.2f}°".format(
                    index,
                    segment,
                )
            )

        previous = target

    print("\n========== 预抓取实机运动 ==========")
    print("原始轨迹点：", len(targets))
    print("发送小段数：", len(selected))
    print("夹爪开口：{:.1f} mm".format(opening * 1000.0))

    print("\n最终关节目标：")
    for name, start, target in zip(
        JOINTS,
        current,
        selected[-1],
    ):
        print(
            "{}：{:+.2f}° -> {:+.2f}°".format(
                name,
                math.degrees(start),
                math.degrees(target),
            )
        )

    print("\n本次只到预抓取点，不下降，不闭合夹爪。")

    confirm = input(
        "确认工作区无人、急停可用，输入 MOVE_PREGRASP："
    ).strip()

    if confirm != "MOVE_PREGRASP":
        print("确认词不正确，未发送任何命令。")
        return

    _, current_again = read_feedback()

    if max_error(current, current_again) > math.radians(1.0):
        raise RuntimeError(
            "确认期间机械臂位置发生变化，停止执行"
        )

    publisher = rospy.Publisher(
        COMMAND_TOPIC,
        JointState,
        queue_size=1,
    )

    deadline = time.time() + 3.0
    while (
        publisher.get_num_connections() == 0
        and time.time() < deadline
    ):
        rospy.sleep(0.05)

    if publisher.get_num_connections() == 0:
        raise RuntimeError(
            "/joint_states 没有订阅者"
        )

    for index, target in enumerate(selected, start=1):
        if not os.path.exists("/sys/class/net/can0"):
            raise RuntimeError(
                "执行过程中 can0 消失，停止发送后续目标"
            )

        send_target(publisher, target)
        error_deg = wait_reached(target)

        print(
            "小段 {}/{} 到位，最大误差 {:.2f}°".format(
                index,
                len(selected),
                error_deg,
            )
        )

    _, final_state = read_feedback()

    print("\n========== 到达预抓取点 ==========")
    print(
        "最终最大误差：{:.2f}°".format(
            math.degrees(
                max_error(final_state, selected[-1])
            )
        )
    )
    print("未执行 Cartesian 下降")
    print("未闭合夹爪")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        rospy.logerr("PREGRASP ABORTED: %s", error)
        raise
