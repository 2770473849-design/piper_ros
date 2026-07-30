#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import rospy

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


OBJECT_NAME = "anygrasp_test_cube"

# 先只允许两根手指接触目标物体。
TOUCH_LINKS = [
    "link7",
    "link8",
]


def ensure_acm_name(acm, name):
    """Ensure that a name exists as one row and column in the ACM."""

    if name in acm.entry_names:
        return acm.entry_names.index(name)

    acm.entry_names.append(name)

    # 给已有的每一行添加新的一列。
    for entry in acm.entry_values:
        entry.enabled.append(False)

    # 创建新名字对应的一整行。
    new_entry = AllowedCollisionEntry()
    new_entry.enabled = [
        False
        for _ in acm.entry_names
    ]
    acm.entry_values.append(new_entry)

    return len(acm.entry_names) - 1


def set_allowed_pair(acm, first_name, second_name):
    first_index = ensure_acm_name(
        acm,
        first_name,
    )
    second_index = ensure_acm_name(
        acm,
        second_name,
    )

    acm.entry_values[
        first_index
    ].enabled[second_index] = True

    acm.entry_values[
        second_index
    ].enabled[first_index] = True


def main():
    rospy.init_node(
        "allow_target_touch_once"
    )

    get_scene_service = rospy.ServiceProxy(
        "/get_planning_scene",
        GetPlanningScene,
    )

    apply_scene_service = rospy.ServiceProxy(
        "/apply_planning_scene",
        ApplyPlanningScene,
    )

    rospy.loginfo(
        "Waiting for MoveIt planning-scene services..."
    )

    rospy.wait_for_service(
        "/get_planning_scene",
        timeout=10.0,
    )
    rospy.wait_for_service(
        "/apply_planning_scene",
        timeout=10.0,
    )

    get_request = GetPlanningSceneRequest()
    get_request.components.components = (
        PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    )

    response = get_scene_service(
        get_request
    )

    acm = response.scene.allowed_collision_matrix

    for link_name in TOUCH_LINKS:
        set_allowed_pair(
            acm,
            OBJECT_NAME,
            link_name,
        )

        rospy.loginfo(
            "Allowing expected contact: %s <-> %s",
            OBJECT_NAME,
            link_name,
        )

    planning_scene = PlanningScene()
    planning_scene.is_diff = True
    planning_scene.allowed_collision_matrix = acm

    apply_request = ApplyPlanningSceneRequest()
    apply_request.scene = planning_scene

    apply_response = apply_scene_service(
        apply_request
    )

    if not apply_response.success:
        raise RuntimeError(
            "MoveIt rejected the Allowed Collision Matrix update."
        )

    rospy.loginfo(
        "Target-object touch permissions applied successfully."
    )

    rospy.loginfo(
        "Only %s and %s may touch %s.",
        TOUCH_LINKS[0],
        TOUCH_LINKS[1],
        OBJECT_NAME,
    )


if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass

    except Exception as error:
        rospy.logerr(
            "Failed to update allowed collisions: %s",
            error,
        )
        sys.exit(1)