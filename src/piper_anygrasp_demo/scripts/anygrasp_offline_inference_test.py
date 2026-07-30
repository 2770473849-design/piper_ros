#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from argparse import Namespace

import numpy as np


def load_anygrasp_api():
    """
    当前PointNet2扩展的实际模块名是pointnet2._ext，
    但AnyGrasp内部会导入pointnet2_ext，因此建立兼容别名。
    """
    from pointnet2 import _ext

    sys.modules.setdefault(
        "pointnet2_ext",
        _ext,
    )

    from gsnet import create_detector

    return create_detector


def adapt_rotation_to_piper(anygrasp_rotation):
    """
    AnyGrasp:
      local +X = approach
      local +Y = gripper opening direction

    Piper当前执行约定:
      local +Z = approach
      local +Y = gripper opening direction

    映射：
      Piper +X = -AnyGrasp +Z
      Piper +Y =  AnyGrasp +Y
      Piper +Z =  AnyGrasp +X
    """
    return np.column_stack(
        (
            -anygrasp_rotation[:, 2],
            anygrasp_rotation[:, 1],
            anygrasp_rotation[:, 0],
        )
    )


def validate_rotation(rotation, name):
    determinant = float(
        np.linalg.det(rotation)
    )

    orthogonality_error = float(
        np.linalg.norm(
            rotation.T @ rotation
            - np.eye(3)
        )
    )

    if abs(determinant - 1.0) > 1.0e-4:
        raise RuntimeError(
            "{} determinant is invalid: {:.6f}".format(
                name,
                determinant,
            )
        )

    if orthogonality_error > 1.0e-4:
        raise RuntimeError(
            "{} is not orthonormal: {:.9f}".format(
                name,
                orthogonality_error,
            )
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run one offline AnyGrasp inference on the "
            "saved Gazebo XYZRGB point cloud."
        )
    )

    parser.add_argument(
        "--input",
        default=os.path.expanduser(
            "~/anygrasp_ws/captures/"
            "hand_camera_xyzrgb_latest.npz"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=os.path.expanduser(
            "~/anygrasp_ws/anygrasp_sdk/"
            "grasp_detection/log/"
            "checkpoint_detection.tar"
        ),
    )

    parser.add_argument(
        "--output",
        default=os.path.expanduser(
            "~/anygrasp_ws/captures/"
            "anygrasp_offline_results_latest.npz"
        ),
    )

    parser.add_argument(
        "--max-gripper-width",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--gripper-height",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Point-cloud sampling stride. "
            "Use 1 for the first test."
        ),
    )

    parser.add_argument(
        "--disable-collision-detection",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_path = os.path.expanduser(
        args.input
    )
    checkpoint_path = os.path.expanduser(
        args.checkpoint
    )
    output_path = os.path.expanduser(
        args.output
    )

    if not os.path.isfile(input_path):
        raise RuntimeError(
            "Input NPZ does not exist: {}".format(
                input_path
            )
        )

    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(
            "Checkpoint does not exist: {}".format(
                checkpoint_path
            )
        )

    print(
        "========== ANYGRASP OFFLINE INFERENCE =========="
    )

    data = np.load(input_path)

    points = np.ascontiguousarray(
        data["points"],
        dtype=np.float32,
    )

    region_mask = np.asarray(
        data["red_mask"],
        dtype=bool,
    )

    frame_id = str(
        data["frame_id"]
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(
            "points must have shape (N,3), got {}".format(
                points.shape
            )
        )

    if region_mask.ndim != 1:
        raise RuntimeError(
            "red_mask must be one-dimensional."
        )

    if points.shape[0] != region_mask.shape[0]:
        raise RuntimeError(
            "points and red_mask lengths differ: "
            "{} versus {}".format(
                points.shape[0],
                region_mask.shape[0],
            )
        )

    stride = max(
        1,
        int(args.stride),
    )

    if stride > 1:
        points = np.ascontiguousarray(
            points[::stride],
            dtype=np.float32,
        )

        region_mask = np.ascontiguousarray(
            region_mask[::stride],
            dtype=bool,
        )

    print("输入文件 =", input_path)
    print("坐标系 =", frame_id)
    print("推理点数 =", points.shape[0])
    print(
        "目标区域点数 =",
        int(region_mask.sum()),
    )
    print("点云类型 =", points.dtype)
    print("加载权重 =", checkpoint_path)

    if int(region_mask.sum()) == 0:
        raise RuntimeError(
            "Region steering mask contains no target points."
        )

    create_detector = load_anygrasp_api()

    config = Namespace(
        checkpoint_path=checkpoint_path,
        max_gripper_width=float(
            args.max_gripper_width
        ),
        gripper_height=float(
            args.gripper_height
        ),
    )

    print(
        "\n正在加载AnyGrasp detector，请等待……"
    )

    detector = create_detector(config)

    if detector is None:
        raise RuntimeError(
            "create_detector returned None."
        )

    print(
        "AnyGrasp detector加载完成：",
        type(detector),
    )

    optional_params = {
        "dense_grasp": False,
        "collision_detection": (
            not args.disable_collision_detection
        ),
        "region_steering": region_mask,
        "approach_steering": None,
        "approach_thresh": np.pi,
    }

    print(
        "\n开始执行 detector.get_grasp()……"
    )

    start_time = time.time()

    grasp_group = detector.get_grasp(
        points,
        optional_params,
    )

    elapsed = time.time() - start_time

    print(
        "推理耗时 = {:.3f} s".format(
            elapsed
        )
    )

    if grasp_group is None:
        print(
            "AnyGrasp返回None：没有候选通过当前推理和筛选。"
        )
        sys.exit(2)

    if len(grasp_group) == 0:
        print(
            "AnyGrasp返回空GraspGroup。"
        )
        sys.exit(2)

    print(
        "NMS前候选数量 =",
        len(grasp_group),
    )

    grasp_group = grasp_group.nms()
    grasp_group = grasp_group.sort_by_score()

    print(
        "NMS后候选数量 =",
        len(grasp_group),
    )

    top_count = min(
        max(1, int(args.top_n)),
        len(grasp_group),
    )

    translations = []
    anygrasp_rotations = []
    piper_rotations = []
    scores = []
    widths = []
    heights = []
    depths = []
    gripper_tips = []

    print(
        "\n========== TOP {} CANDIDATES ==========".format(
            top_count
        )
    )

    for index in range(top_count):
        grasp = grasp_group[index]

        translation = np.asarray(
            grasp.translation,
            dtype=np.float64,
        )

        rotation = np.asarray(
            grasp.rotation_matrix,
            dtype=np.float64,
        )

        validate_rotation(
            rotation,
            "AnyGrasp Candidate {}".format(
                index
            ),
        )

        piper_rotation = (
            adapt_rotation_to_piper(
                rotation
            )
        )

        validate_rotation(
            piper_rotation,
            "Piper Candidate {}".format(
                index
            ),
        )

        score = float(grasp.score)
        width = float(grasp.width)
        height = float(grasp.height)
        depth = float(grasp.depth)

        anygrasp_approach = rotation[:, 0]
        piper_approach = piper_rotation[:, 2]

        gripper_tip = (
            translation
            + depth * anygrasp_approach
        )

        mapping_error = float(
            np.linalg.norm(
                anygrasp_approach
                - piper_approach
            )
        )

        print(
            "\nCandidate {}:".format(
                index
            )
        )

        print(
            "  score = {:.6f}".format(
                score
            )
        )

        print(
            "  width = {:.4f} m".format(
                width
            )
        )

        print(
            "  height = {:.4f} m".format(
                height
            )
        )

        print(
            "  depth = {:.4f} m".format(
                depth
            )
        )

        print(
            "  translation = {}".format(
                np.array2string(
                    translation,
                    precision=4,
                    suppress_small=True,
                )
            )
        )

        print(
            "  AnyGrasp +X approach = {}".format(
                np.array2string(
                    anygrasp_approach,
                    precision=4,
                    suppress_small=True,
                )
            )
        )

        print(
            "  Piper +Z approach = {}".format(
                np.array2string(
                    piper_approach,
                    precision=4,
                    suppress_small=True,
                )
            )
        )

        print(
            "  axis mapping error = {:.9f}".format(
                mapping_error
            )
        )

        print(
            "  calculated gripper tip = {}".format(
                np.array2string(
                    gripper_tip,
                    precision=4,
                    suppress_small=True,
                )
            )
        )

        translations.append(
            translation
        )
        anygrasp_rotations.append(
            rotation
        )
        piper_rotations.append(
            piper_rotation
        )
        scores.append(score)
        widths.append(width)
        heights.append(height)
        depths.append(depth)
        gripper_tips.append(
            gripper_tip
        )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    np.savez_compressed(
        output_path,
        frame_id=np.array(frame_id),
        translations=np.asarray(
            translations,
            dtype=np.float64,
        ),
        anygrasp_rotation_matrices=np.asarray(
            anygrasp_rotations,
            dtype=np.float64,
        ),
        piper_rotation_matrices=np.asarray(
            piper_rotations,
            dtype=np.float64,
        ),
        scores=np.asarray(
            scores,
            dtype=np.float32,
        ),
        widths=np.asarray(
            widths,
            dtype=np.float32,
        ),
        heights=np.asarray(
            heights,
            dtype=np.float32,
        ),
        depths=np.asarray(
            depths,
            dtype=np.float32,
        ),
        gripper_tips=np.asarray(
            gripper_tips,
            dtype=np.float64,
        ),
        inference_time=np.array(
            elapsed,
            dtype=np.float64,
        ),
    )

    print(
        "\n结果已保存：",
        output_path,
    )

    print(
        "========== ANYGRASP INFERENCE COMPLETED =========="
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        pass

    except Exception as error:
        print(
            "AnyGrasp离线推理失败：",
            error,
            file=sys.stderr,
        )
        sys.exit(1)