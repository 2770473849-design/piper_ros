#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs  # 注册 geometry_msgs 的 tf2 转换
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped


DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/camera/aligned_depth_to_color/camera_info"
TARGET_FRAME = "arm_base"


def decode_depth(msg: Image) -> np.ndarray:
    """把 ROS 深度图转换为以米为单位的二维 numpy 数组。"""
    h, w = msg.height, msg.width

    if msg.encoding in ("16UC1", "mono16"):
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        return raw.astype(np.float32) * 0.001  # RealSense 16UC1 通常为毫米
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)

    raise RuntimeError(f"不支持的深度图编码：{msg.encoding}")


def choose_valid_pixel(depth_m: np.ndarray, requested_u=None, requested_v=None):
    """优先使用指定像素；无效时在附近搜索最近的有效深度点。"""
    h, w = depth_m.shape
    u0 = w // 2 if requested_u is None else int(requested_u)
    v0 = h // 2 if requested_v is None else int(requested_v)

    if not (0 <= u0 < w and 0 <= v0 < h):
        raise RuntimeError(f"像素 ({u0}, {v0}) 超出图像范围 0~{w-1}, 0~{h-1}")

    z0 = float(depth_m[v0, u0])
    if math.isfinite(z0) and z0 > 0.05:
        return u0, v0, z0

    # 中心点无深度时，在 41×41 邻域内找距离中心最近的有效像素
    radius = 20
    best = None
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            u = u0 + du
            v = v0 + dv
            if not (0 <= u < w and 0 <= v < h):
                continue
            z = float(depth_m[v, u])
            if not (math.isfinite(z) and z > 0.05):
                continue
            d2 = du * du + dv * dv
            if best is None or d2 < best[0]:
                best = (d2, u, v, z)

    if best is None:
        raise RuntimeError("指定像素及其附近没有有效深度，请把相机对准有深度数据的物体后重试。")

    _, u, v, z = best
    return u, v, z


def main():
    rospy.init_node("rgbd_pixel_to_base_test", anonymous=True)

    requested_u = int(sys.argv[1]) if len(sys.argv) >= 2 else None
    requested_v = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    rospy.loginfo("等待相机内参...")
    info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=5.0)

    rospy.loginfo("等待对齐深度图...")
    depth_msg = rospy.wait_for_message(DEPTH_TOPIC, Image, timeout=5.0)

    depth_m = decode_depth(depth_msg)
    u, v, z = choose_valid_pixel(depth_m, requested_u, requested_v)

    fx = info.K[0]
    fy = info.K[4]
    cx = info.K[2]
    cy = info.K[5]

    # 针孔模型反投影：像素 + 深度 -> 相机光学坐标系三维点
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    camera_frame = depth_msg.header.frame_id or info.header.frame_id
    if not camera_frame:
        camera_frame = "camera_color_optical_frame"

    p_cam = PointStamped()
    p_cam.header.stamp = rospy.Time(0)
    p_cam.header.frame_id = camera_frame
    p_cam.point.x = x
    p_cam.point.y = y
    p_cam.point.z = z

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)

    p_base = tf_buffer.transform(
        p_cam,
        TARGET_FRAME,
        timeout=rospy.Duration(3.0),
    )

    print("\n===== RGB-D 像素到机械臂基座坐标测试 =====")
    print(f"深度图编码：{depth_msg.encoding}")
    print(f"图像尺寸：{depth_msg.width} × {depth_msg.height}")
    print(f"实际使用像素：(u={u}, v={v})")
    print(f"深度：{z:.6f} m")
    print(f"\n{camera_frame} 中的三维点：")
    print(f"x = {x:.6f} m")
    print(f"y = {y:.6f} m")
    print(f"z = {z:.6f} m")
    print(f"\n{TARGET_FRAME} 中的三维点：")
    print(f"x = {p_base.point.x:.6f} m")
    print(f"y = {p_base.point.y:.6f} m")
    print(f"z = {p_base.point.z:.6f} m")
    print("\n转换成功：RGB-D → camera_color_optical_frame → arm_base")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
