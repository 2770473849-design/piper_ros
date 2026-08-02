#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState

class RealJointStateAdapter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/joint_states_single")
        self.output_topic = rospy.get_param("~output_topic", "/joint_states_calibration")
        self.arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint = rospy.get_param("~gripper_joint", "gripper")
        self.publisher = rospy.Publisher(self.output_topic, JointState, queue_size=20)
        self.subscriber = rospy.Subscriber(self.input_topic, JointState, self.callback, queue_size=50)
        rospy.loginfo("Real joint-state adapter ready: %s -> %s", self.input_topic, self.output_topic)

    def callback(self, msg):
        indices = {name: index for index, name in enumerate(msg.name)}
        required = self.arm_joints + [self.gripper_joint]
        missing = [name for name in required if name not in indices]
        if missing:
            rospy.logwarn_throttle(2.0, "Missing real joints: %s", missing)
            return

        if len(msg.position) != len(msg.name):
            rospy.logwarn_throttle(2.0, "Invalid JointState position array")
            return

        opening = float(msg.position[indices[self.gripper_joint]])
        opening = max(0.0, min(0.1, opening))
        finger_position = opening / 2.0

        output = JointState()
        output.header = msg.header
        output.name = self.arm_joints + ["joint7", "joint8"]
        output.position = [msg.position[indices[name]] for name in self.arm_joints]
        output.position += [finger_position, -finger_position]

        if len(msg.velocity) == len(msg.name):
            gripper_velocity = float(msg.velocity[indices[self.gripper_joint]]) / 2.0
            output.velocity = [msg.velocity[indices[name]] for name in self.arm_joints]
            output.velocity += [gripper_velocity, -gripper_velocity]

        self.publisher.publish(output)

if __name__ == "__main__":
    rospy.init_node("real_joint_state_adapter")
    RealJointStateAdapter()
    rospy.spin()
