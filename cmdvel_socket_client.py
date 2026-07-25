#!/usr/bin/env python3
# ROS1 bridge node: subscribes to /cmd_vel and forwards each velocity command
# to the Husky robot over a raw TCP socket, as a "linear_x,angular_z\n" CSV
# line, so the robot-side listener can drive the motors from it.
# (Same role as send_command.py, targeting a different Husky unit's IP.)

import socket
import rospy
from geometry_msgs.msg import Twist

HUSKY_IP = "192.168.1.216"  # Husky's TCP listener IP on the robot network
PORT = 5005                 # must match the port the robot-side listener binds to


class CmdVelClient:

    def __init__(self):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # persistent TCP connection to the Husky

        while not rospy.is_shutdown():
            try:
                self.sock.connect((HUSKY_IP, PORT))
                rospy.loginfo("Connected to Husky")
                break
            except:
                rospy.logwarn("Waiting for Husky...")
                rospy.sleep(2)

        rospy.Subscriber("/cmd_vel", Twist, self.callback)

    def callback(self, msg):

        data = "{:.6f},{:.6f}\n".format(  # CSV line: "linear_x,angular_z\n" expected by the robot-side listener
            msg.linear.x,
            msg.angular.z
        )

        try:
            self.sock.sendall(data.encode())

        except Exception as e:
            rospy.logerr(e)


if __name__ == "__main__":

    rospy.init_node("cmdvel_socket_client")

    CmdVelClient()

    rospy.spin()