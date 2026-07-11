#!/usr/bin/env python3

import socket
import rospy
from geometry_msgs.msg import Twist

HUSKY_IP = "192.168.1.216"
PORT = 5005


class CmdVelClient:

    def __init__(self):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

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

        data = "{:.6f},{:.6f}\n".format(
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