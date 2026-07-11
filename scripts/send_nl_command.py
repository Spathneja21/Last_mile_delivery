#!/usr/bin/env python3
"""
send_nl_command.py
Tiny interactive CLI: type a plain-English driving command, it gets published
to /vision_pilot/nl_command for nl_command_node.py to parse and act on.

Usage:
  ./run_send_nl_command.sh
  > slow down
  > stop
  > turn left for 2 seconds
  > (Ctrl-D to quit)
"""
import rospy
from std_msgs.msg import String


def main():
    rospy.init_node('send_nl_command', anonymous=True)
    command_topic = rospy.get_param('~command_topic', '/vision_pilot/nl_command')
    pub = rospy.Publisher(command_topic, String, queue_size=10)

    rospy.sleep(0.5)  # give the publisher a moment to register with the master
    print(f'Publishing typed commands to {command_topic}. Ctrl-D to quit.')

    try:
        while not rospy.is_shutdown():
            try:
                text = input('> ').strip()
            except EOFError:
                print()
                break
            if not text:
                continue
            pub.publish(String(data=text))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
