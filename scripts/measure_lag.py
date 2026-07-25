#!/usr/bin/env python3
"""Measure inter-frame lag: time between consecutive /cmd_vel messages (one per processed frame)."""
# Usage: python3 measure_lag.py [N]  -- collects N+1 /cmd_vel messages, then
# prints the N inter-message deltas and their average (i.e. pipeline rate).
import sys
import rospy
from geometry_msgs.msg import Twist

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # number of lag samples (deltas) to collect; needs N+1 messages
timestamps = []  # wall-clock arrival time of each /cmd_vel message


def cb(msg):
    timestamps.append(rospy.Time.now().to_sec())
    if len(timestamps) > N + 1:
        rospy.signal_shutdown('done')


rospy.init_node('lag_measurer', anonymous=True)
rospy.Subscriber('/cmd_vel', Twist, cb, queue_size=10)

rospy.loginfo(f'Collecting {N + 1} frames to compute {N} deltas...')
while not rospy.is_shutdown() and len(timestamps) <= N:
    rospy.sleep(0.05)

deltas = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]  # time gap between consecutive messages
avg = sum(deltas) / len(deltas)  # average inter-frame lag in seconds

print('\nPer-frame lag (s):', [f'{d:.3f}' for d in deltas])
print(f'Average lag over {len(deltas)} frames: {avg:.3f} seconds  ({1/avg:.2f} Hz)')
