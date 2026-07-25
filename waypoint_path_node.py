#!/usr/bin/env python3
# Alternative /cmd_vel producer: consumes the same /gps/path waypoint list as
# fusion_navigator_node.py, but drives each leg via move_base action goals
# instead of a direct proportional controller (real costmap obstacle avoidance).
"""
Waypoint path follower (ROS1 / husky_catkin_ws) — real path planning via move_base.

Takes an ordered list of lat/lon waypoints (published as a JSON array on
/gps/path by map_app.py whenever the user commits a path from the map) and
drives through them in order by sending sequential move_base goals. Unlike
gps_navigator_node.py's direct /cmd_vel proportional controller, this
delegates routing to move_base's own global/local planner, so each leg gets
real collision-free path planning and real-time obstacle avoidance from the
costmap (built from /scan — see husky_navigation/move_base.launch
no_static_map:=true).

gps_navigator_node.py is left untouched and still works independently for
simple single-goal driving — this is a separate node/topic, not a
replacement.

Failure policy: if move_base fails to reach any waypoint (any terminal state
other than SUCCEEDED), the rest of the path is abandoned and the node goes
idle, waiting for a new path. It does not skip ahead on failure.

Frame note: matches gps_navigator_node.py's convention — the Husky's GPS
plugin and spawn point are both aligned with ENU at the world origin, so
lat/lon converts to local (x, y) once and that's used directly as the
move_base goal in the *odom* frame (no static map / no AMCL in this setup —
see costmap_global_laser.yaml, loaded when move_base.launch is run with
no_static_map:=true).
"""
import json
import math
import threading

import actionlib
import rospy
from std_msgs.msg import String
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped

EARTH_RADIUS_M = 6371000.0


def latlon_to_local_xy(lat_deg, lon_deg, anchor_lat_deg, anchor_lon_deg):
    """Equirectangular projection around the anchor — same math as
    gps_navigator_node.py, kept duplicated rather than imported since these
    are two independently runnable scripts, not a shared package."""
    lat0 = math.radians(anchor_lat_deg)
    x_east = EARTH_RADIUS_M * math.radians(lon_deg - anchor_lon_deg) * math.cos(lat0)
    y_north = EARTH_RADIUS_M * math.radians(lat_deg - anchor_lat_deg)
    return x_east, y_north


def make_goal(x, y, frame_id):
    """Build a MoveBaseGoal at (x, y) with identity orientation — waypoints
    only constrain position, not final heading, matching
    gps_navigator_node.py's behavior (it doesn't manage heading either)."""
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = frame_id
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0
    return goal


class WaypointPathNode:
    def __init__(self):
        gp = rospy.get_param
        self.anchor_lat = gp('~anchor_lat', 31.781306)  # local-xy projection origin; matches map_app.py's anchor
        self.anchor_lon = gp('~anchor_lon', 76.997611)
        self.frame_id = gp('~frame_id', 'odom')  # move_base goal frame; no static map/AMCL in this setup
        path_topic = gp('~path_topic', '/gps/path')

        self._lock = threading.Lock()
        self._latest_path = []          # list of (x, y), set by path_cb
        self._new_path_event = threading.Event()

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo('Waiting for move_base action server...')
        self.client.wait_for_server()
        rospy.loginfo('move_base action server found.')

        rospy.Subscriber(path_topic, String, self.path_cb, queue_size=1)

        self._worker_thread = threading.Thread(target=self._run_paths, daemon=True)
        self._worker_thread.start()

        rospy.loginfo('Waypoint path node ready, waiting for a path on %s.', path_topic)

    def path_cb(self, msg: String):
        try:
            waypoints = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            rospy.logwarn('waypoint_path_node: could not parse path JSON: %r', msg.data)
            return

        local_path = []
        for wp in waypoints:
            x, y = latlon_to_local_xy(wp['latitude'], wp['longitude'],
                                      self.anchor_lat, self.anchor_lon)
            local_path.append((x, y))

        with self._lock:
            self._latest_path = local_path
        rospy.loginfo('New path received: %d waypoint(s).', len(local_path))
        self._new_path_event.set()      # (re)start the worker on this path

    def _run_paths(self):
        while not rospy.is_shutdown():
            self._new_path_event.wait()
            self._new_path_event.clear()

            with self._lock:
                path = list(self._latest_path)

            self.client.cancel_all_goals()

            for i, (x, y) in enumerate(path):
                if self._new_path_event.is_set():
                    rospy.loginfo('Newer path arrived — abandoning current path.')
                    break

                rospy.loginfo('Sending waypoint %d/%d: (%.2f, %.2f)',
                             i + 1, len(path), x, y)
                self.client.send_goal(make_goal(x, y, self.frame_id))
                self.client.wait_for_result()
                state = self.client.get_state()

                if state != GoalStatus.SUCCEEDED:
                    rospy.logwarn('Waypoint %d/%d failed (state=%d) — stopping path.',
                                 i + 1, len(path), state)
                    break
            else:
                rospy.loginfo('Path complete — all %d waypoint(s) reached.', len(path))


if __name__ == '__main__':
    rospy.init_node('waypoint_path_node')
    node = WaypointPathNode()
    rospy.spin()
