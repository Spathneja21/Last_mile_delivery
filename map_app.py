import json
import os
import time

import rospy
from sensor_msgs.msg import NavSatFix # For GPS data
from std_msgs.msg import String, Empty
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

import road_network

app = Flask(__name__)
CORS(app) # Allows your HTML file to talk to this backend

# Anchor: user-provided reference point (31°46'52.7"N 76°59'51.4"E), within the
# Himachal Pradesh extract (map/planet_76.993,31.78_76.999,31.783.osm.pbf).
# Matches husky.urdf.xacro's GPS plugin / gps_navigator_node.py / waypoint_path_node.py.
# Global variable to store current position
current_pose = {"latitude": 31.781306, "longitude": 76.997611}

_PBF_PATH = os.path.join(os.path.dirname(__file__), 'map',
                         'planet_76.993,31.78_76.999,31.783.osm.pbf')
road_graph = road_network.load_graph(_PBF_PATH)

# Precomputed once at startup for the /api/road-nodes and /api/road-edges
# visualization endpoints — the graph is static (loaded from a cached file),
# no need to recompute per-request.
road_nodes = [{"latitude": data['lat'], "longitude": data['lon']}
             for _, data in road_graph.nodes(data=True)]

road_edges = [
    {
        "from": {"latitude": road_graph.nodes[u]['lat'], "longitude": road_graph.nodes[u]['lon']},
        "to": {"latitude": road_graph.nodes[v]['lat'], "longitude": road_graph.nodes[v]['lon']},
    }
    for u, v in road_graph.edges()
]

# Target the robot should navigate to. None until the user sets one from the map.
current_target = None

# Destination (lat, lon) of the currently staged/active path — remembered so
# the fusion navigator's /replan_request can reroute from the robot's CURRENT
# position to the SAME destination after an obstacle detour, with no new click.
current_destination = None

# Spacing (metres) the road route is densified to before it's sent — gives the
# fusion navigator a fine reference line so cross-track deviation off an
# obstacle detour is measurable (Dijkstra road nodes alone are ~15 m apart).
_PATH_SPACING_M = 2.0

# Ordered list of waypoints for the current path (multi-point navigation via
# waypoint_path_node.py / move_base). Empty until the user commits a path.
current_path = []

# Actual GPS trail the robot has physically traveled — recorded at ~1 Hz
# (the GPS plugin itself publishes at 40 Hz, far denser than useful for a
# map polyline) so it can be drawn alongside the planned red path to see
# whether the robot is actually following it or diverging.
robot_trail = []
_TRAIL_MIN_INTERVAL_S = 1.0
_last_trail_time = 0.0

# Set once run_ros() has called rospy.init_node(), so Flask (main thread) can
# publish through it. rospy publishers are safe to call from other threads.
target_pub = None
path_pub = None

def gps_callback(msg):
    global current_pose, _last_trail_time
    # Safely update the global dictionary with real telemetry data
    current_pose["latitude"] = msg.latitude
    current_pose["longitude"] = msg.longitude

    now = time.time()
    if now - _last_trail_time >= _TRAIL_MIN_INTERVAL_S:
        robot_trail.append({"latitude": msg.latitude, "longitude": msg.longitude})
        _last_trail_time = now

@app.route('/api/robot-pose', methods=['GET'])
def get_robot_pose():
    return jsonify(current_pose)

@app.route('/api/target', methods=['GET'])
def get_target():
    return jsonify(current_target)

@app.route('/api/set-target', methods=['POST'])
def set_target():
    global current_target
    data = request.get_json(force=True)
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])
    current_target = {"latitude": latitude, "longitude": longitude}
    if target_pub is not None:
        msg = NavSatFix()
        msg.latitude = latitude
        msg.longitude = longitude
        target_pub.publish(msg)
    return jsonify(current_target)

def _build_path(dest):
    """Route from the robot's CURRENT position to dest=(lat, lon) along the
    road graph (Dijkstra shortest path), densified to _PATH_SPACING_M. Falls
    back to a straight line if no road path connects them (disconnected
    components / click outside the mapped area). Returns a waypoint-dict list.

    Shared by /api/set-path (preview) and the /replan_request handler (reroute
    mid-drive) so both produce identically-shaped, equally-dense paths."""
    start = (current_pose['latitude'], current_pose['longitude'])
    leg = road_network.route_latlon(road_graph, start, dest)
    if leg is None:
        leg = [start, dest]
    else:
        # route_latlon() starts at the nearest ROAD NODE, not the robot's
        # actual position (they're rarely the same point — this extract has
        # the robot's spawn ~27m from the nearest road). Prepend the real
        # start so the drawn/sent path visibly includes that initial off-road
        # leg instead of silently skipping it.
        leg = [start] + leg

    dense = road_network.densify_latlon(leg, spacing_m=_PATH_SPACING_M)
    return [{"latitude": lat, "longitude": lon} for lat, lon in dense]


@app.route('/api/set-path', methods=['POST'])
def set_path():
    """Compute the road route from the robot's current position to the
    clicked destination and stage it for preview — does NOT send anything to
    the robot yet. The frontend draws this in red; the robot only starts
    moving once /api/approve-path is called (the "Approve" button)."""
    global current_path, current_destination, _last_trail_time
    robot_trail.clear()   # fresh destination -> fresh trail to compare against this attempt
    _last_trail_time = 0.0   # don't let the old throttle timer delay the first new point
    data = request.get_json(force=True)
    dest = (float(data['latitude']), float(data['longitude']))

    # Remember the destination so an obstacle-triggered /replan_request can
    # reroute to it from the robot's new position without another click.
    current_destination = dest
    current_path = _build_path(dest)
    return jsonify(current_path)

@app.route('/api/approve-path', methods=['POST'])
def approve_path():
    """Send the currently staged path (from the last /api/set-path call) to
    the robot. This is the actual "go" trigger — nothing drives before this."""
    if not current_path:
        return jsonify({"error": "no path staged — click the map first"}), 400
    if path_pub is not None:
        path_pub.publish(String(data=json.dumps(current_path)))
    return jsonify({"status": "approved", "waypoints": len(current_path)})

@app.route('/api/path', methods=['GET'])
def get_path():
    return jsonify(current_path)

@app.route('/api/road-nodes', methods=['GET'])
def get_road_nodes():
    return jsonify(road_nodes)

@app.route('/api/road-edges', methods=['GET'])
def get_road_edges():
    return jsonify(road_edges)

@app.route('/api/robot-trail', methods=['GET'])
def get_robot_trail():
    return jsonify(robot_trail)

@app.route('/api/clear-trail', methods=['POST'])
def clear_trail():
    global _last_trail_time
    robot_trail.clear()
    _last_trail_time = 0.0
    return jsonify({"status": "cleared"})

def replan_callback(_msg):
    """The fusion navigator asks for a reroute when an obstacle detour has
    pushed the robot off the planned line. Recompute from the robot's CURRENT
    position to the stored destination and publish it straight to /gps/path —
    this is an automatic reroute mid-drive, so it does NOT wait for the Approve
    button (the robot is already moving toward this destination)."""
    global current_path
    if current_destination is None:
        rospy.logwarn('replan requested but no destination stored — ignoring.')
        return
    current_path = _build_path(current_destination)
    if path_pub is not None:
        path_pub.publish(String(data=json.dumps(current_path)))
    rospy.loginfo('Replanned from current position: %d waypoint(s).', len(current_path))


def run_ros():
    global target_pub, path_pub
    # disable_signals: rospy's default SIGINT handler can only be installed from
    # the main thread, and this runs on a background thread so Flask can own main.
    rospy.init_node('web_bridge_node', anonymous=False, disable_signals=True)
    rospy.Subscriber('/navsat/fix', NavSatFix, gps_callback)
    rospy.Subscriber('/replan_request', Empty, replan_callback)
    target_pub = rospy.Publisher('/gps/target', NavSatFix, queue_size=10)
    path_pub = rospy.Publisher('/gps/path', String, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    # Run the ROS subscriber in a separate thread so Flask can run simultaneously
    threading.Thread(target=run_ros, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)