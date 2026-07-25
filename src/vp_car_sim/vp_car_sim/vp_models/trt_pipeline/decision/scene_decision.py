"""
SceneSeg + Scene3D reactive obstacle-avoidance decision logic.

Pure functions (no ROS, no vehicle-interface dependencies) so they can be unit-tested
and reused by any node that has SceneSeg/Scene3D predictions for a frame, regardless
of what it's driving (or not driving) with the result. Originally written for
awsim_navigator_node.py - moved here so non-AWSIM consumers (e.g. webcam_navigator_node.py)
don't need to import autoware_control_msgs/autoware_vehicle_msgs just to reach these.
"""
import math

import numpy as np
import cv2

CLS_FOREGROUND = 1  # SceneSeg class id for obstacle/foreground pixels
CLS_ROAD = 2  # SceneSeg class id for drivable road pixels


def compute_command(seg_pred, depth_pred, p):
    """
    seg_pred   : (H, W) int array of SceneSeg class ids
    depth_pred : (H, W) float array of Scene3D raw relative depth (higher = nearer,
                 confirmed empirically - see AWSIM_INTEGRATION_PLAN.md S4.4a)
    p          : dict of parameters (see node defaults)

    Returns (linear_x, angular_z, info) where info has per-bin debug fields.
    """
    h, w = seg_pred.shape
    top = int(p['roi_top_frac'] * h)  # row where the region-of-interest (ignore sky/hood) starts
    roi_seg = seg_pred[top:, :]
    roi_depth = depth_pred[top:, :]

    # Relative depth only - normalize within the ROI, per frame (no fixed scale exists).
    d_min, d_max = float(roi_depth.min()), float(roi_depth.max())
    norm_depth = (roi_depth - d_min) / max(d_max - d_min, 1e-6)

    n = p['num_bins']  # number of vertical column bins the frame width is split into
    cols = np.array_split(np.arange(w), n)  # column-index groups, one per bin
    half_fov = math.radians(p['camera_hfov_deg']) / 2.0  # half of camera horizontal field of view, radians

    bin_bearing = np.zeros(n)      # steering bearing (rad) toward the center of each bin
    fg_frac = np.zeros(n)          # fraction of foreground/obstacle pixels in each bin
    road_frac = np.zeros(n)        # fraction of road pixels in each bin
    fg_proximity = np.zeros(n)     # nearest obstacle pixel in this bin, 0 if none
    for i, c in enumerate(cols):
        seg = roi_seg[:, c[0]:c[-1] + 1]  # class-id slice of this bin's columns
        depth = norm_depth[:, c[0]:c[-1] + 1]  # normalized-depth slice of this bin's columns
        fg_mask = (seg == CLS_FOREGROUND)  # boolean mask of obstacle pixels within the bin
        fg_frac[i] = float(np.mean(fg_mask))
        road_frac[i] = float(np.mean(seg == CLS_ROAD))
        fg_proximity[i] = float(depth[fg_mask].max()) if np.any(fg_mask) else 0.0
        u = (((c[0] + c[-1]) / 2.0) - w / 2.0) / (w / 2.0)      # bin center column, normalized to image half-width and
                                                                # recentered on the image midline: -1 = left edge,
                                                                # 0 = straight ahead (image center), +1 = right edge
        bin_bearing[i] = -u * half_fov                          # +bearing = left = +yaw

    # Depth-weighted obstacle penalty: a near obstacle (fg_proximity -> 1) penalizes a
    # bin up to 2x harder than a far one (fg_proximity -> 0) of the same pixel area.
    score = (p['road_weight'] * road_frac
             - p['obstacle_weight'] * fg_frac * (0.5 + 0.5 * fg_proximity))

    best = int(np.argmax(score))  # index of the highest-scoring (most drivable) bin
    desired_bearing = float(bin_bearing[best])

    # Central third of the bins = path straight ahead.
    lo, hi = n // 3, n - n // 3  # the middle 3 bins
    center_fg = float(np.max(fg_frac[lo:hi])) if hi > lo else float(fg_frac[best])  # worst obstacle coverage straight ahead
    center_proximity = float(np.max(fg_proximity[lo:hi])) if hi > lo else float(fg_proximity[best])  # closest obstacle straight ahead

    angular = p['kp_steer'] * desired_bearing  # proportional steering gain applied to bearing error
    angular = max(-p['max_angular_speed'], min(p['max_angular_speed'], angular)) #putting minimax filter to keep angular speed in range [-1 rad/s to 1 rad/s]

    area_blocked = center_fg >= p['blocked_stop_fraction']  # too much obstacle area ahead -> stop
    depth_blocked = center_proximity >= p['brake_depth_threshold']  # obstacle too close ahead -> stop
    blocked = area_blocked or depth_blocked

    block_amount = max(center_fg / max(1e-3, p['blocked_stop_fraction']),
                        center_proximity / max(1e-3, p['brake_depth_threshold']))
    block_amount = min(1.0, block_amount)  # 0 = clear, 1 = fully blocked; scales down linear speed
    linear = p['max_linear_speed'] * (1.0 - block_amount)
    if blocked or abs(desired_bearing) > p['rotate_in_place_angle']:
        linear = 0.0
    if blocked:
        angular = 0.0

    info = {
        'fg_frac': fg_frac, 'road_frac': road_frac, 'fg_proximity': fg_proximity,
        'bin_bearing': bin_bearing, 'score': score, 'best': best, 'cols': cols,
        'desired_bearing': desired_bearing,
        'center_fg': center_fg, 'center_proximity': center_proximity,
        'area_blocked': area_blocked, 'depth_blocked': depth_blocked, 'blocked': blocked,
    }
    return linear, angular, info


def overlay_image(seg_pred, depth_pred, info, p):
    """Road=green, obstacle red-intensity scaled by proximity, chosen bin highlighted,
    BRAKE/CLEAR banner."""
    h, w = seg_pred.shape
    top = int(p['roi_top_frac'] * h)
    norm_depth = np.zeros((h, w), dtype=np.float32)
    roi_depth = depth_pred[top:, :]
    d_min, d_max = float(roi_depth.min()), float(roi_depth.max())
    norm_depth[top:, :] = (roi_depth - d_min) / max(d_max - d_min, 1e-6)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[seg_pred == CLS_ROAD] = (40, 180, 40)
    img[seg_pred == 0] = (60, 60, 60)
    fg_mask = seg_pred == CLS_FOREGROUND
    # Closer obstacle pixels -> brighter red; farther ones -> dimmer red.
    red = (80 + 175 * norm_depth).astype(np.uint8)
    img[fg_mask] = np.stack([red[fg_mask], np.zeros_like(red[fg_mask]), np.zeros_like(red[fg_mask])], axis=-1)

    cols = info['cols'][info['best']]
    c0, c1 = cols[0], cols[-1]
    img[:, c0:c1 + 1, :] = (img[:, c0:c1 + 1, :] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)

    banner = (0, 0, 200) if info['blocked'] else (0, 160, 0)
    img[0:24, :, :] = banner
    label = f"BRAKE area={info['center_fg']:.2f} prox={info['center_proximity']:.2f}" \
        if info['blocked'] else f"CLEAR area={info['center_fg']:.2f} prox={info['center_proximity']:.2f}"
    cv2.putText(img, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
