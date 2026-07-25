#!/data/archit0030/miniforge3/envs/lerobot/bin/python3
"""
Natural-language command parser node.

ROS 1 (Noetic).

Subscribes to /vision_pilot/nl_command (plain English text), calls the
Gemini API to translate it into a small fixed-vocabulary JSON intent, then
validates/clamps the result before publishing on /vision_pilot/nl_intent.

Design intent: this node NEVER emits velocities directly. It only produces a
bounded, validated intent that webcam_navigator_node applies as a
post-processing step on top of its own unmodified obstacle-avoidance logic.
A malformed or nonsensical command degrades to "nothing changes", not to an
unsafe motion command.
"""
import os
import sys
import json
import time

import requests
import rospy
from std_msgs.msg import String

GEMINI_MODEL = 'gemini-2.5-flash'
GEMINI_URL = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'

# Must match yolo_detector_node's model classes (YOLOv8n, COCO-pretrained).
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

SYSTEM_PROMPT = f"""You translate a short spoken driving command into exactly one JSON object, no prose, no markdown fences.
Allowed schemas (pick exactly one shape):
{{"action": "stop"}}
{{"action": "go"}}
{{"action": "set_speed_scale", "scale": <float 0.1-2.0>}}
{{"action": "turn", "direction": "left"|"right", "duration_s": <float 0.5-5.0>}}
{{"action": "find_and_stop", "target_class": "<one of the known object classes>"}}
{{"action": "unknown", "reason": "<short reason>"}}

Rules for set_speed_scale:
- If the command gives an explicit number/percentage, use it directly (e.g. "half speed" -> 0.5, "double speed" -> 2.0).
- If the command is a vague relative phrase with no number, use a sensible default:
  "slow down" / "slower" / "ease up" -> 0.6
  "slow down a lot" / "much slower"  -> 0.3
  "speed up" / "faster" / "go faster" -> 1.4
  "speed up a lot" / "much faster"  -> 1.8
Rules for turn: if no duration is given, default duration_s to 1.5.

Rules for find_and_stop:
- Use this for commands like "find a mouse and stop", "look for a person then stop",
  "drive until you see a bottle and stop".
- target_class MUST be exactly one string from this known object vocabulary (the object
  detector cannot recognize anything outside this list):
  {', '.join(COCO_CLASSES)}
- Map the command's target noun to the closest matching class in that list (e.g. "phone" ->
  "cell phone", "TV" -> "tv"). If nothing in the command reasonably maps to a class in that
  list, use "unknown" instead of guessing a class that isn't in the list.

If the command doesn't clearly map to one of the schemas above (e.g. unrelated small talk), use "unknown"."""

# Hard server-side bounds — enforced regardless of what the model returns.
SCALE_MIN, SCALE_MAX = 0.2, 1.5  # allowed range for set_speed_scale
TURN_DURATION_MAX = 5.0  # seconds, cap on requested turn duration
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # min YOLO confidence to accept a find_and_stop match
NO_DETECTIONS_WARNING_DELAY = 5.0  # seconds to wait before warning that no detections arrived for find_and_stop

RETRYABLE_STATUS = {429, 500, 503}  # HTTP codes worth retrying the Gemini call on
MAX_RETRIES = 2


class NLCommandNode:
    def __init__(self):
        rospy.init_node('nl_command_node', anonymous=False)

        self.api_key = os.environ.get('GEMINI_API_KEY', '')
        if not self.api_key:
            rospy.logfatal('GEMINI_API_KEY environment variable is not set.')
            sys.exit(1)

        command_topic    = rospy.get_param('~command_topic', '/vision_pilot/nl_command')  # raw English text in
        intent_topic     = rospy.get_param('~intent_topic', '/vision_pilot/nl_intent')  # validated JSON intent out
        detections_topic = rospy.get_param('~detections_topic',
                                           '/vision_pilot/yolo_detector/detections')

        self.intent_pub = rospy.Publisher(intent_topic, String, queue_size=10)
        rospy.Subscriber(command_topic, String, self.command_cb, queue_size=10)

        # --- "find_and_stop" watch state — resolved against live YOLO detections.
        self.watch_target = None       # COCO class name currently being searched for, or None if idle
        self._detection_count = 0      # count of detection messages seen since the current watch started
        rospy.Subscriber(detections_topic, String, self.detections_cb, queue_size=10)

        rospy.loginfo(f'Listening for commands on {command_topic}, '
                      f'publishing intents on {intent_topic}, '
                      f'watching detections on {detections_topic} (model={GEMINI_MODEL}).')

    # ------------------------------------------------------------------
    def _call_gemini(self, text):
        url = f'{GEMINI_URL}?key={self.api_key}'
        payload = {
            'systemInstruction': {'parts': [{'text': SYSTEM_PROMPT}]},
            'contents': [{'parts': [{'text': text}]}],
            'generationConfig': {'responseMimeType': 'application/json'},
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, timeout=15)
            except requests.RequestException as e:
                rospy.logwarn(f'Gemini request failed ({e}); attempt {attempt + 1}')
                time.sleep(1.0)
                continue

            if resp.status_code == 200:
                data = resp.json()
                return data['candidates'][0]['content']['parts'][0]['text']

            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                retry_delay = 2.0
                try:
                    details = resp.json()['error'].get('details', [])
                    for d in details:
                        if d.get('@type', '').endswith('RetryInfo'):
                            retry_delay = float(str(d['retryDelay']).rstrip('s'))
                except Exception:
                    pass
                rospy.logwarn(f'Gemini returned {resp.status_code}, retrying in '
                             f'{retry_delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})')
                time.sleep(retry_delay)
                continue

            rospy.logwarn(f'Gemini call failed permanently: {resp.status_code} {resp.text[:200]}')
            return None

        rospy.logwarn('Gemini call exhausted retries.')
        return None

    # ------------------------------------------------------------------
    def _validate_and_clamp(self, raw_text):
        """Parse + sanitize the model's JSON. Returns a safe intent dict,
        or an {"action": "unknown"} dict if anything is malformed."""
        try:
            intent = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return {'action': 'unknown', 'reason': 'model did not return valid JSON'}

        action = intent.get('action')

        if action in ('stop', 'go'):
            return {'action': action}

        if action == 'set_speed_scale':
            try:
                scale = float(intent['scale'])
            except (KeyError, TypeError, ValueError):
                return {'action': 'unknown', 'reason': 'missing/invalid scale'}
            scale = max(SCALE_MIN, min(SCALE_MAX, scale))
            return {'action': 'set_speed_scale', 'scale': scale}

        if action == 'turn':
            direction = intent.get('direction')
            if direction not in ('left', 'right'):
                return {'action': 'unknown', 'reason': 'missing/invalid turn direction'}
            try:
                duration = float(intent.get('duration_s', 1.5))
            except (TypeError, ValueError):
                duration = 1.5
            duration = max(0.1, min(TURN_DURATION_MAX, duration))
            return {'action': 'turn', 'direction': direction, 'duration_s': duration}

        if action == 'find_and_stop':
            target_class = intent.get('target_class')
            if target_class not in COCO_CLASSES:
                return {'action': 'unknown',
                       'reason': f'target_class {target_class!r} not in known object vocabulary'}
            return {'action': 'find_and_stop', 'target_class': target_class}

        return {'action': 'unknown', 'reason': intent.get('reason', 'unrecognized action')}

    # ------------------------------------------------------------------
    def _publish_intent(self, intent):
        rospy.loginfo(f'Publishing intent: {intent}')
        self.intent_pub.publish(String(data=json.dumps(intent)))

    # ------------------------------------------------------------------
    def command_cb(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        rospy.loginfo(f'Command received: {text!r}')
        raw_reply = self._call_gemini(text)

        if raw_reply is None:
            intent = {'action': 'unknown', 'reason': 'LLM call failed'}
        else:
            intent = self._validate_and_clamp(raw_reply)

        if intent['action'] == 'find_and_stop':
            target = intent['target_class']
            self.watch_target = target
            self._detection_count = 0
            rospy.loginfo(f'find_and_stop: now driving and watching for {target!r} '
                         f'(confidence >= {DETECTION_CONFIDENCE_THRESHOLD})')
            self._publish_intent({'action': 'go'})
            rospy.Timer(rospy.Duration(NO_DETECTIONS_WARNING_DELAY),
                       self._check_detections_alive, oneshot=True)
            return

        self._publish_intent(intent)

    # ------------------------------------------------------------------
    def _check_detections_alive(self, _event):
        if self.watch_target is not None and self._detection_count == 0:
            rospy.logwarn('find_and_stop: no detections received yet — is '
                         'yolo_detector_node running?')

    # ------------------------------------------------------------------
    def detections_cb(self, msg: String):
        self._detection_count += 1
        if self.watch_target is None:
            return

        try:
            detections = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return

        for det in detections:
            if (det.get('label') == self.watch_target
                    and det.get('confidence', 0.0) >= DETECTION_CONFIDENCE_THRESHOLD):
                rospy.loginfo(f'find_and_stop: found {self.watch_target!r} '
                             f'(confidence={det["confidence"]:.2f}) — stopping')
                self.watch_target = None
                self._publish_intent({'action': 'stop'})
                return

    def spin(self):
        rospy.spin()


def main():
    node = NLCommandNode()
    node.spin()


if __name__ == '__main__':
    main()
