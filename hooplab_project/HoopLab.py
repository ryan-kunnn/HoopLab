import argparse
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


# ---------------- VIDEO ----------------
parser = argparse.ArgumentParser(description="HoopLab")
parser.add_argument(
    "video",
    nargs="?",
    default="basketball_video1.mp4",
    help="Input video path. Relative paths are resolved from the HoopLab.py folder.",
)
parser.add_argument(
    "--no-display",
    action="store_true",
    help="Run analysis without opening the OpenCV preview window.",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=0,
    help="Optional frame limit for quick tests. Use 0 to analyze the full video.",
)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = args.video
if not os.path.isabs(video_path):
    video_path = os.path.join(script_dir, video_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Video not found: {video_path}")
    exit()

# ---------------- MEDIAPIPE POSE ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# ---------------- POSTURE TRAINING ANALYZER ----------------
class PostureTrainingAnalyzer:
    TRAINING_PLAN = {
        "low_release": "Release extension: 3 sets of 10 one-hand form shots, pausing with elbow high and arm fully extended.",
        "overextended_release": "Soft release control: 3 sets of 8 close-range shots with a relaxed elbow and smooth wrist snap.",
        "good_release": "Release consistency: 25 repeat-form shots from the same spot, keeping the same elbow finish each rep.",
        "low_shoulder": "Shoulder lift: 2 sets of 12 wall-form shots, starting with the shooting shoulder slightly higher.",
        "high_shoulder": "Relaxed shoulder path: 2 sets of 10 slow-motion shots, keeping the shoulder down and aligned.",
        "stable_shoulder": "Shoulder stability: 20 catch-and-hold reps, freeze at set point, then shoot.",
        "shallow_knee_bend": "Lower-body load: 3 sets of 8 squat-to-shot reps, bending knees before the ball rises.",
        "deep_knee_bend": "Compact dip: 3 sets of 8 quick-dip jump shots, stopping before the hips sink too low.",
        "good_knee_bend": "Leg rhythm: 20 rhythm shots using the same knee bend and upward timing.",
        "missing_follow_through": "Follow-through hold: 25 shots where the wrist stays down until the ball reaches the rim.",
        "solid_follow_through": "Finish repeatability: 20 shots with a two-second held follow-through.",
        "torso_leaning": "Torso control: 3 sets of 8 balance shots, landing where you jumped and keeping chest vertical.",
        "upright_torso": "Posture maintenance: 20 stationary shots with eyes level and chest stacked over hips.",
        "off_balance_base": "Base alignment: 3 sets of 10 foot-line shots, keeping hips centered between both feet.",
        "balanced_base": "Base consistency: 20 shots with the same foot width and landing position.",
    }

    def __init__(self):
        self.sessions = []
        self.current_session = None

    def classify_metric_outcomes(self, metrics):
        outcomes = []

        release_angle = metrics["release_angle"]
        shoulder_angle = metrics["shoulder_angle"]
        knee_bend = metrics["knee_bend"]
        torso_lean = metrics.get("torso_lean", 0)
        balance_shift = metrics.get("balance_shift", 0)

        if release_angle < 135:
            outcomes.append("low_release")
        elif release_angle > 165:
            outcomes.append("overextended_release")
        else:
            outcomes.append("good_release")

        if shoulder_angle < 75:
            outcomes.append("low_shoulder")
        elif shoulder_angle > 105:
            outcomes.append("high_shoulder")
        else:
            outcomes.append("stable_shoulder")

        if knee_bend < 30:
            outcomes.append("shallow_knee_bend")
        elif knee_bend > 45:
            outcomes.append("deep_knee_bend")
        else:
            outcomes.append("good_knee_bend")

        if metrics["follow_through"]:
            outcomes.append("solid_follow_through")
        else:
            outcomes.append("missing_follow_through")

        if torso_lean > 14:
            outcomes.append("torso_leaning")
        else:
            outcomes.append("upright_torso")

        if balance_shift > 0.22:
            outcomes.append("off_balance_base")
        else:
            outcomes.append("balanced_base")

        return outcomes

    def get_training_recommendations(self, metrics):
        return [self.TRAINING_PLAN[outcome] for outcome in self.classify_metric_outcomes(metrics)]

    def start_session(self):
        self.current_session = {
            "session_number": len(self.sessions) + 1,
            "postures": [],
            "training_plan": [],
        }

    def record_posture(self, frame_num, metrics):
        if not self.current_session:
            self.start_session()

        posture = {
            "frame": frame_num,
            "score": metrics["posture_score"],
            "release_angle": metrics["release_angle"],
            "shoulder_angle": metrics["shoulder_angle"],
            "knee_bend": metrics["knee_bend"],
            "hip_angle": metrics.get("hip_angle", 0),
            "torso_lean": metrics.get("torso_lean", 0),
            "balance_shift": metrics.get("balance_shift", 0),
            "follow_through": metrics["follow_through"],
            "outcomes": self.classify_metric_outcomes(metrics),
            "recommendations": self.get_training_recommendations(metrics),
        }
        self.current_session["postures"].append(posture)
        return posture

    def end_session(self):
        if self.current_session and self.current_session["postures"]:
            self.current_session["training_plan"] = self.build_video_training_plan(
                self.current_session["postures"]
            )
            self.sessions.append(self.current_session)
            self.current_session = None

    def build_video_training_plan(self, postures):
        outcome_counts = {}

        for posture in postures:
            for outcome in posture["outcomes"]:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        issue_outcomes = {
            key: count
            for key, count in outcome_counts.items()
            if not key.startswith(("good_", "solid_", "stable_", "upright_", "balanced_"))
        }
        positive_outcomes = {
            key: count for key, count in outcome_counts.items()
            if key not in issue_outcomes
        }
        ranked_outcomes = (
            sorted(issue_outcomes.items(), key=lambda x: x[1], reverse=True)
            + sorted(positive_outcomes.items(), key=lambda x: x[1], reverse=True)
        )

        plan = []
        for outcome, _ in ranked_outcomes:
            rec = self.TRAINING_PLAN[outcome]
            if rec not in plan:
                plan.append(rec)

        return plan

    def get_training_plan(self):
        if self.current_session and self.current_session["postures"]:
            return self.build_video_training_plan(self.current_session["postures"])

        if not self.sessions:
            return ["No training data available. Start practicing to get personalized recommendations."]

        return self.sessions[-1].get("training_plan", ["Continue practicing to refine your technique."])

    def print_summary(self):
        print("\n=== INPUT VIDEO POSTURE ANALYSIS ===")

        if not self.sessions:
            print("No posture data recorded.")
            print("Recommendation: Practice shooting motions in front of the camera to analyze your form.")
            print("====================================")
            return

        postures = self.sessions[-1]["postures"]
        print(f"Video: {video_path}")
        print(f"Detected shots/postures: {len(postures)}")

        outcome_counts = {}
        for posture in postures:
            for outcome in posture["outcomes"]:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        print("\nPosture outcomes in this video:")
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {outcome.replace('_', ' ')}: {count}")

        print("\nPosture details:")
        for i, posture in enumerate(postures, 1):
            print(f"   Arm extension: {posture['release_angle']:.1f} deg")
            print(f"   Shoulder position: {posture['shoulder_angle']:.1f} deg")
            print(f"   Knee bend: {posture['knee_bend']:.1f} deg")
            print(f"   Body lean: {posture['torso_lean']:.1f} deg")
            print(f"   Balance shift: {posture['balance_shift']:.2f}")

        print("\nTraining Plan:")
        for i, rec in enumerate(self.get_training_plan(), 1):
            print(f"{i}. {rec}")

        print("====================================")


# ---------------- SHOOTING POSTURE ANALYZER ----------------
class ShootingPostureAnalyzer:
    RELEASE_DISPLAY_FRAMES = 9

    def __init__(self):
        self.angle_history = {
            "elbow": deque(maxlen=5),
            "shoulder": deque(maxlen=5),
            "knee": deque(maxlen=5),
        }
        self.phase = "READY"
        self.display_phase = "READY"
        self.release_display_until_frame = 0
        self.phase_start_frame = 0
        self.shooting_arm = None
        self.balance_shift_peak = 0
        self.best_candidate = None
        self.best_candidate_frame = 0
        self.metrics = self.empty_metrics()
        self.feedback = []

    def empty_metrics(self):
        return {
            "release_angle": 0,
            "shoulder_angle": 0,
            "knee_bend": 0,
            "hip_angle": 0,
            "torso_lean": 0,
            "balance_shift": 0,
            "follow_through": False,
            "posture_score": 0,
        }

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denominator == 0:
            return 0
        cosine = np.dot(ba, bc) / denominator
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def get_smoothed_angle(self, angle_name, new_angle):
        self.angle_history[angle_name].append(new_angle)
        return np.mean(self.angle_history[angle_name])

    def get_landmark_coords(self, landmarks, width, height, landmark_id):
        lm = landmarks[landmark_id]
        return (int(lm.x * width), int(lm.y * height))

    def get_landmark_point(self, landmarks, width, height, landmark_id):
        lm = landmarks[landmark_id]
        return (lm.x * width, lm.y * height, lm.z * width)

    def get_visibility(self, landmarks, landmark_id):
        return landmarks[landmark_id].visibility

    def arm_landmarks(self, side):
        if side == "right":
            return {
                "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
                "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
                "wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
                "hip": mp_pose.PoseLandmark.RIGHT_HIP,
                "knee": mp_pose.PoseLandmark.RIGHT_KNEE,
                "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
                "opposite_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            }

        return {
            "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
            "hip": mp_pose.PoseLandmark.LEFT_HIP,
            "knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "opposite_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        }

    def update_display_phase(self, frame_num):
        if self.phase == "RELEASE":
            self.display_phase = "RELEASE"
            self.release_display_until_frame = max(
                self.release_display_until_frame,
                self.phase_start_frame + self.RELEASE_DISPLAY_FRAMES,
            )
        elif frame_num <= self.release_display_until_frame:
            self.display_phase = "RELEASE"
        else:
            self.display_phase = self.phase

    def score_arm_activity(self, landmarks, width, height, side):
        ids = self.arm_landmarks(side)
        shoulder = ids["shoulder"]
        elbow = ids["elbow"]
        wrist = ids["wrist"]

        visibility = (
            self.get_visibility(landmarks, shoulder)
            + self.get_visibility(landmarks, elbow)
            + self.get_visibility(landmarks, wrist)
        ) / 3

        shoulder_coords = self.get_landmark_coords(landmarks, width, height, shoulder)
        wrist_coords = self.get_landmark_coords(landmarks, width, height, wrist)
        shoulder_point = self.get_landmark_point(landmarks, width, height, shoulder)
        elbow_point = self.get_landmark_point(landmarks, width, height, elbow)
        wrist_point = self.get_landmark_point(landmarks, width, height, wrist)
        elbow_angle = self.calculate_angle(shoulder_point, elbow_point, wrist_point)

        wrist_above_shoulder = max(0, shoulder_coords[1] - wrist_coords[1])
        wrist_near_shoulder = max(0, 70 - abs(wrist_coords[1] - shoulder_coords[1]))
        shooting_shape = max(0, 170 - abs(145 - elbow_angle))

        return (
            visibility * 120
            + wrist_above_shoulder * 0.8
            + wrist_near_shoulder * 0.35
            + shooting_shape * 0.4
        )

    def detect_shooting_arm(self, landmarks, width, height):
        right_score = self.score_arm_activity(landmarks, width, height, "right")
        left_score = self.score_arm_activity(landmarks, width, height, "left")

        if self.shooting_arm == "right":
            right_score += 25
        elif self.shooting_arm == "left":
            left_score += 25

        return "right" if right_score >= left_score else "left"

    def analyze_posture(self, landmarks, width, height, frame_num):
        if not landmarks:
            return None, None

        detected_arm = self.detect_shooting_arm(landmarks, width, height)
        if self.shooting_arm != detected_arm and self.phase == "READY":
            for key in self.angle_history:
                self.angle_history[key].clear()
        self.shooting_arm = detected_arm

        selected = self.arm_landmarks(self.shooting_arm)
        shoulder = selected["shoulder"]
        elbow = selected["elbow"]
        wrist = selected["wrist"]
        hip = selected["hip"]
        knee = selected["knee"]
        ankle = selected["ankle"]
        opposite_shoulder = selected["opposite_shoulder"]

        shoulder_coords = self.get_landmark_coords(landmarks, width, height, shoulder)
        elbow_coords = self.get_landmark_coords(landmarks, width, height, elbow)
        wrist_coords = self.get_landmark_coords(landmarks, width, height, wrist)
        hip_coords = self.get_landmark_coords(landmarks, width, height, hip)
        knee_coords = self.get_landmark_coords(landmarks, width, height, knee)
        ankle_coords = self.get_landmark_coords(landmarks, width, height, ankle)

        shoulder_point = self.get_landmark_point(landmarks, width, height, shoulder)
        elbow_point = self.get_landmark_point(landmarks, width, height, elbow)
        wrist_point = self.get_landmark_point(landmarks, width, height, wrist)
        hip_point = self.get_landmark_point(landmarks, width, height, hip)
        knee_point = self.get_landmark_point(landmarks, width, height, knee)
        ankle_point = self.get_landmark_point(landmarks, width, height, ankle)
        opposite_shoulder_point = self.get_landmark_point(landmarks, width, height, opposite_shoulder)

        elbow_angle = self.calculate_angle(shoulder_point, elbow_point, wrist_point)
        shoulder_angle = self.calculate_angle(hip_point, shoulder_point, elbow_point)
        knee_angle = self.calculate_angle(hip_point, knee_point, ankle_point)
        hip_angle = self.calculate_angle(shoulder_point, hip_point, knee_point)
        torso_angle = self.calculate_angle(hip_point, shoulder_point, opposite_shoulder_point)
        torso_lean = abs(
            np.degrees(
                np.arctan2(
                    shoulder_coords[0] - hip_coords[0],
                    max(1, hip_coords[1] - shoulder_coords[1]),
                )
            )
        )

        left_ankle = self.get_landmark_coords(landmarks, width, height, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = self.get_landmark_coords(landmarks, width, height, mp_pose.PoseLandmark.RIGHT_ANKLE)
        left_hip = self.get_landmark_coords(landmarks, width, height, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = self.get_landmark_coords(landmarks, width, height, mp_pose.PoseLandmark.RIGHT_HIP)
        foot_center_x = (left_ankle[0] + right_ankle[0]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        body_scale = max(1, abs(hip_coords[1] - shoulder_coords[1]))
        balance_shift = abs(hip_center_x - foot_center_x) / body_scale
        self.balance_shift_peak = max(self.balance_shift_peak, balance_shift)

        elbow_angle_smooth = self.get_smoothed_angle("elbow", elbow_angle)
        shoulder_angle_smooth = self.get_smoothed_angle("shoulder", shoulder_angle)
        knee_angle_smooth = self.get_smoothed_angle("knee", knee_angle)
        current_metrics = {
            "release_angle": elbow_angle_smooth,
            "shoulder_angle": shoulder_angle_smooth,
            "knee_bend": 180 - knee_angle_smooth,
            "hip_angle": hip_angle,
            "torso_lean": torso_lean,
            "balance_shift": self.balance_shift_peak,
            "follow_through": wrist_coords[1] < shoulder_coords[1] - 15,
            "posture_score": 0,
        }
        current_metrics["posture_score"] = self.score_metrics(current_metrics)

        candidate_pose = (
            wrist_coords[1] < shoulder_coords[1] + 40
            or elbow_angle_smooth > 125
            or self.phase in ["LIFT", "RELEASE", "FOLLOW_THROUGH"]
        )
        if candidate_pose and (
            self.best_candidate is None
            or current_metrics["release_angle"] > self.best_candidate["release_angle"]
        ):
            self.best_candidate = current_metrics.copy()
            self.best_candidate_frame = frame_num

        if self.phase == "READY":
            if elbow_angle_smooth < 120:
                self.phase = "LIFT"
                self.phase_start_frame = frame_num
        elif self.phase == "LIFT":
            if elbow_angle_smooth > 140:
                self.phase = "RELEASE"
                self.phase_start_frame = frame_num
                self.metrics["release_angle"] = elbow_angle_smooth
                self.metrics["shoulder_angle"] = shoulder_angle_smooth
                self.metrics["knee_bend"] = 180 - knee_angle_smooth
                self.metrics["hip_angle"] = hip_angle
                self.metrics["torso_lean"] = torso_lean
                self.metrics["balance_shift"] = self.balance_shift_peak
        elif self.phase == "RELEASE":
            release_frames = frame_num - self.phase_start_frame
            wrist_above_shoulder = wrist_coords[1] < shoulder_coords[1] - 15
            arm_started_relaxing = elbow_angle_smooth < 150

            if wrist_above_shoulder or (release_frames >= 1 and arm_started_relaxing):
                self.phase = "FOLLOW_THROUGH"
                self.phase_start_frame = frame_num
        elif self.phase == "FOLLOW_THROUGH":
            if wrist_coords[1] < shoulder_coords[1] - 15:
                self.metrics["follow_through"] = True

        self.update_display_phase(frame_num)

        self.feedback = []
        if self.phase == "RELEASE":
            if 140 <= elbow_angle_smooth <= 160:
                self.feedback.append(("OK Elbow angle ideal", (0, 255, 0)))
            elif elbow_angle_smooth < 140:
                self.feedback.append(("WARN Elbow too bent - extend more", (0, 165, 255)))
            else:
                self.feedback.append(("WARN Elbow overextended", (0, 165, 255)))

        if 80 <= shoulder_angle_smooth <= 100:
            self.feedback.append(("OK Good shoulder alignment", (0, 255, 0)))
        elif shoulder_angle_smooth < 80:
            self.feedback.append(("WARN Shoulder too low", (0, 165, 255)))
        else:
            self.feedback.append(("WARN Shoulder too high", (0, 165, 255)))

        if self.phase in ["LIFT", "RELEASE"]:
            knee_bend = 180 - knee_angle_smooth
            if 30 <= knee_bend <= 45:
                self.feedback.append(("OK Proper knee bend", (0, 255, 0)))
            elif knee_bend < 30:
                self.feedback.append(("WARN Bend knees more", (0, 165, 255)))
            else:
                self.feedback.append(("WARN Knees too bent", (0, 165, 255)))

        if torso_lean < 14:
            self.feedback.append(("OK Good torso alignment", (0, 255, 0)))
        else:
            self.feedback.append(("WARN Keep torso straight", (0, 165, 255)))

        keypoints = {
            "shoulder": shoulder_coords,
            "elbow": elbow_coords,
            "wrist": wrist_coords,
            "hip": hip_coords,
            "knee": knee_coords,
            "ankle": ankle_coords,
        }

        return keypoints, {
            "elbow": elbow_angle_smooth,
            "shoulder": shoulder_angle_smooth,
            "knee": knee_angle_smooth,
            "hip": hip_angle,
            "torso": torso_angle,
            "torso_lean": torso_lean,
            "balance_shift": balance_shift,
        }

    def score_metrics(self, metrics):
        score = 0

        if 145 <= metrics["release_angle"] <= 155:
            score += 30
        elif 135 <= metrics["release_angle"] <= 165:
            score += 20
        else:
            score += 10

        if 85 <= metrics["shoulder_angle"] <= 95:
            score += 25
        elif 75 <= metrics["shoulder_angle"] <= 105:
            score += 15
        else:
            score += 5

        if 35 <= metrics["knee_bend"] <= 40:
            score += 25
        elif 30 <= metrics["knee_bend"] <= 45:
            score += 15
        else:
            score += 5

        if metrics["follow_through"]:
            score += 20

        return score

    def calculate_score(self):
        score = self.score_metrics(self.metrics)
        self.metrics["posture_score"] = score
        return score

    def reset_for_next_shot(self):
        self.phase = "READY"
        self.display_phase = "READY"
        self.release_display_until_frame = 0
        self.shooting_arm = None
        self.metrics = self.empty_metrics()
        self.balance_shift_peak = 0
        self.best_candidate = None
        self.best_candidate_frame = 0
        for key in self.angle_history:
            self.angle_history[key].clear()


# ---------------- DRAWING UTILITIES ----------------
def draw_skeleton(frame, landmarks, connections, color=(0, 0, 255), thickness=2):
    h, w = frame.shape[:2]

    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        start = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))

        if 0 <= start[0] < w and 0 <= start[1] < h and 0 <= end[0] < w and 0 <= end[1] < h:
            cv2.line(frame, start, end, color, thickness)

    important = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
    ]

    for idx, lm in enumerate(landmarks):
        x, y = int(lm.x * w), int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            if idx in important:
                cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 6, (0, 0, 255), 1)
            else:
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)


def draw_panel(frame, x, y, width, height, alpha=0.68):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 22, 28), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 80, 90), 1)


def draw_label(frame, text, x, y, color=(235, 235, 235), scale=0.5, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_status_panel(frame, posture_analyzer, training_analyzer, last_recorded_posture):
    h, w = frame.shape[:2]
    panel_w, panel_h = 410, 135
    x = max(10, w - panel_w - 16)
    y = 14

    draw_panel(frame, x, y, panel_w, panel_h, alpha=0.88)

    display_phase = posture_analyzer.display_phase
    phase_color = {
        "READY": (255, 255, 140),
        "LIFT": (255, 180, 110),
        "RELEASE": (140, 255, 140),
        "FOLLOW_THROUGH": (180, 255, 255),
    }.get(display_phase, (255, 255, 255))

    draw_label(frame, f"Phase: {display_phase}", x + 20, y + 34, phase_color, 0.62, 2)

    recorded_count = 0
    if training_analyzer.current_session:
        recorded_count = len(training_analyzer.current_session["postures"])
    elif training_analyzer.sessions:
        recorded_count = len(training_analyzer.sessions[-1]["postures"])
    draw_label(frame, f"Video postures: {recorded_count}", x + 20, y + 66, (180, 245, 180), 0.52, 2)

    if last_recorded_posture:
        outcome = ", ".join(last_recorded_posture["outcomes"][:2]).replace("_", " ")
        draw_label(frame, f"Latest: {outcome}", x + 20, y + 96, (200, 220, 255), 0.48, 1)
        draw_label(
            frame,
            f"Angles E/S/K: {last_recorded_posture['release_angle']:.0f}/"
            f"{last_recorded_posture['shoulder_angle']:.0f}/"
            f"{last_recorded_posture['knee_bend']:.0f}",
            x + 20,
            y + 122,
            (200, 220, 255),
            0.48,
            1,
        )
    else:
        draw_label(frame, "Latest: waiting for release", x + 20, y + 96, (200, 220, 255), 0.48, 1)


# ---------------- MAIN LOOP ----------------
posture_analyzer = ShootingPostureAnalyzer()
training_analyzer = PostureTrainingAnalyzer()
frame_num = 0
last_posture_frame = 0
posture_recorded = False
last_recorded_posture = None

POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
]

print(f"\nStarting posture analysis for {video_path}")
print("Press 'r' reset | 'q' quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    if args.max_frames and frame_num > args.max_frames:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)

    if pose_results.pose_landmarks:
        posture_analyzer.analyze_posture(pose_results.pose_landmarks.landmark, w, h, frame_num)
        draw_skeleton(frame, pose_results.pose_landmarks.landmark, POSE_CONNECTIONS, (0, 0, 255), 2)
    posture_analyzer.update_display_phase(frame_num)

    draw_status_panel(frame, posture_analyzer, training_analyzer, last_recorded_posture)

    if (
        posture_analyzer.phase == "FOLLOW_THROUGH"
        and frame_num - posture_analyzer.phase_start_frame > 2
        and frame_num - last_posture_frame > 30
    ):
        posture_analyzer.calculate_score()

        if not posture_recorded:
            last_recorded_posture = training_analyzer.record_posture(
                frame_num,
                posture_analyzer.metrics.copy(),
            )
            posture_recorded = True
            print(
                f"Posture recorded at frame {frame_num}: "
                f"{', '.join(last_recorded_posture['outcomes']).replace('_', ' ')}"
            )

        if frame_num - last_posture_frame > 90:
            posture_analyzer.reset_for_next_shot()
            last_posture_frame = frame_num
            posture_recorded = False

    cv2.putText(
        frame,
        "Press 'r' reset | 'q' quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    max_width, max_height = 960, 540
    scale = min(max_width / w, max_height / h, 1)
    if not args.no_display:
        display = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.imshow("HoopLab", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            posture_analyzer.reset_for_next_shot()
            training_analyzer.end_session()
            training_analyzer.start_session()
            last_posture_frame = frame_num
            posture_recorded = False
            last_recorded_posture = None

if not posture_recorded and posture_analyzer.best_candidate:
    last_recorded_posture = training_analyzer.record_posture(
        posture_analyzer.best_candidate_frame,
        posture_analyzer.best_candidate.copy(),
    )
    print(
        f"Fallback posture recorded at frame {posture_analyzer.best_candidate_frame}: "
        f"{', '.join(last_recorded_posture['outcomes']).replace('_', ' ')}"
    )

cap.release()
if not args.no_display:
    cv2.destroyAllWindows()

training_analyzer.end_session()
training_analyzer.print_summary()
