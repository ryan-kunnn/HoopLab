import cv2
import mediapipe as mp
import math

# Function to calculate angle
def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) -
        math.atan2(ay - by, ax - bx)
    )

    if angle < 0:
        angle += 360

    return angle


cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get right arm points
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]

        h, w, _ = image.shape

        shoulder_coords = (int(shoulder.x * w), int(shoulder.y * h))
        elbow_coords = (int(elbow.x * w), int(elbow.y * h))
        wrist_coords = (int(wrist.x * w), int(wrist.y * h))

        # Calculate elbow angle
        angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)

        # Show angle
        cv2.putText(image, str(int(angle)),
                    elbow_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        # Basic shooting feedback
        if 80 <= angle <= 110:
            feedback = "Good Shooting Form"
        elif angle < 80:
            feedback = "Arm Too Bent"
        else:
            feedback = "Arm Too Straight"

        cv2.putText(image, feedback,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        mp_draw.draw_landmarks(image,
                               results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    cv2.imshow("HoopLab Shooting Analyzer", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
