import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    if not success:
        break

    # Flip camera
    image = cv2.flip(image, 1)

    # Convert color
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect body
    results = pose.process(image_rgb)

    # Draw body landmarks
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Body Tracking", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
