import cv2
import mediapipe as mp
# Video Input (Sequence of imgs) Local Storage
# Folder OS Module

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose # Algorithm 
pose = mp_pose.Pose()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

mp_draw = mp.solutions.drawing_utils

# for loop img length
while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1) # Add Resolution Input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process body
    pose_results = pose.process(image_rgb)

    # Process hands
    hand_results = hands.process(image_rgb)

    # Draw body
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Draw hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Pose + Hand Tracking", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
