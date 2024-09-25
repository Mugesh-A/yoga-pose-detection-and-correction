import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

def inFrame(lst):
    # Checks if key body parts are visible in the frame
    return all(part.visibility > 0.6 for part in [lst[28], lst[27], lst[15], lst[16]])

# Load the trained model and labels
model = load_model("model.h5")
labels = np.load("labels.npy")

# Load reference images for each posture (dictionary mapping labels to images)
reference_images = {
    'Tadasana': cv2.imread('tadasana.png'),
    'Vrikshasana': cv2.imread('vrikshasana.png'),
    # Add paths to reference images for other poses here
}

# Display available poses and let the user select one
print("Available Poses:")
for i, pose in enumerate(labels):
    print(f"{i + 1}. {pose}")

# User selects a pose
selected_pose_index = int(input("Select the pose by number: ")) - 1
selected_pose = labels[selected_pose_index]

# Initialize Mediapipe Pose and Drawing utilities
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    lst = []

    # Capture frame-by-frame
    _, frm = cap.read()
    window = np.zeros((940, 1200, 3), dtype="uint8")  # Adjusted window size
    frm = cv2.flip(frm, 1)  # Flip the frame horizontally
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    frm = cv2.blur(frm, (4, 4))  # Apply blur to smooth the frame

    # Check if the pose landmarks are detected and the user is in frame
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        lst = np.array(lst).reshape(1, -1)
        p = model.predict(lst)
        pred = labels[np.argmax(p)]

        # Check if the user is doing the selected pose
        if pred == selected_pose and p[0][np.argmax(p)] > 0.75:
            cv2.putText(window, f"Good Job! Pose: {pred}", (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
        else:
            # Message for incorrect or untrained pose
            cv2.putText(window, "Incorrect posture detected! Please adjust.", (50, 180), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            
            # Display correct posture reference image
            correct_pose_img = reference_images.get(selected_pose, None)
            if correct_pose_img is not None:
                correct_pose_img = cv2.resize(correct_pose_img, (200, 400))  # Resize to fit window
                window[20:420, 20:220] = correct_pose_img  # Position the image on the window

    else:
        # Message if the body is not fully visible
        cv2.putText(frm, "Make Sure Full body is visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Draw pose landmarks
    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                           connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

    # Display the frame
    window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))
    cv2.imshow("window", window)

    # Exit the loop if 'Esc' is pressed
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
