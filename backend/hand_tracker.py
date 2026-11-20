import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.results = None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        detected_hands = []

        if self.results.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                landmarks = [
                    {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                    for landmark in hand_landmarks.landmark
                ]
                handedness = "Unknown"
                if self.results.multi_handedness:
                    handedness = self.results.multi_handedness[index].classification[0].label
                detected_hands.append({
                    "hand": handedness,
                    "landmarks": landmarks,
                    "landmark_count": len(landmarks)
                })
        return frame, detected_hands

    def close(self):
        self.hands.close()


def main():
    tracker = HandTracker()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not access webcam.")
        tracker.close()
        return

    print("SIGNBRIDGE HAND TRACKER ONLINE")
    print("Show your hand to the camera.")
    print("Press Q to exit.")
    previous_time = time.time()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("ERROR: Could not read webcam frame.")
                break
            frame = cv2.flip(frame, 1)
            frame, detected_hands = tracker.process_frame(frame)
            current_time = time.time()
            fps = 1 / max(current_time - previous_time, 0.001)
            previous_time = current_time
            cv2.putText(frame, f"Hands detected: {len(detected_hands)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for index, hand in enumerate(detected_hands):
                print(f"\rHand {index + 1}: {hand['hand']} | "
                      f"Landmarks: {hand['landmark_count']}", end=" " * 10)
            cv2.imshow("SIGNBRIDGE - Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("\nSIGNBRIDGE HAND TRACKER STOPPED")


if __name__ == "__main__":
    main()
