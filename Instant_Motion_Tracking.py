import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode = False, max_num_hands = 2) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks( frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow('Instant Motion Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





