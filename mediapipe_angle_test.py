import cv2
import math

try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose
except (ImportError, AttributeError):
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.pose as mp_pose


CAMERA_FRAME_WIDTH = 1000
CAMERA_FRAME_HEIGHT = 1000
WINDOW_NAME = "MediaPipe Angle Test"

# Same side as main_integrated.py. Change both files together if the task arm changes.
SELECTED_SIDE = "right"
VISIBILITY_THRESHOLD = 0.6
FLIP_FRAME = False

LANDMARK_INDEX = {
    "left": {
        "hip": 23,
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
    },
    "right": {
        "hip": 24,
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
    },
}


def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cosine_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine_angle))


def landmark_to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def put_text(frame, text, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_selected_arm(frame, hip_pt, shoulder_pt, elbow_pt, wrist_pt):
    cv2.line(frame, hip_pt, shoulder_pt, (255, 180, 0), 3)
    cv2.line(frame, shoulder_pt, elbow_pt, (0, 255, 255), 3)
    cv2.line(frame, elbow_pt, wrist_pt, (0, 255, 0), 3)

    for label, point, color in (
        ("HIP", hip_pt, (255, 180, 0)),
        ("SH", shoulder_pt, (0, 255, 255)),
        ("EL", elbow_pt, (0, 255, 0)),
        ("WR", wrist_pt, (0, 180, 255)),
    ):
        cv2.circle(frame, point, 7, color, -1)
        cv2.putText(
            frame,
            label,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def main():
    if SELECTED_SIDE not in LANDMARK_INDEX:
        raise ValueError('SELECTED_SIDE must be "left" or "right".')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    selected = LANDMARK_INDEX[SELECTED_SIDE]

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            put_text(frame, f"Side: {SELECTED_SIDE}", 30, (0, 255, 255))
            put_text(
                frame,
                "Landmarks: "
                f"hip={selected['hip']} shoulder={selected['shoulder']} "
                f"elbow={selected['elbow']} wrist={selected['wrist']}",
                60,
                (0, 255, 255),
            )
            put_text(frame, "ESC or Q: quit", 90, (180, 180, 180))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                lm = results.pose_landmarks.landmark
                hip = lm[selected["hip"]]
                shoulder = lm[selected["shoulder"]]
                elbow = lm[selected["elbow"]]
                wrist = lm[selected["wrist"]]

                visibility = {
                    "hip": hip.visibility,
                    "shoulder": shoulder.visibility,
                    "elbow": elbow.visibility,
                    "wrist": wrist.visibility,
                }
                visibility_ok = all(
                    value >= VISIBILITY_THRESHOLD for value in visibility.values()
                )

                height, width, _ = frame.shape
                hip_pt = landmark_to_pixel(hip, width, height)
                shoulder_pt = landmark_to_pixel(shoulder, width, height)
                elbow_pt = landmark_to_pixel(elbow, width, height)
                wrist_pt = landmark_to_pixel(wrist, width, height)

                draw_selected_arm(frame, hip_pt, shoulder_pt, elbow_pt, wrist_pt)

                shoulder_angle = calculate_angle(hip_pt, shoulder_pt, elbow_pt)
                elbow_angle = calculate_angle(shoulder_pt, elbow_pt, wrist_pt)

                status_color = (0, 255, 0) if visibility_ok else (0, 0, 255)
                put_text(
                    frame,
                    "Visibility: "
                    f"H={visibility['hip']:.2f} S={visibility['shoulder']:.2f} "
                    f"E={visibility['elbow']:.2f} W={visibility['wrist']:.2f}",
                    130,
                    status_color,
                )
                put_text(
                    frame,
                    f"Shoulder angle hip-shoulder-elbow: {shoulder_angle:.1f} deg",
                    165,
                    status_color,
                )
                put_text(
                    frame,
                    f"Elbow angle shoulder-elbow-wrist: {elbow_angle:.1f} deg",
                    200,
                    status_color,
                )

                if not visibility_ok:
                    put_text(
                        frame,
                        f"LOW VISIBILITY: threshold={VISIBILITY_THRESHOLD:.2f}",
                        235,
                        (0, 0, 255),
                    )
            else:
                put_text(frame, "No pose landmarks detected", 130, (0, 0, 255))

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
