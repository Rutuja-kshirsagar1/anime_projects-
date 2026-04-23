import cv2
import numpy as np
import mediapipe as mp
import pygame
import os

# INIT AUDIO

pygame.mixer.init()
audio_playing = False


# MEDIA MAP (SHAPE → VIDEO + AUDIO)

media_map = {
    "circle": {
        "video": "ShapeStream/shape_assets/tsukuyomi_s.mp4",
        "audio": "shape_assets/tsukuyomi.mp3"
    },
    "rectangle": {
        "video": "shape_assets/madara_uchiha.mp4",
        "audio": "shape_assets/madara_uchiha.mp3"
    },
    "triangle": {
        "video": "shape_assets/jarvis.mp4",
        "audio": "shape_assets/jarvis audio.mp3"
    }
}

current_video = None
current_shape = None

# MediaPipe

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Setup

cap = cv2.VideoCapture(0)
canvas = None
draw_points = []
shape_points = []

prev_point = None
smooth_prev = None

ALPHA = 0.4
MIN_DIST = 5

# Finger detection

def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    fingers.append(hand_landmarks.landmark[4].x <
                   hand_landmarks.landmark[3].x)

    for tip in tips:
        fingers.append(hand_landmarks.landmark[tip].y <
                       hand_landmarks.landmark[tip - 2].y)

    return fingers

# Smooth

def smooth(pt):
    global smooth_prev
    if smooth_prev is None:
        smooth_prev = pt
        return pt

    x = int(ALPHA * pt[0] + (1 - ALPHA) * smooth_prev[0])
    y = int(ALPHA * pt[1] + (1 - ALPHA) * smooth_prev[1])

    smooth_prev = (x, y)
    return (x, y)


# Glow draw

def draw_glow(canvas, p1, p2):
    if p1 is None or p2 is None:
        return

    temp = np.zeros_like(canvas)
    color = (255, 180, 80)

    cv2.line(temp, p1, p2, color, 2)
    glow = cv2.GaussianBlur(temp, (21, 21), 0)

    cv2.addWeighted(glow, 0.6, canvas, 1.0, 0, canvas)


# Shape Detection

def detect_shape(points):
    pts = np.array(points, dtype=np.int32)
    contour = pts.reshape((-1, 1, 2))

    area = cv2.contourArea(contour)
    if area < 800:
        return None

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "rectangle"
    else:
        circularity = 4 * np.pi / (peri * peri / (area + 1e-5))
        if 0.75 < circularity <= 1.2:
            return "circle"

    return None


# Draw Shape + Mask

def draw_shape_and_mask(shape, points, canvas, mask):
    pts = np.array(points, dtype=np.int32)

    if shape == "circle":
        (x, y), r = cv2.minEnclosingCircle(pts)
        center = (int(x), int(y))
        radius = int(r)

        cv2.circle(canvas, center, radius, (190, 127, 247), 3)
        cv2.circle(mask, center, radius, 255, -1)

    elif shape == "rectangle":
        x, y, w, h = cv2.boundingRect(pts)

        cv2.rectangle(canvas, (x, y), (x + w, y + h), (194, 169, 115), 3)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    elif shape == "triangle":
        contour = pts.reshape((-1, 1, 2))
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        cv2.drawContours(canvas, [approx], -1, (120, 50, 200), 3)
        cv2.drawContours(mask, [approx], -1, 255, -1)

# Main loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        h, w, _ = frame.shape
        x = int(hand_landmarks.landmark[8].x * w)
        y = int(hand_landmarks.landmark[8].y * h)

        finger_state = fingers_up(hand_landmarks)

        # ✊ Clear
        if sum(finger_state) == 0:
            canvas = np.zeros_like(frame)
            draw_points = []
            current_shape = None

        # ☝️ Draw
        elif finger_state[1] and sum(finger_state) == 1:
            smoothed = smooth((x, y))

            if prev_point is None:
                prev_point = smoothed

            draw_glow(canvas, prev_point, smoothed)
            draw_points.append(smoothed)
            prev_point = smoothed

        # ✋ Stop → Detect shape
        else:
            if len(draw_points) > 20:
                detected = detect_shape(draw_points)

                if detected in media_map:
                    current_shape = detected
                    shape_points = draw_points.copy()

                    # Load video
                    if current_video is not None:
                        current_video.release()

                    video_path = media_map[current_shape]["video"]
                    if os.path.exists(video_path):
                        current_video = cv2.VideoCapture(video_path)

                    # Load audio
                    audio_path = media_map[current_shape]["audio"]
                    if os.path.exists(audio_path):
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load(audio_path)

                    canvas = np.zeros_like(frame)

            draw_points = []
            prev_point = None
            smooth_prev = None

        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

    # Draw shape + mask
    if current_shape:
        draw_shape_and_mask(current_shape, shape_points, canvas, mask)

    # Gesture to play video
    show_video = False
    if result.multi_hand_landmarks:
        fingers = fingers_up(result.multi_hand_landmarks[0])
        if fingers[0] and fingers[1] and fingers[2]:
            show_video = True

    # Video playback
    if show_video and current_video is not None and np.sum(mask) > 0:
         if not audio_playing:
             pygame.mixer.music.play(-1)
             audio_playing = True
         ret_vid, vid_frame = current_video.read()
         if not ret_vid:
             current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
             ret_vid, vid_frame = current_video.read()
         #  FIT VIDEO TO SHAPE
         pts = np.array(shape_points, dtype=np.int32)
         x, y, w, h = cv2.boundingRect(pts)
         resized_video = cv2.resize(vid_frame, (w, h))
         video_canvas = np.zeros_like(frame)
         video_canvas[y:y+h, x:x+w] = resized_video
         masked_video = cv2.bitwise_and(video_canvas, video_canvas, mask=mask)
         inv_mask = cv2.bitwise_not(mask)
         background = cv2.bitwise_and(frame, frame, mask=inv_mask)
         frame = cv2.add(background, masked_video)
    else:
        if audio_playing:
            pygame.mixer.music.stop()
            audio_playing = False
    

    # Overlay
    frame = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

    cv2.imshow("ShapeStream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if current_video:
    current_video.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
