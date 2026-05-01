import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math

# ---------------- INIT ----------------
pygame.mixer.init()
pygame.init()

try:
    whoosh = pygame.mixer.Sound("hand_tracking/hands_assets/sound_track.mp3")
    channel = pygame.mixer.Channel(0)
except:
    whoosh = None
    channel = None

mp_hands = mp.solutions.hands
mp_seg = mp.solutions.selfie_segmentation

# ---------------- CHARACTER ----------------
class MagicCharacter:
    def __init__(self, path, side, depth):
        self.valid = False
        self.progress = 0
        self.cx = 0
        self.cy = 0
        self.time_offset = np.random.uniform(0, 2*np.pi)

        self.side = side
        self.depth = depth

        self.img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if self.img is None:
            print(f"❌ Failed to load: {path}")
            return

        self.valid = True

        if self.img.shape[2] == 3:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2BGRA)

        b, g, r, a = cv2.split(self.img)

        mask = (r < 240) | (g < 240) | (b < 240)
        a = np.where(mask, 255, 0).astype(np.uint8)

        kernel = np.ones((5,5), np.uint8)
        eroded = cv2.erode(a, kernel, iterations=2)
        blurred = cv2.GaussianBlur(a, (9, 9), 0)

        a = np.where(eroded == 255, 255, blurred)

        self.img = cv2.merge([b, g, r, a])

        base_h = 420
        aspect = self.img.shape[1] / self.img.shape[0]
        self.img = cv2.resize(self.img, (int(base_h * aspect), base_h))

    def update(self, dt):
        if self.progress < 1:
            self.progress += dt * 4
            t = 1 - (1 - self.progress) ** 3
            self.cx = t
            self.cy = t

    def draw(self, frame, center, global_time, spread_multiplier=1.6):
        h, w = frame.shape[:2]

        x_ratios = [0.11, 0.27, 0.36]
        y_ratios = [0.07, 0.17, 0.22]
        scales = [2.7, 2, 1.9]

        x_offset = int(w * x_ratios[self.depth] * spread_multiplier)
        y_offset = int(h * y_ratios[self.depth])
        scale = scales[self.depth]

        if self.side == "left":
            x_offset = -x_offset

        y_offset = -y_offset - (self.depth * int(h * 0.02))

        x_offset *= self.cx
        y_offset *= self.cy

        float_y = int(10 * math.sin(global_time * 2 + self.time_offset))
        y_offset += float_y

        img = self.img
        ih, iw = img.shape[:2]
        img = cv2.resize(img, (int(iw * scale), int(ih * scale)))

        x = int(center[0] + x_offset)
        y = int(center[1] + y_offset - int(h * 0.12))

        ih, iw = img.shape[:2]
        x -= iw // 2
        y -= ih // 2

        if x >= w or y >= h or x+iw <= 0 or y+ih <= 0:
            return

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x+iw)
        y2 = min(h, y+ih)

        ix1 = max(0, -x)
        iy1 = max(0, -y)
        ix2 = ix1 + (x2-x1)
        iy2 = iy1 + (y2-y1)

        alpha = img[iy1:iy2, ix1:ix2, 3] / 255.0

        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                alpha * img[iy1:iy2, ix1:ix2, c] +
                (1 - alpha) * frame[y1:y2, x1:x2, c]
            )

# ---------------- GESTURES ----------------
def is_plus(hands):
    if len(hands) < 2:
        return False
    p1 = hands[0].landmark[8]
    p2 = hands[1].landmark[8]
    return math.hypot(p1.x - p2.x, p1.y - p2.y) < 0.08

def is_thumb_up(hand):
    thumb_tip = hand.landmark[4]
    thumb_ip = hand.landmark[3]

    index_tip = hand.landmark[8]
    middle_tip = hand.landmark[12]
    ring_tip = hand.landmark[16]
    pinky_tip = hand.landmark[20]

    return (
        thumb_tip.y < thumb_ip.y and
        index_tip.y > thumb_ip.y and
        middle_tip.y > thumb_ip.y and
        ring_tip.y > thumb_ip.y and
        pinky_tip.y > thumb_ip.y
    )

# ---------------- MAIN ----------------
def main():
    paths = [
        "hand_tracking/hands_assets/naruto.png",
        "hand_tracking/hands_assets/madara.png",
        "hand_tracking/hands_assets/iron man.png",
        "hand_tracking/hands_assets/spider.png",
        "hand_tracking/hands_assets/captain_america.png",
        "hand_tracking/hands_assets/Thor.png",
    ]

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    cv2.namedWindow("Magic Summon", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Magic Summon", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    hands = mp_hands.Hands(max_num_hands=2)
    seg = mp_seg.SelfieSegmentation(1)

    chars = []
    active = False
    spread_multiplier = 1.0  # 👈 NEW

    last = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        seg_res = seg.process(rgb)
        mask = cv2.GaussianBlur(seg_res.segmentation_mask, (31, 31), 0)
        mask = np.clip(mask, 0, 1)

        ys, xs = np.where(mask > 0.5)
        bottom_y = np.max(ys) if len(ys) > 0 else int(h * 0.85)
        center = (w // 2, bottom_y)

        res = hands.process(rgb)
        hand_list = res.multi_hand_landmarks if res.multi_hand_landmarks else []

        plus_gesture = is_plus(hand_list)
        thumb_gesture = len(hand_list) == 1 and is_thumb_up(hand_list[0])

        if (plus_gesture or thumb_gesture) and not active:
            active = True
            chars.clear()

            if whoosh and channel:
                channel.play(whoosh)

            if thumb_gesture:
                selected = paths[:2]
                spread_multiplier = 1.6   # 👈 ONLY CHANGE HERE
            else:
                selected = paths
                spread_multiplier = 1.0

            for i in range(len(selected)):
                side = "right" if i % 2 == 0 else "left"
                depth = i // 2
                c = MagicCharacter(selected[i], side, depth)
                if c.valid:
                    chars.append(c)

        elif not (plus_gesture or thumb_gesture) and active:
            active = False
            chars.clear()
            spread_multiplier = 1.0

            if channel and channel.get_busy():
                channel.stop()

        dt = time.time() - last
        last = time.time()
        global_time = time.time()

        for c in chars:
            c.update(dt)

        layer = frame.copy()

        for c in sorted(chars, key=lambda x: x.depth, reverse=True):
            c.draw(layer, center, global_time, spread_multiplier)

        final = (mask[:, :, None] * frame + (1 - mask[:, :, None]) * layer).astype(np.uint8)

        cv2.imshow("Magic Summon", final)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()