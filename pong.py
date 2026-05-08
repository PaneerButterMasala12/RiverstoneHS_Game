"""
Pong — OpenCV hand-controlled paddles (pure OpenCV, no MediaPipe)
-----------------------------------------------------------------
Uses HSV skin-colour detection to find your hands in the webcam feed.

  Left hand on the LEFT side of the camera  → left paddle  (green)
  Right hand on the RIGHT side of the camera → right paddle (red)

Move your hand UP / DOWN to move the paddle.

Tips for best tracking
  • Good, even lighting on your hands
  • Plain, non-skin-coloured background if possible
  • If your skin tone isn't being picked up, press +/- to widen/narrow
    the HSV hue range while the game is running

ESC → quit
"""

import cv2
import numpy as np

# ── Window layout ────────────────────────────────────────────────────────────
GAME_W, GAME_H   = 800, 600
CAM_PREV_W       = 480
CAM_PREV_H       = 360
GAME_WIN         = "Pong"
CAM_WIN          = "Hand Tracking  (press +/- to adjust skin detection)"

# ── Colours (BGR) ────────────────────────────────────────────────────────────
BLACK  = (0,   0,   0)
WHITE  = (255, 255, 255)
GREY   = (80,  80,  80)
GREEN  = (0,   220, 100)
RED    = (60,  60,  220)

# ── Paddle ───────────────────────────────────────────────────────────────────
PADDLE_W      = 12
PADDLE_H      = 80
PADDLE_MARGIN = 30

# ── Ball ─────────────────────────────────────────────────────────────────────
BALL_SIZE    = 12
BALL_SPEED_X = 5.0
BALL_SPEED_Y = 4.0
MAX_SPEED    = 14.0

# ── Rules ────────────────────────────────────────────────────────────────────
WINNING_SCORE = 7

# ── Skin HSV detection defaults ──────────────────────────────────────────────
# Hue range covers typical skin tones; adjust with +/- if needed
SKIN_HUE_LOW  = 0      # lower hue (0–180 in OpenCV)
SKIN_HUE_HIGH = 25     # upper hue  — press + to raise, - to lower
SKIN_SAT_LOW  = 30
SKIN_SAT_HIGH = 170
SKIN_VAL_LOW  = 60
SKIN_VAL_HIGH = 255
MIN_BLOB_AREA = 8000   # px² — high enough to ignore face/neck false positives

FRAME_MS = 16          # ~60 fps


# ─────────────────────────────────────────────────────────────────────────────
# Game state
# ─────────────────────────────────────────────────────────────────────────────

def initial_state():
    return {
        "p1": {"x": PADDLE_MARGIN,              "y": GAME_H // 2 - PADDLE_H // 2},
        "p2": {"x": GAME_W - PADDLE_MARGIN - PADDLE_W, "y": GAME_H // 2 - PADDLE_H // 2},
        "ball": {
            "x": float(GAME_W // 2 - BALL_SIZE // 2),
            "y": float(GAME_H // 2 - BALL_SIZE // 2),
            "vx": BALL_SPEED_X,
            "vy": BALL_SPEED_Y,
        },
        "score":        [0, 0],
        "phase":        "waiting",
        "winner":       None,
        "pause_frames": 0,
        "p1_detected":  False,
        "p2_detected":  False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skin detection
# ─────────────────────────────────────────────────────────────────────────────

def skin_mask(hsv, hue_low, hue_high):
    """Return a binary mask of skin-coloured pixels."""
    lo1 = np.array([hue_low,  SKIN_SAT_LOW,  SKIN_VAL_LOW],  dtype=np.uint8)
    hi1 = np.array([hue_high, SKIN_SAT_HIGH, SKIN_VAL_HIGH], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo1, hi1)

    # Also catch the wrap-around reds (hue 160–180) for light skin tones
    lo2 = np.array([160, SKIN_SAT_LOW,  SKIN_VAL_LOW],  dtype=np.uint8)
    hi2 = np.array([180, SKIN_SAT_HIGH, SKIN_VAL_HIGH], dtype=np.uint8)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

    # Clean up noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)
    return mask


def largest_blob_y(mask_roi, roi_h):
    """
    Find the centroid Y of the largest blob in a mask region.
    Returns a normalised value [0, 1], or None if nothing found.
    """
    cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_BLOB_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (M["m01"] / M["m00"]) / roi_h


def detect_hands(frame, hue_low, hue_high):
    """
    Split frame into left / right halves and find a hand in each.
    Returns (left_y_norm, right_y_norm, full_mask)  — y values may be None.
    """
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = skin_mask(hsv, hue_low, hue_high)

    h, w    = frame.shape[:2]
    mid     = w // 2

    left_y  = largest_blob_y(mask[:, :mid],  h)
    right_y = largest_blob_y(mask[:, mid:],  h)

    return left_y, right_y, mask


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def update_paddles(state, left_y, right_y):
    # margin: full paddle range is mapped to the MIDDLE 70% of the camera frame,
    # so players reach top/bottom well before their hand leaves the screen.
    margin = 0.15
    smooth = 0.5   # lerp factor: 1.0 = instant snap, lower = smoother

    if left_y is not None:
        frac   = clamp((left_y - margin) / (1 - 2 * margin), 0.0, 1.0)
        target = int(frac * (GAME_H - PADDLE_H))
        state["p1"]["y"]     = int(state["p1"]["y"] + smooth * (target - state["p1"]["y"]))
        state["p1_detected"] = True
    else:
        state["p1_detected"] = False
        # Hold last position — do NOT update p1["y"]

    if right_y is not None:
        frac   = clamp((right_y - margin) / (1 - 2 * margin), 0.0, 1.0)
        target = int(frac * (GAME_H - PADDLE_H))
        state["p2"]["y"]     = int(state["p2"]["y"] + smooth * (target - state["p2"]["y"]))
        state["p2_detected"] = True
    else:
        state["p2_detected"] = False
        # Hold last position — do NOT update p2["y"]


def move_ball(state):
    b  = state["ball"]
    p1 = state["p1"]
    p2 = state["p2"]

    b["x"] += b["vx"]
    b["y"] += b["vy"]

    # Top / bottom walls
    if b["y"] <= 0:
        b["y"] = 0;  b["vy"] = abs(b["vy"])
    elif b["y"] + BALL_SIZE >= GAME_H:
        b["y"] = GAME_H - BALL_SIZE;  b["vy"] = -abs(b["vy"])

    # P1 paddle (left)
    if (b["vx"] < 0
            and b["x"] <= p1["x"] + PADDLE_W
            and b["x"] + BALL_SIZE >= p1["x"]
            and b["y"] + BALL_SIZE >= p1["y"]
            and b["y"] <= p1["y"] + PADDLE_H):
        b["x"]  = p1["x"] + PADDLE_W
        b["vx"] = min(abs(b["vx"]) + 0.4, MAX_SPEED)
        b["vy"] = ((b["y"] + BALL_SIZE / 2) - (p1["y"] + PADDLE_H / 2)) * 0.22

    # P2 paddle (right)
    if (b["vx"] > 0
            and b["x"] + BALL_SIZE >= p2["x"]
            and b["x"] <= p2["x"] + PADDLE_W
            and b["y"] + BALL_SIZE >= p2["y"]
            and b["y"] <= p2["y"] + PADDLE_H):
        b["x"]  = p2["x"] - BALL_SIZE
        b["vx"] = -min(abs(b["vx"]) + 0.4, MAX_SPEED)
        b["vy"] = ((b["y"] + BALL_SIZE / 2) - (p2["y"] + PADDLE_H / 2)) * 0.22

    if b["x"] + BALL_SIZE < 0:
        state["score"][1] += 1;  return "scored"
    if b["x"] > GAME_W:
        state["score"][0] += 1;  return "scored"
    return "playing"


def reset_ball(state):
    b = state["ball"]
    b["x"]  = float(GAME_W // 2 - BALL_SIZE // 2)
    b["y"]  = float(GAME_H // 2 - BALL_SIZE // 2)
    b["vx"] = -(b["vx"] / abs(b["vx"])) * BALL_SPEED_X
    b["vy"] = BALL_SPEED_Y


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — game canvas
# ─────────────────────────────────────────────────────────────────────────────

def draw_dashed_centre(c):
    x, y, dash, gap = GAME_W // 2 - 1, 0, 20, 15
    while y < GAME_H:
        cv2.rectangle(c, (x, y), (x + 2, min(y + dash, GAME_H)), GREY, -1)
        y += dash + gap


def draw_score(c, score):
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(c, str(score[0]), (GAME_W // 4,          60), f, 2.0, WHITE, 3, cv2.LINE_AA)
    cv2.putText(c, str(score[1]), (3 * GAME_W // 4 - 20, 60), f, 2.0, WHITE, 3, cv2.LINE_AA)


def draw_overlay(c, text, sub=""):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, f, 1.3, 2)
    x = (GAME_W - tw) // 2
    y = (GAME_H + th) // 2
    cv2.putText(c, text, (x, y), f, 1.3, WHITE, 2, cv2.LINE_AA)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, f, 0.60, 1)
        cv2.putText(c, sub, ((GAME_W - sw) // 2, y + 46), f, 0.60, GREY, 1, cv2.LINE_AA)


def render_game(state):
    c = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)
    draw_dashed_centre(c)
    draw_score(c, state["score"])

    p1col = GREEN if state["p1_detected"] else GREY
    p2col = RED   if state["p2_detected"] else GREY
    p1, p2 = state["p1"], state["p2"]
    cv2.rectangle(c, (p1["x"], p1["y"]), (p1["x"] + PADDLE_W, p1["y"] + PADDLE_H), p1col, -1)
    cv2.rectangle(c, (p2["x"], p2["y"]), (p2["x"] + PADDLE_W, p2["y"] + PADDLE_H), p2col, -1)

    if state["phase"] in ("playing", "scored"):
        b = state["ball"]
        cv2.rectangle(c, (int(b["x"]), int(b["y"])),
                         (int(b["x"]) + BALL_SIZE, int(b["y"]) + BALL_SIZE), WHITE, -1)

    if state["phase"] == "waiting":
        if not state["p1_detected"] and not state["p2_detected"]:
            draw_overlay(c, "Show both hands to the camera",
                            "Left hand = left paddle   |   Right hand = right paddle")
        elif not state["p1_detected"]:
            draw_overlay(c, "P1: show your left hand on the left side")
        elif not state["p2_detected"]:
            draw_overlay(c, "P2: show your right hand on the right side")
        else:
            draw_overlay(c, "Get ready!", "Launching...")
    elif state["phase"] == "scored":
        draw_overlay(c, "POINT!")
    elif state["phase"] == "won":
        w = "Player 1" if state["winner"] == 0 else "Player 2"
        draw_overlay(c, f"{w} wins!", "Hide your hands, then show them again to restart")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — webcam preview
# ─────────────────────────────────────────────────────────────────────────────

def render_cam(frame, mask, state, hue_high):
    prev = cv2.resize(frame, (CAM_PREV_W, CAM_PREV_H))
    mid  = CAM_PREV_W // 2

    # Tint detected skin pixels so the player can see what's tracked
    mask_small = cv2.resize(mask, (CAM_PREV_W, CAM_PREV_H))
    overlay    = prev.copy()
    overlay[mask_small > 0] = [0, 180, 80]   # green tint on skin pixels
    cv2.addWeighted(overlay, 0.35, prev, 0.65, 0, prev)

    # Centre divider + zone labels
    cv2.line(prev, (mid, 0), (mid, CAM_PREV_H), GREY, 1)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(prev, "P1",  (8,       CAM_PREV_H - 10), f, 0.6, GREEN, 1, cv2.LINE_AA)
    cv2.putText(prev, "P2",  (mid + 8, CAM_PREV_H - 10), f, 0.6, RED,   1, cv2.LINE_AA)

    # Paddle position indicators on the edges
    ph_px = int(PADDLE_H / GAME_H * CAM_PREV_H)
    p1y   = int(state["p1"]["y"] / GAME_H * CAM_PREV_H)
    p2y   = int(state["p2"]["y"] / GAME_H * CAM_PREV_H)
    cv2.rectangle(prev, (0,            p1y), (6,            p1y + ph_px), GREEN, -1)
    cv2.rectangle(prev, (CAM_PREV_W-6, p2y), (CAM_PREV_W,   p2y + ph_px), RED,   -1)

    # Hue range indicator
    cv2.putText(prev, f"Hue max: {hue_high}  (+/-)",
                (8, 18), f, 0.5, YELLOW if hue_high > 20 else WHITE, 1, cv2.LINE_AA)

    return prev

YELLOW = (0, 220, 220)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cv2.namedWindow(GAME_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(CAM_WIN,  cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(GAME_WIN, 0,          50)
    cv2.moveWindow(CAM_WIN,  GAME_W + 10, 50)

    state    = initial_state()
    hue_low  = SKIN_HUE_LOW
    hue_high = SKIN_HUE_HIGH

    while True:
        ret, raw = cap.read()
        if not ret:
            continue

        frame = cv2.flip(raw, 1)   # mirror so left/right feel natural

        # ── Hand detection ────────────────────────────────────────────────
        left_y, right_y, mask = detect_hands(frame, hue_low, hue_high)
        update_paddles(state, left_y, right_y)
        both = state["p1_detected"] and state["p2_detected"]

        # ── Game logic ────────────────────────────────────────────────────
        if state["phase"] == "waiting":
            if both:
                state["pause_frames"] += 1
                if state["pause_frames"] >= 60:
                    state["pause_frames"] = 0
                    state["phase"] = "playing"
            else:
                state["pause_frames"] = 0

        elif state["phase"] == "playing":
            result = move_ball(state)
            if result == "scored":
                if max(state["score"]) >= WINNING_SCORE:
                    state["phase"]  = "won"
                    state["winner"] = state["score"].index(max(state["score"]))
                else:
                    state["phase"]        = "scored"
                    state["pause_frames"] = 90

        elif state["phase"] == "scored":
            state["pause_frames"] -= 1
            if state["pause_frames"] <= 0:
                reset_ball(state)
                state["phase"] = "playing"

        elif state["phase"] == "won":
            if not both:
                state["pause_frames"] = 0
            else:
                state["pause_frames"] += 1
                if state["pause_frames"] >= 90:
                    state = initial_state()

        # ── Render ────────────────────────────────────────────────────────
        cv2.imshow(GAME_WIN, render_game(state))
        cv2.imshow(CAM_WIN,  render_cam(frame, mask, state, hue_high))

        # ── Input ─────────────────────────────────────────────────────────
        key = cv2.waitKey(FRAME_MS) & 0xFF
        if key == 27:                        # ESC → quit
            break
        elif key == ord('+') or key == ord('='):
            hue_high = min(hue_high + 1, 40)
        elif key == ord('-'):
            hue_high = max(hue_high - 1, 5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
