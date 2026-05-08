"""
Pong — OpenCV + MediaPipe hand-controlled paddles
--------------------------------------------------
Hold your LEFT hand up on the LEFT side of the webcam  → controls left paddle
Hold your RIGHT hand up on the RIGHT side of the webcam → controls right paddle

Move your hand UP / DOWN to move your paddle.
The wrist position is tracked — no specific gesture needed, just raise your hand.

Press ESC to quit, R to restart after a game ends.
"""

import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ── Window & canvas ──────────────────────────────────────────────────────────
GAME_W, GAME_H = 800, 600
CAM_PREVIEW_W  = 480          # width of the webcam preview shown alongside
CAM_PREVIEW_H  = 360

GAME_WIN = "Pong"
CAM_WIN  = "Hand Tracking"

# ── Colours (BGR) ────────────────────────────────────────────────────────────
BLACK       = (0,   0,   0)
WHITE       = (255, 255, 255)
GREY        = (80,  80,  80)
GREEN       = (0,   220, 100)
RED         = (60,  60,  220)
YELLOW      = (0,   220, 220)

# ── Paddle settings ──────────────────────────────────────────────────────────
PADDLE_W      = 12
PADDLE_H      = 80
PADDLE_MARGIN = 30

# ── Ball settings ────────────────────────────────────────────────────────────
BALL_SIZE   = 12
BALL_SPEED_X = 5
BALL_SPEED_Y = 4
MAX_SPEED    = 14

# ── Scoring ──────────────────────────────────────────────────────────────────
WINNING_SCORE = 7

# ── Frame rate ───────────────────────────────────────────────────────────────
FRAME_MS = 16     # ~60 fps target


# ─────────────────────────────────────────────────────────────────────────────
# Game state
# ─────────────────────────────────────────────────────────────────────────────

def initial_state():
    return {
        "p1": {"x": PADDLE_MARGIN,
               "y": GAME_H // 2 - PADDLE_H // 2},
        "p2": {"x": GAME_W - PADDLE_MARGIN - PADDLE_W,
               "y": GAME_H // 2 - PADDLE_H // 2},
        "ball": {
            "x":  float(GAME_W // 2 - BALL_SIZE // 2),
            "y":  float(GAME_H // 2 - BALL_SIZE // 2),
            "vx": float(BALL_SPEED_X),
            "vy": float(BALL_SPEED_Y),
        },
        "score":        [0, 0],
        "phase":        "waiting",   # "waiting" | "playing" | "scored" | "won"
        "winner":       None,
        "pause_frames": 0,
        # Hand visibility flags (used to drive the waiting screen)
        "p1_detected":  False,
        "p2_detected":  False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def move_ball(state):
    ball = state["ball"]
    p1   = state["p1"]
    p2   = state["p2"]

    ball["x"] += ball["vx"]
    ball["y"] += ball["vy"]

    # Top / bottom walls
    if ball["y"] <= 0:
        ball["y"] = 0
        ball["vy"] = abs(ball["vy"])
    elif ball["y"] + BALL_SIZE >= GAME_H:
        ball["y"] = GAME_H - BALL_SIZE
        ball["vy"] = -abs(ball["vy"])

    # Left paddle (P1)
    if (ball["vx"] < 0
            and ball["x"] <= p1["x"] + PADDLE_W
            and ball["x"] + BALL_SIZE >= p1["x"]
            and ball["y"] + BALL_SIZE >= p1["y"]
            and ball["y"] <= p1["y"] + PADDLE_H):
        ball["x"] = p1["x"] + PADDLE_W
        speed = min(abs(ball["vx"]) + 0.4, MAX_SPEED)
        ball["vx"] = speed
        hit = (ball["y"] + BALL_SIZE / 2) - (p1["y"] + PADDLE_H / 2)
        ball["vy"] = hit * 0.22

    # Right paddle (P2)
    if (ball["vx"] > 0
            and ball["x"] + BALL_SIZE >= p2["x"]
            and ball["x"] <= p2["x"] + PADDLE_W
            and ball["y"] + BALL_SIZE >= p2["y"]
            and ball["y"] <= p2["y"] + PADDLE_H):
        ball["x"] = p2["x"] - BALL_SIZE
        speed = min(abs(ball["vx"]) + 0.4, MAX_SPEED)
        ball["vx"] = -speed
        hit = (ball["y"] + BALL_SIZE / 2) - (p2["y"] + PADDLE_H / 2)
        ball["vy"] = hit * 0.22

    # Ball exits left → P2 scores
    if ball["x"] + BALL_SIZE < 0:
        state["score"][1] += 1
        return "scored"

    # Ball exits right → P1 scores
    if ball["x"] > GAME_W:
        state["score"][0] += 1
        return "scored"

    return "playing"


def reset_ball(state):
    b = state["ball"]
    b["x"]  = float(GAME_W // 2 - BALL_SIZE // 2)
    b["y"]  = float(GAME_H // 2 - BALL_SIZE // 2)
    b["vx"] = -b["vx"] / abs(b["vx"]) * BALL_SPEED_X
    b["vy"] = float(BALL_SPEED_Y)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-tracking → paddle mapping
# ─────────────────────────────────────────────────────────────────────────────

def update_paddles_from_hands(state, results, cam_h):
    """
    Map detected wrist positions to paddle Y coordinates.
    We use the wrist landmark (index 0) and its Y in the camera frame.
    Left half of the (mirrored) frame → P1 paddle
    Right half                        → P2 paddle
    """
    state["p1_detected"] = False
    state["p2_detected"] = False

    if not results.multi_hand_landmarks:
        return

    for hand_lms in results.multi_hand_landmarks:
        wrist   = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
        # wrist.x / wrist.y are normalised [0, 1] on the (already flipped) frame
        y_norm  = wrist.y
        x_norm  = wrist.x

        # Map hand Y → paddle Y (with a little dead-zone at top/bottom of frame)
        margin  = 0.1          # ignore the outer 10 % of the frame for stability
        y_frac  = clamp((y_norm - margin) / (1 - 2 * margin), 0.0, 1.0)
        paddle_y = int(y_frac * (GAME_H - PADDLE_H))

        if x_norm < 0.5:      # left half of mirrored frame → P1
            state["p1"]["y"]   = paddle_y
            state["p1_detected"] = True
        else:                  # right half → P2
            state["p2"]["y"]   = paddle_y
            state["p2_detected"] = True


# ─────────────────────────────────────────────────────────────────────────────
# Drawing — game canvas
# ─────────────────────────────────────────────────────────────────────────────

def draw_dashed_centre(canvas):
    dash, gap = 20, 15
    x = GAME_W // 2 - 1
    y = 0
    while y < GAME_H:
        cv2.rectangle(canvas, (x, y), (x + 2, min(y + dash, GAME_H)), GREY, -1)
        y += dash + gap


def draw_score(canvas, score):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, str(score[0]),
                (GAME_W // 4,      60), font, 2.0, WHITE, 3, cv2.LINE_AA)
    cv2.putText(canvas, str(score[1]),
                (3 * GAME_W // 4 - 20, 60), font, 2.0, WHITE, 3, cv2.LINE_AA)


def draw_overlay(canvas, text, sub=""):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.4, 2)
    x = (GAME_W - tw) // 2
    y = (GAME_H - th) // 2
    cv2.putText(canvas, text,  (x, y),      font, 1.4, WHITE, 2, cv2.LINE_AA)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, font, 0.65, 1)
        cv2.putText(canvas, sub, ((GAME_W - sw) // 2, y + 48),
                    font, 0.65, GREY, 1, cv2.LINE_AA)


def render_game(state):
    canvas = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)

    draw_dashed_centre(canvas)
    draw_score(canvas, state["score"])

    # Paddles
    p1, p2 = state["p1"], state["p2"]
    p1_col = GREEN if state["p1_detected"] else GREY
    p2_col = RED   if state["p2_detected"] else GREY
    cv2.rectangle(canvas,
                  (p1["x"], p1["y"]),
                  (p1["x"] + PADDLE_W, p1["y"] + PADDLE_H), p1_col, -1)
    cv2.rectangle(canvas,
                  (p2["x"], p2["y"]),
                  (p2["x"] + PADDLE_W, p2["y"] + PADDLE_H), p2_col, -1)

    # Ball (only during play)
    if state["phase"] in ("playing", "scored"):
        b = state["ball"]
        cv2.rectangle(canvas,
                      (int(b["x"]),             int(b["y"])),
                      (int(b["x"]) + BALL_SIZE, int(b["y"]) + BALL_SIZE),
                      WHITE, -1)

    # Phase overlays
    if state["phase"] == "waiting":
        p1_ok = state["p1_detected"]
        p2_ok = state["p2_detected"]
        if not p1_ok and not p2_ok:
            draw_overlay(canvas, "Show both hands", "Left hand = left paddle   Right hand = right paddle")
        elif not p1_ok:
            draw_overlay(canvas, "P1: show your left hand", "Hold it on the LEFT side of the camera")
        elif not p2_ok:
            draw_overlay(canvas, "P2: show your right hand", "Hold it on the RIGHT side of the camera")
        else:
            draw_overlay(canvas, "Get ready!", "Ball launching in 1 second...")

    elif state["phase"] == "scored":
        draw_overlay(canvas, "POINT!")

    elif state["phase"] == "won":
        w = "Player 1" if state["winner"] == 0 else "Player 2"
        draw_overlay(canvas, f"{w} wins!", "Remove hands then show again to restart")

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Drawing — webcam preview
# ─────────────────────────────────────────────────────────────────────────────

def render_cam(frame, results, state):
    """Resize the camera frame and draw hand landmarks + zone labels on it."""
    preview = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))

    # Draw vertical centre divider
    mid = CAM_PREVIEW_W // 2
    cv2.line(preview, (mid, 0), (mid, CAM_PREVIEW_H), GREY, 1)

    # Zone labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(preview, "P1 (left)",
                (10, CAM_PREVIEW_H - 10), font, 0.55, GREEN, 1, cv2.LINE_AA)
    cv2.putText(preview, "P2 (right)",
                (mid + 10, CAM_PREVIEW_H - 10), font, 0.55, RED, 1, cv2.LINE_AA)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Scale landmarks to preview dimensions
            wrist_x = hand_lms.landmark[mp_hands.HandLandmark.WRIST].x
            colour  = GREEN if wrist_x < 0.5 else RED

            # Draw connections manually so we can colour by player
            for connection in mp_hands.HAND_CONNECTIONS:
                lm_a = hand_lms.landmark[connection[0]]
                lm_b = hand_lms.landmark[connection[1]]
                pt_a = (int(lm_a.x * CAM_PREVIEW_W), int(lm_a.y * CAM_PREVIEW_H))
                pt_b = (int(lm_b.x * CAM_PREVIEW_W), int(lm_b.y * CAM_PREVIEW_H))
                cv2.line(preview, pt_a, pt_b, colour, 1)

            # Draw landmark dots
            for lm in hand_lms.landmark:
                pt = (int(lm.x * CAM_PREVIEW_W), int(lm.y * CAM_PREVIEW_H))
                cv2.circle(preview, pt, 3, WHITE, -1)

    # Paddle position indicators (small horizontal bars)
    p1_y_cam = int(state["p1"]["y"] / GAME_H * CAM_PREVIEW_H)
    p2_y_cam = int(state["p2"]["y"] / GAME_H * CAM_PREVIEW_H)
    cv2.rectangle(preview, (0,       p1_y_cam),
                            (8,       p1_y_cam + int(PADDLE_H / GAME_H * CAM_PREVIEW_H)),
                  GREEN, -1)
    cv2.rectangle(preview, (CAM_PREVIEW_W - 8, p2_y_cam),
                            (CAM_PREVIEW_W,     p2_y_cam + int(PADDLE_H / GAME_H * CAM_PREVIEW_H)),
                  RED, -1)

    return preview


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check it's connected and not in use.")
        return

    cv2.namedWindow(GAME_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(CAM_WIN,  cv2.WINDOW_AUTOSIZE)

    # Position windows side by side (optional — works on most OSes)
    cv2.moveWindow(GAME_WIN, 0,          50)
    cv2.moveWindow(CAM_WIN,  GAME_W + 10, 50)

    state = initial_state()

    with mp_hands.Hands(
        model_complexity=0,           # fastest model
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            # ── Grab camera frame ─────────────────────────────────────────
            ret, raw_frame = cap.read()
            if not ret:
                print("WARNING: Dropped camera frame.")
                continue

            # Mirror so it feels like a mirror — left hand stays on left
            frame = cv2.flip(raw_frame, 1)

            # ── Hand detection ────────────────────────────────────────────
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            update_paddles_from_hands(state, results, frame.shape[0])

            # ── Game logic ────────────────────────────────────────────────
            both_detected = state["p1_detected"] and state["p2_detected"]

            if state["phase"] == "waiting":
                if both_detected:
                    state["pause_frames"] += 1
                    if state["pause_frames"] >= 60:   # 1 s at ~60 fps
                        state["pause_frames"] = 0
                        state["phase"] = "playing"
                else:
                    state["pause_frames"] = 0         # reset countdown

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
                # Restart when no hands detected, then both shown again
                if not both_detected:
                    state["pause_frames"] = 0
                else:
                    state["pause_frames"] += 1
                    if state["pause_frames"] >= 90:
                        state = initial_state()

            # ── Render ────────────────────────────────────────────────────
            game_frame = render_game(state)
            cam_frame  = render_cam(frame, results, state)

            cv2.imshow(GAME_WIN, game_frame)
            cv2.imshow(CAM_WIN,  cam_frame)

            # ── Quit ──────────────────────────────────────────────────────
            key = cv2.waitKey(FRAME_MS) & 0xFF
            if key == 27:   # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
