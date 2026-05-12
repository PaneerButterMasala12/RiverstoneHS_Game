"""
Pong — pen-controlled paddles
------------------------------
Each player holds a coloured pen (or marker) up to the webcam and moves it
up/down to control their paddle.  No hand-tracking, no skin detection —
just vivid colour blobs.

  P1 (left paddle)  — ORANGE / RED pen      default hue 5–22
  P2 (right paddle) — BLUE pen              default hue 100–125

Both players can move anywhere on screen; there is no left/right split.

Tuning keys (if the pen isn't being picked up):
  Q / A   →  raise / lower P1 (orange) upper hue
  P / L   →  raise / lower P2 (blue)   upper hue
  ESC     →  quit
"""

import cv2
import numpy as np

# ── Windows ───────────────────────────────────────────────────────────────────
GAME_W, GAME_H = 800, 600
CAM_PREV_W     = 480
CAM_PREV_H     = 360
GAME_WIN       = "Pong"
CAM_WIN        = "Pen Tracking  (Q/A = P1 hue   P/L = P2 hue)"

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
BLACK       = (0,   0,   0)
WHITE       = (255, 255, 255)
GREY        = (60,  60,  60)
GREY_LIGHT  = (130, 130, 130)
GREEN       = (0,   210, 90)
GREEN_DARK  = (0,   90,  30)
GREEN_FIELD = (0,   55,  20)
RED         = (50,  50,  210)
RED_DARK    = (30,  20,  100)
RED_FIELD   = (30,  15,  70)
YELLOW      = (0,   220, 220)
ORANGE_BGR  = (0,   140, 255)   # display colour for P1 pen hint
BLUE_BGR    = (220, 100, 50)    # display colour for P2 pen hint
BORDER_COL  = (18,  18,  18)

# ── Paddle ────────────────────────────────────────────────────────────────────
PADDLE_W      = 14
PADDLE_H      = 80
PADDLE_MARGIN = 28

# ── Field ─────────────────────────────────────────────────────────────────────
FIELD_TOP = PADDLE_H
FIELD_BOT = GAME_H - PADDLE_H
FIELD_H   = FIELD_BOT - FIELD_TOP

# ── Ball ──────────────────────────────────────────────────────────────────────
BALL_SIZE      = 13
BALL_SPEED_X   = 7.5
BALL_SPEED_Y   = 5.5
BALL_SPEED_MAX = 16.0
BALL_SPEED_K   = 0.22

# ── Physics ───────────────────────────────────────────────────────────────────
SPIN_FACTOR  = 0.88
NUDGE_RANGE  = 0.08
SERVE_ANGLE  = 28

# ── Rules ─────────────────────────────────────────────────────────────────────
WINNING_SCORE = 7

# ── Pen colour defaults (HSV, OpenCV scale: H 0-180, S/V 0-255) ──────────────
# P1 — orange/warm pen
P1_HUE_LOW  = 5
P1_HUE_HIGH = 22     # press Q to raise, A to lower
P1_SAT_LOW  = 130    # high saturation: pen caps are vivid
P1_VAL_LOW  = 80

# P2 — blue pen
P2_HUE_LOW  = 100
P2_HUE_HIGH = 125    # press P to raise, L to lower
P2_SAT_LOW  = 130
P2_VAL_LOW  = 80

# Pen blob size limits (pen tips are small — much tighter than palm limits)
MIN_PEN_AREA = 80     # px²  — ignore tiny specks
MAX_PEN_AREA = 8000   # px²  — ignore large blobs that are definitely not a pen

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTH_ALPHA = 0.45   # EMA weight — higher = snappier, lower = smoother
DEAD_ZONE    = 0.005  # normalised — kills micro-jitter
FALLBACK_MAX = 10     # frames to hold last position on detection loss

FRAME_MS = 16


# ─────────────────────────────────────────────────────────────────────────────
# Game state
# ─────────────────────────────────────────────────────────────────────────────

def initial_state():
    mid_y = GAME_H // 2 - PADDLE_H // 2
    return {
        "p1": {"x": PADDLE_MARGIN,                     "y": mid_y},
        "p2": {"x": GAME_W - PADDLE_MARGIN - PADDLE_W, "y": mid_y},
        "ball": {
            "x":  float(GAME_W // 2 - BALL_SIZE // 2),
            "y":  float((FIELD_TOP + FIELD_BOT) // 2 - BALL_SIZE // 2),
            "vx": BALL_SPEED_X,
            "vy": BALL_SPEED_Y,
        },
        "score":        [0, 0],
        "phase":        "waiting",
        "winner":       None,
        "pause_frames": 0,
        "p1_detected":  False,
        "p2_detected":  False,
        "rally_hits":   0,
        "max_rally":    0,
        "serve_toward": 1,
        "_p1_smooth":   None,
        "_p2_smooth":   None,
        "_p1_fallback": 0,
        "_p2_fallback": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pen detection
# ─────────────────────────────────────────────────────────────────────────────

def _colour_mask(hsv, hue_low, hue_high, sat_low, val_low):
    """
    Build a binary mask for a specific vivid colour.
    Handles the red hue wrap-around (hue > 160) automatically when hue_low < 10.
    """
    lo  = np.array([hue_low,  sat_low, val_low], dtype=np.uint8)
    hi  = np.array([hue_high, 255,     255],      dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    # Red wraps around 180 → also check 160-180 if this is a warm/red colour
    if hue_low <= 15:
        lo2  = np.array([160, sat_low, val_low], dtype=np.uint8)
        hi2  = np.array([180, 255,     255],     dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

    # Light morphology to fill small holes in the blob
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _best_blob_y(mask, frame_h):
    """
    Find the centroid Y of the largest blob inside MIN/MAX_PEN_AREA bounds.
    Returns normalised [0, 1] or None.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Pick largest blob that's within the area window
    valid = [(cv2.contourArea(c), c) for c in cnts
             if MIN_PEN_AREA <= cv2.contourArea(c) <= MAX_PEN_AREA]
    if not valid:
        return None

    _, best = max(valid, key=lambda t: t[0])
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (M["m01"] / M["m00"]) / frame_h


def detect_pens(frame, p1_hue_high, p2_hue_high):
    """
    Detect P1 (orange) and P2 (blue) pen tips independently.
    No frame split — each colour is tracked across the whole frame.
    Returns (p1_y_norm, p2_y_norm, p1_mask, p2_mask).
    """
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h       = frame.shape[0]

    mask1 = _colour_mask(hsv, P1_HUE_LOW, p1_hue_high, P1_SAT_LOW, P1_VAL_LOW)
    mask2 = _colour_mask(hsv, P2_HUE_LOW, p2_hue_high, P2_SAT_LOW, P2_VAL_LOW)

    y1 = _best_blob_y(mask1, h)
    y2 = _best_blob_y(mask2, h)

    return y1, y2, mask1, mask2


# ─────────────────────────────────────────────────────────────────────────────
# Physics / paddle update
# ─────────────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _smooth(prev, raw):
    if prev is None:
        return raw
    delta = raw - prev
    if abs(delta) < DEAD_ZONE:
        return prev
    return prev + SMOOTH_ALPHA * delta


def update_paddles(state, y1, y2):
    margin = 0.08          # reach top/bottom at ~8% from camera edge
    span   = 1 - 2 * margin

    def apply(raw_y, smooth_key, fallback_key, player_key, detected_key):
        if raw_y is not None:
            state[smooth_key]   = _smooth(state[smooth_key], raw_y)
            state[fallback_key] = FALLBACK_MAX
            state[detected_key] = True
        else:
            if state[fallback_key] > 0:
                state[fallback_key] -= 1
                state[detected_key]  = True
            else:
                state[detected_key] = False
                state[smooth_key]   = None
                return

        sy = state[smooth_key]
        if sy is None:
            return
        frac = clamp((sy - margin) / span, 0.0, 1.0)
        state[player_key]["y"] = int(frac * (GAME_H - PADDLE_H))

    apply(y1, "_p1_smooth", "_p1_fallback", "p1", "p1_detected")
    apply(y2, "_p2_smooth", "_p2_fallback", "p2", "p2_detected")


def _rally_speed(hits):
    import math
    base = BALL_SPEED_X
    top  = BALL_SPEED_MAX
    return base + (top - base) * (1.0 - math.exp(-hits * BALL_SPEED_K))


def move_ball(state):
    b  = state["ball"]
    p1 = state["p1"]
    p2 = state["p2"]

    b["x"] += b["vx"]
    b["y"] += b["vy"]

    if b["y"] <= FIELD_TOP:
        b["y"]  = float(FIELD_TOP);  b["vy"] = abs(b["vy"])
    elif b["y"] + BALL_SIZE >= FIELD_BOT:
        b["y"]  = float(FIELD_BOT - BALL_SIZE);  b["vy"] = -abs(b["vy"])

    def paddle_hit(px, py, direction):
        state["rally_hits"] += 1
        state["max_rally"]   = max(state["max_rally"], state["rally_hits"])
        spd    = _rally_speed(state["rally_hits"])
        offset = clamp(((b["y"] + BALL_SIZE/2) - (py + PADDLE_H/2)) / (PADDLE_H/2), -1, 1)
        nudge  = 1.0 + np.random.uniform(-NUDGE_RANGE, NUDGE_RANGE)
        b["vx"] = direction * abs(spd) * nudge
        b["vy"] = offset * spd * SPIN_FACTOR

    if (b["vx"] < 0
            and b["x"] <= p1["x"] + PADDLE_W
            and b["x"] + BALL_SIZE >= p1["x"]
            and b["y"] + BALL_SIZE >= p1["y"]
            and b["y"] <= p1["y"] + PADDLE_H):
        b["x"] = p1["x"] + PADDLE_W + 1
        paddle_hit(p1["x"], p1["y"], +1)

    if (b["vx"] > 0
            and b["x"] + BALL_SIZE >= p2["x"]
            and b["x"] <= p2["x"] + PADDLE_W
            and b["y"] + BALL_SIZE >= p2["y"]
            and b["y"] <= p2["y"] + PADDLE_H):
        b["x"] = p2["x"] - BALL_SIZE - 1
        paddle_hit(p2["x"], p2["y"], -1)

    if b["x"] + BALL_SIZE < 0:
        state["score"][1]    += 1
        state["serve_toward"] = -1
        return "scored"
    if b["x"] > GAME_W:
        state["score"][0]    += 1
        state["serve_toward"] = 1
        return "scored"
    return "playing"


def reset_ball(state):
    import math
    b = state["ball"]
    b["x"]  = float(GAME_W // 2 - BALL_SIZE // 2)
    b["y"]  = float((FIELD_TOP + FIELD_BOT) // 2 - BALL_SIZE // 2)
    angle   = np.random.uniform(-SERVE_ANGLE, SERVE_ANGLE)
    rad     = math.radians(angle)
    b["vx"] = state["serve_toward"] * BALL_SPEED_X * math.cos(rad)
    b["vy"] = BALL_SPEED_Y * math.sin(rad)
    state["rally_hits"] = 0


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — game canvas  (identical to pong_colours_physics.py)
# ─────────────────────────────────────────────────────────────────────────────

def _rounded_rect(img, x1, y1, x2, y2, color, r=6):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def draw_field(c):
    mid = GAME_W // 2
    cv2.rectangle(c, (0, 0),         (GAME_W, FIELD_TOP), BORDER_COL, -1)
    cv2.rectangle(c, (0, FIELD_BOT), (GAME_W, GAME_H),    BORDER_COL, -1)
    cv2.rectangle(c, (0,   FIELD_TOP), (mid,    FIELD_BOT), GREEN_FIELD, -1)
    cv2.rectangle(c, (mid, FIELD_TOP), (GAME_W, FIELD_BOT), RED_FIELD,   -1)
    strip = 18
    for i in range(strip):
        alpha = int(40 * (1 - i / strip))
        col_g = (0, min(80+alpha,255), min(30+alpha//2,255))
        col_r = (min(40+alpha,255), min(20+alpha//2,255), min(90+alpha,255))
        cv2.line(c, (i,          FIELD_TOP), (i,          FIELD_BOT), col_g, 1)
        cv2.line(c, (GAME_W-1-i, FIELD_TOP), (GAME_W-1-i, FIELD_BOT), col_r, 1)
    cv2.line(c, (0, FIELD_TOP), (GAME_W, FIELD_TOP), WHITE, 2)
    cv2.line(c, (0, FIELD_BOT), (GAME_W, FIELD_BOT), WHITE, 2)
    x, y, dash, gap = mid-1, FIELD_TOP, 18, 12
    while y < FIELD_BOT:
        cv2.rectangle(c, (x, y), (x+2, min(y+dash, FIELD_BOT)), (100,100,100), -1)
        y += dash + gap
    r = 5
    for px, py, col in [
        (0, FIELD_TOP, GREEN), (mid-r, FIELD_TOP, WHITE), (mid+r, FIELD_TOP, WHITE), (GAME_W, FIELD_TOP, RED),
        (0, FIELD_BOT, GREEN), (mid-r, FIELD_BOT, WHITE), (mid+r, FIELD_BOT, WHITE), (GAME_W, FIELD_BOT, RED),
    ]:
        cv2.circle(c, (px, py), r, col, -1)


def draw_score(c, score, rally_hits, max_rally):
    f   = cv2.FONT_HERSHEY_DUPLEX
    mid = GAME_W // 2
    cy  = FIELD_TOP // 2 + 10
    s1  = str(score[0])
    (w1,h1),_ = cv2.getTextSize(s1, f, 1.8, 2)
    cv2.putText(c, s1, (mid//2 - w1//2, cy+h1//2), f, 1.8, GREEN, 2, cv2.LINE_AA)
    s2  = str(score[1])
    (w2,h2),_ = cv2.getTextSize(s2, f, 1.8, 2)
    cv2.putText(c, s2, (mid+mid//2-w2//2, cy+h2//2), f, 1.8, RED, 2, cv2.LINE_AA)
    cv2.line(c, (mid,4), (mid, FIELD_TOP-4), (50,50,50), 1)
    bot_cy = FIELD_BOT + (GAME_H - FIELD_BOT) // 2
    rally_text = f"RALLY  {rally_hits}"
    (rw,rh),_ = cv2.getTextSize(rally_text, f, 0.7, 1)
    intensity = min(rally_hits / 10.0, 1.0)
    r_col = tuple(min(255,v) for v in (int(80+175*intensity), int(130+90*intensity), int(80+175*intensity)))
    cv2.putText(c, rally_text, (mid-rw//2, bot_cy+rh//2), f, 0.7, r_col, 1, cv2.LINE_AA)
    if max_rally > 0:
        best_text = f"best  {max_rally}"
        (bw,_),_ = cv2.getTextSize(best_text, f, 0.45, 1)
        cv2.putText(c, best_text, (mid-bw//2, bot_cy+rh//2+20), f, 0.45, GREY_LIGHT, 1, cv2.LINE_AA)


def draw_paddle(c, x, y, color):
    _rounded_rect(c, x, y, x+PADDLE_W, y+PADDLE_H, color, r=4)
    cv2.line(c, (x+2, y+4), (x+2, y+PADDLE_H-4), WHITE, 1)


def draw_ball(c, bx, by):
    cx, cy = bx + BALL_SIZE//2, by + BALL_SIZE//2
    r = BALL_SIZE // 2
    cv2.circle(c, (cx, cy), r, WHITE, -1)
    cv2.circle(c, (cx-2, cy-2), max(r//3,2), (220,220,220), -1)


def draw_overlay(c, text, sub=""):
    f         = cv2.FONT_HERSHEY_SIMPLEX
    field_mid = (FIELD_TOP + FIELD_BOT) // 2
    (tw,th),_ = cv2.getTextSize(text, f, 1.2, 2)
    pad = 18
    x1 = (GAME_W-tw)//2 - pad;  y1 = field_mid - th - pad
    x2 = (GAME_W+tw)//2 + pad;  y2 = field_mid + (40 if sub else 10)
    overlay = c.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, c, 0.45, 0, c)
    cv2.putText(c, text, ((GAME_W-tw)//2, field_mid), f, 1.2, WHITE, 2, cv2.LINE_AA)
    if sub:
        (sw,_),_ = cv2.getTextSize(sub, f, 0.55, 1)
        cv2.putText(c, sub, ((GAME_W-sw)//2, field_mid+36), f, 0.55, GREY_LIGHT, 1, cv2.LINE_AA)


def render_game(state):
    c = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)
    draw_field(c)
    draw_score(c, state["score"], state["rally_hits"], state["max_rally"])
    p1, p2 = state["p1"], state["p2"]
    draw_paddle(c, p1["x"], p1["y"], GREEN if state["p1_detected"] else GREY_LIGHT)
    draw_paddle(c, p2["x"], p2["y"], RED   if state["p2_detected"] else GREY_LIGHT)
    if state["phase"] in ("playing", "scored"):
        b = state["ball"]
        draw_ball(c, int(b["x"]), int(b["y"]))
    if state["phase"] == "waiting":
        if not state["p1_detected"] and not state["p2_detected"]:
            draw_overlay(c, "Show both pens to camera",
                            "P1 = orange pen   |   P2 = blue pen")
        elif not state["p1_detected"]:
            draw_overlay(c, "P1: show your orange pen")
        elif not state["p2_detected"]:
            draw_overlay(c, "P2: show your blue pen")
        else:
            draw_overlay(c, "Get ready!", "Launching...")
    elif state["phase"] == "scored":
        draw_overlay(c, "POINT!")
    elif state["phase"] == "won":
        w = "Player 1" if state["winner"] == 0 else "Player 2"
        draw_overlay(c, f"{w} wins!", "Hide pens then show again to restart")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — webcam preview
# ─────────────────────────────────────────────────────────────────────────────

def render_cam(frame, mask1, mask2, state, p1_hue_high, p2_hue_high):
    prev = cv2.resize(frame, (CAM_PREV_W, CAM_PREV_H))
    m1   = cv2.resize(mask1, (CAM_PREV_W, CAM_PREV_H))
    m2   = cv2.resize(mask2, (CAM_PREV_W, CAM_PREV_H))

    # Tint P1 blobs orange, P2 blobs blue
    overlay = prev.copy()
    overlay[m1 > 0] = [0, 140, 255]   # orange-ish tint for P1
    overlay[m2 > 0] = [220, 80,  40]  # blue tint for P2
    cv2.addWeighted(overlay, 0.5, prev, 0.5, 0, prev)

    f = cv2.FONT_HERSHEY_SIMPLEX

    # Paddle position indicators
    ph_px = int(PADDLE_H / GAME_H * CAM_PREV_H)
    p1y   = int(state["p1"]["y"] / GAME_H * CAM_PREV_H)
    p2y   = int(state["p2"]["y"] / GAME_H * CAM_PREV_H)
    cv2.rectangle(prev, (0,            p1y), (5,           p1y+ph_px), GREEN, -1)
    cv2.rectangle(prev, (CAM_PREV_W-5, p2y), (CAM_PREV_W,  p2y+ph_px), RED,   -1)

    # Hue readouts
    cv2.putText(prev, f"P1 hue max: {p1_hue_high}  (Q/A)",
                (8, 16), f, 0.42, ORANGE_BGR, 1, cv2.LINE_AA)
    cv2.putText(prev, f"P2 hue max: {p2_hue_high}  (P/L)",
                (8, 30), f, 0.42, BLUE_BGR,   1, cv2.LINE_AA)

    # Detected labels
    if state["p1_detected"]:
        cv2.putText(prev, "P1 OK", (8, CAM_PREV_H-10), f, 0.55, GREEN, 1, cv2.LINE_AA)
    if state["p2_detected"]:
        cv2.putText(prev, "P2 OK", (CAM_PREV_W-70, CAM_PREV_H-10), f, 0.55, RED, 1, cv2.LINE_AA)

    return prev


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
    cv2.moveWindow(GAME_WIN, 0,           50)
    cv2.moveWindow(CAM_WIN,  GAME_W + 10, 50)

    state      = initial_state()
    p1_hue_high = P1_HUE_HIGH
    p2_hue_high = P2_HUE_HIGH

    print("=" * 55)
    print("  PONG — Pen Edition")
    print("=" * 55)
    print("  P1: hold an ORANGE or RED pen to the camera")
    print("  P2: hold a BLUE pen to the camera")
    print("  Move the pen UP / DOWN to move your paddle.")
    print("  Q/A to tune P1 colour detection.")
    print("  P/L to tune P2 colour detection.")
    print("  ESC to quit.")
    print("=" * 55)

    while True:
        ret, raw = cap.read()
        if not ret:
            continue

        frame = cv2.flip(raw, 1)

        y1, y2, mask1, mask2 = detect_pens(frame, p1_hue_high, p2_hue_high)
        update_paddles(state, y1, y2)
        both = state["p1_detected"] and state["p2_detected"]

        # ── Game logic (identical to pong_colours_physics.py) ─────────────
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
        cv2.imshow(CAM_WIN,  render_cam(frame, mask1, mask2, state, p1_hue_high, p2_hue_high))

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(FRAME_MS) & 0xFF
        if   key == 27:          break                                    # ESC
        elif key == ord('q'):    p1_hue_high = min(p1_hue_high + 1, 40)  # P1 hue up
        elif key == ord('a'):    p1_hue_high = max(p1_hue_high - 1, 5)   # P1 hue down
        elif key == ord('p'):    p2_hue_high = min(p2_hue_high + 1, 135) # P2 hue up
        elif key == ord('l'):    p2_hue_high = max(p2_hue_high - 1, 95)  # P2 hue down

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
