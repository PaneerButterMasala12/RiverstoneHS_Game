"""
Pong — OpenCV palm-controlled paddles
--------------------------------------
Uses HSV skin detection + shape filtering to track PALMS specifically,
rejecting arms, faces and necks.

Palm vs arm rejection strategy (no ML required):
  • Compactness  — a palm is roughly circular; an arm is a long thin strip.
                   We compute 4π·area / perimeter² and reject low values.
  • Aspect ratio — bounding box of a palm is close to square; an arm is tall/narrow.
  • Area ceiling — ignore blobs far too large to be just a hand.
  • Smoothing    — exponential moving average + dead-zone kills jitter.

Field layout:
  • Window        800 × 600  (unchanged)
  • Playing field inset PADDLE_H px top and bottom
  • LEFT  half → solid green,  RIGHT half → solid red  (full opacity halves)
  • Paddles travel the full window height (can enter the border zones)
  • Ball bounces inside the inset field only

Controls
  +/-   widen / narrow skin hue range
  ESC   quit
"""

import cv2
import numpy as np

# ── Window ───────────────────────────────────────────────────────────────────
GAME_W, GAME_H = 800, 600
CAM_PREV_W     = 480
CAM_PREV_H     = 360
GAME_WIN       = "Pong"
CAM_WIN        = "Palm Tracking  (+/- adjust skin hue)"

# ── Colours (BGR) ────────────────────────────────────────────────────────────
BLACK        = (0,   0,   0)
WHITE        = (255, 255, 255)
GREY         = (60,  60,  60)
GREY_LIGHT   = (130, 130, 130)
GREEN        = (0,   210, 90)
GREEN_DARK   = (0,   90,  30)
GREEN_FIELD  = (0,   55,  20)       # solid field colour — left half
RED          = (50,  50,  210)
RED_DARK     = (30,  20,  100)
RED_FIELD    = (30,  15,  70)       # solid field colour — right half
YELLOW       = (0,   220, 220)
BORDER_COL   = (18,  18,  18)       # top/bottom dead-zone colour

# ── Paddle ───────────────────────────────────────────────────────────────────
PADDLE_W      = 14
PADDLE_H      = 80
PADDLE_MARGIN = 28

# ── Field ────────────────────────────────────────────────────────────────────
FIELD_TOP  = PADDLE_H               # 80 px from top
FIELD_BOT  = GAME_H - PADDLE_H      # 80 px from bottom  → field height = 440 px
FIELD_H    = FIELD_BOT - FIELD_TOP

# ── Ball ─────────────────────────────────────────────────────────────────────
BALL_SIZE      = 13
BALL_SPEED_X   = 7.5       # serve speed — noticeably faster from the start
BALL_SPEED_Y   = 5.5       # starting vertical component
BALL_SPEED_MAX = 16.0      # ceiling — very fast but still trackable by a good player
BALL_SPEED_K   = 0.22      # steeper curve — speed climbs faster per hit

# ── Complexity ───────────────────────────────────────────────────────────────
SPIN_FACTOR    = 0.88      # how much paddle edge deflects the ball (higher = sharper angles)
NUDGE_RANGE    = 0.08      # ± random nudge on each hit (higher = less predictable)
SERVE_ANGLE    = 28        # max degrees off horizontal on serve (adds variety)

# ── Rules ────────────────────────────────────────────────────────────────────
WINNING_SCORE = 7

# ── Skin HSV ─────────────────────────────────────────────────────────────────
SKIN_HUE_LOW  = 0
SKIN_HUE_HIGH = 25
SKIN_SAT_LOW  = 35
SKIN_SAT_HIGH = 170
SKIN_VAL_LOW  = 60
SKIN_VAL_HIGH = 255

# ── Palm shape filters ───────────────────────────────────────────────────────
# These are the key numbers that reject arms:
MIN_BLOB_AREA   = 3000    # px² — lowered slightly so P2 (often further away) still detected
MAX_BLOB_AREA   = 55000   # px² — reject huge skin blobs (torso / whole arm)
MIN_COMPACTNESS = 0.22    # 4π·A/P² — slightly relaxed so angled palms still pass
MAX_ASPECT      = 3.2     # h/w of bounding box — palm ≈ 1.0, arm ≈ 4–8

# ── Smoothing ────────────────────────────────────────────────────────────────
SMOOTH_ALPHA  = 0.30      # EMA weight for new reading (lower = smoother)
DEAD_ZONE     = 0.008     # normalised — ignore tiny movements (kills jitter)
FALLBACK_MAX  = 12        # frames to hold last position after detection loss

FRAME_MS = 16             # ~60 fps


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
        # Rally tracking
        "rally_hits":   0,          # paddle hits this rally
        "max_rally":    0,          # longest rally this game
        "serve_toward": 1,          # 1 = serve right (toward P2), -1 = serve left (toward P1)
        # Smoothed normalised Y positions (float 0-1), None = unknown
        "_p1_smooth":   None,
        "_p2_smooth":   None,
        # Fallback counters
        "_p1_fallback": 0,
        "_p2_fallback": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Palm detection — the improved core
# ─────────────────────────────────────────────────────────────────────────────

def skin_mask(hsv, hue_low, hue_high):
    """Binary mask of skin-coloured pixels."""
    lo1  = np.array([hue_low,  SKIN_SAT_LOW,  SKIN_VAL_LOW],  dtype=np.uint8)
    hi1  = np.array([hue_high, SKIN_SAT_HIGH, SKIN_VAL_HIGH], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo1, hi1)
    # Wrap-around reds for lighter skin tones
    lo2  = np.array([160, SKIN_SAT_LOW,  SKIN_VAL_LOW],  dtype=np.uint8)
    hi2  = np.array([180, SKIN_SAT_HIGH, SKIN_VAL_HIGH], dtype=np.uint8)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    # Clean up — small kernel keeps fine detail, doesn't bloat thin arm blobs
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def palm_centroid_y(mask_roi, roi_h):
    """
    Return the normalised centroid-Y [0,1] of the blob that looks most
    like an open palm, or None if nothing passes the shape filters.

    Key filters:
      1. Area bounds      — ignore tiny noise and huge torso/arm blobs
      2. Compactness      — 4π·A/P²  (rejects thin arm strips)
      3. Aspect ratio     — h/w of bounding box (rejects tall narrow arms)

    Among blobs that pass all three, we pick the one with the highest
    compactness score (most palm-like).
    """
    cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best_y     = None
    best_score = -1.0

    for c in cnts:
        area = cv2.contourArea(c)

        # ── Filter 1: area bounds ─────────────────────────────────────────────
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue

        # ── Filter 2: compactness (roundness) ────────────────────────────────
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        compactness = (4 * np.pi * area) / (perim * perim)
        if compactness < MIN_COMPACTNESS:
            continue        # arm: long thin → very low compactness

        # ── Filter 3: bounding-box aspect ratio ──────────────────────────────
        _, _, bw, bh = cv2.boundingRect(c)
        aspect = bh / max(bw, 1)
        if aspect > MAX_ASPECT:
            continue        # arm: tall narrow rectangle

        # ── Best candidate: highest compactness ──────────────────────────────
        if compactness > best_score:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            best_y     = (M["m01"] / M["m00"]) / roi_h
            best_score = compactness

    return best_y


def detect_palms(frame, hue_low, hue_high):
    """
    Detect the palm centroid Y in each camera half.
    Each half has a small inward overlap so a hand near the centre
    line isn't split in two and loses area (which caused P2 to fail
    the MIN_BLOB_AREA filter far more often than P1).
    Returns (left_y_norm, right_y_norm, full_mask).
    """
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = skin_mask(hsv, hue_low, hue_high)

    h, w   = frame.shape[:2]
    mid    = w // 2
    MARGIN = w // 10     # 10 % overlap — blob near centre isn't cut

    # Left player: columns 0 … mid+MARGIN
    # Right player: columns mid-MARGIN … w
    # The overlap means a hand near the divider is fully visible in both halves.
    # We use the full frame height (h) so normalised Y is consistent.
    left_y  = palm_centroid_y(mask[:, :mid + MARGIN],  h)
    right_y = palm_centroid_y(mask[:, mid - MARGIN:],  h)

    return left_y, right_y, mask


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _smooth_y(prev_smooth, raw_y):
    """Exponential moving average with dead-zone to suppress jitter."""
    if prev_smooth is None:
        return raw_y
    delta = raw_y - prev_smooth
    if abs(delta) < DEAD_ZONE:
        return prev_smooth          # ignore tiny wobble
    return prev_smooth + SMOOTH_ALPHA * delta


def update_paddles(state, left_y, right_y):
    """
    Map smoothed hand Y (normalised 0-1) → paddle pixel Y.
    Full window range: paddles can enter the border zones.
    Lost detection: hold last position for FALLBACK_MAX frames.
    """
    margin = 0.12
    span   = 1 - 2 * margin

    def apply(raw_y, smooth_key, fallback_key, player_key, detected_key):
        if raw_y is not None:
            state[smooth_key]    = _smooth_y(state[smooth_key], raw_y)
            state[fallback_key]  = FALLBACK_MAX
            state[detected_key]  = True
        else:
            if state[fallback_key] > 0:
                state[fallback_key] -= 1
                state[detected_key]  = True   # hold position during fallback
            else:
                state[detected_key]  = False
                state[smooth_key]    = None
                return

        sy = state[smooth_key]
        if sy is None:
            return
        frac   = clamp((sy - margin) / span, 0.0, 1.0)
        target = int(frac * (GAME_H - PADDLE_H))
        # ── BUG FIX: was always writing to "p1" regardless of player ──────────
        state[player_key]["y"] = target

    apply(left_y,  "_p1_smooth", "_p1_fallback", "p1", "p1_detected")
    apply(right_y, "_p2_smooth", "_p2_fallback", "p2", "p2_detected")


def _rally_speed(hits):
    """
    Exponential approach curve:
      hits=0  → BALL_SPEED_X  (serve speed)
      hits→∞  → BALL_SPEED_MAX (never quite reached)
    Formula: base + (max - base) * (1 - e^(-hits * K))
    """
    import math
    base  = BALL_SPEED_X
    top   = BALL_SPEED_MAX
    return base + (top - base) * (1.0 - math.exp(-hits * BALL_SPEED_K))


def move_ball(state):
    b  = state["ball"]
    p1 = state["p1"]
    p2 = state["p2"]

    b["x"] += b["vx"]
    b["y"] += b["vy"]

    # ── Bounce off inset field top / bottom (NO speed change on wall hits) ────
    if b["y"] <= FIELD_TOP:
        b["y"]  = float(FIELD_TOP)
        b["vy"] = abs(b["vy"])
    elif b["y"] + BALL_SIZE >= FIELD_BOT:
        b["y"]  = float(FIELD_BOT - BALL_SIZE)
        b["vy"] = -abs(b["vy"])

    # ── P1 paddle collision ───────────────────────────────────────────────────
    if (b["vx"] < 0
            and b["x"] <= p1["x"] + PADDLE_W
            and b["x"] + BALL_SIZE >= p1["x"]
            and b["y"] + BALL_SIZE >= p1["y"]
            and b["y"] <= p1["y"] + PADDLE_H):

        state["rally_hits"] += 1
        state["max_rally"]   = max(state["max_rally"], state["rally_hits"])

        new_spd = _rally_speed(state["rally_hits"])
        offset  = ((b["y"] + BALL_SIZE / 2) - (p1["y"] + PADDLE_H / 2)) / (PADDLE_H / 2)
        offset  = max(-1.0, min(1.0, offset))
        nudge   = 1.0 + (np.random.uniform(-NUDGE_RANGE, NUDGE_RANGE))

        b["vx"] = abs(new_spd) * nudge
        b["vy"] = offset * new_spd * SPIN_FACTOR
        b["x"]  = p1["x"] + PADDLE_W + 1

    # ── P2 paddle collision ───────────────────────────────────────────────────
    if (b["vx"] > 0
            and b["x"] + BALL_SIZE >= p2["x"]
            and b["x"] <= p2["x"] + PADDLE_W
            and b["y"] + BALL_SIZE >= p2["y"]
            and b["y"] <= p2["y"] + PADDLE_H):

        state["rally_hits"] += 1
        state["max_rally"]   = max(state["max_rally"], state["rally_hits"])

        new_spd = _rally_speed(state["rally_hits"])
        offset  = ((b["y"] + BALL_SIZE / 2) - (p2["y"] + PADDLE_H / 2)) / (PADDLE_H / 2)
        offset  = max(-1.0, min(1.0, offset))
        nudge   = 1.0 + (np.random.uniform(-NUDGE_RANGE, NUDGE_RANGE))

        b["vx"] = -abs(new_spd) * nudge
        b["vy"] = offset * new_spd * SPIN_FACTOR
        b["x"]  = p2["x"] - BALL_SIZE - 1

    # ── Scoring ───────────────────────────────────────────────────────────────
    if b["x"] + BALL_SIZE < 0:
        state["score"][1]    += 1
        state["serve_toward"] = -1   # serve toward P1 (who just lost the point)
        return "scored"
    if b["x"] > GAME_W:
        state["score"][0]    += 1
        state["serve_toward"] = 1    # serve toward P2 (who just lost the point)
        return "scored"

    return "playing"


def reset_ball(state):
    """Reset ball to centre, restore base speed, serve toward last loser."""
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
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rounded_rect(img, x1, y1, x2, y2, color, r=6):
    """Filled rounded rectangle."""
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def _text_centre(img, text, cy, scale, color, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, ((GAME_W - tw) // 2, cy + th // 2),
                font, scale, color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — game canvas
# ─────────────────────────────────────────────────────────────────────────────

def draw_field(c):
    mid = GAME_W // 2

    # ── Border zones (dead areas above and below the field) ──────────────────
    cv2.rectangle(c, (0, 0),         (GAME_W, FIELD_TOP), BORDER_COL, -1)
    cv2.rectangle(c, (0, FIELD_BOT), (GAME_W, GAME_H),    BORDER_COL, -1)

    # ── Solid coloured halves — full opacity ─────────────────────────────────
    cv2.rectangle(c, (0,   FIELD_TOP), (mid,    FIELD_BOT), GREEN_FIELD, -1)
    cv2.rectangle(c, (mid, FIELD_TOP), (GAME_W, FIELD_BOT), RED_FIELD,   -1)

    # ── Subtle inner gradient strips along each side wall ────────────────────
    # (gives depth without being distracting)
    strip = 18
    for i in range(strip):
        alpha = int(40 * (1 - i / strip))
        col_g = (0,         min(80 + alpha, 255), min(30 + alpha // 2, 255))
        col_r = (min(40 + alpha, 255), min(20 + alpha // 2, 255), min(90 + alpha, 255))
        cv2.line(c, (i,            FIELD_TOP), (i,            FIELD_BOT), col_g, 1)
        cv2.line(c, (GAME_W - 1 - i, FIELD_TOP), (GAME_W - 1 - i, FIELD_BOT), col_r, 1)

    # ── Field boundary lines ─────────────────────────────────────────────────
    cv2.line(c, (0, FIELD_TOP), (GAME_W, FIELD_TOP), WHITE, 2)
    cv2.line(c, (0, FIELD_BOT), (GAME_W, FIELD_BOT), WHITE, 2)

    # ── Centre divider — dashed white line ───────────────────────────────────
    x, y, dash, gap = mid - 1, FIELD_TOP, 18, 12
    while y < FIELD_BOT:
        cv2.rectangle(c, (x, y), (x + 2, min(y + dash, FIELD_BOT)), (100, 100, 100), -1)
        y += dash + gap

    # ── Corner accent dots ───────────────────────────────────────────────────
    r = 5
    for px, py, col in [
        (0,      FIELD_TOP, GREEN), (mid - r, FIELD_TOP, WHITE),
        (mid + r, FIELD_TOP, WHITE), (GAME_W,  FIELD_TOP, RED),
        (0,      FIELD_BOT, GREEN), (mid - r, FIELD_BOT, WHITE),
        (mid + r, FIELD_BOT, WHITE), (GAME_W,  FIELD_BOT, RED),
    ]:
        cv2.circle(c, (px, py), r, col, -1)


def draw_score(c, score, rally_hits, max_rally):
    f   = cv2.FONT_HERSHEY_DUPLEX
    mid = GAME_W // 2
    cy  = FIELD_TOP // 2 + 10   # vertically centred in the top border

    # P1 score — left of centre
    s1 = str(score[0])
    (w1, h1), _ = cv2.getTextSize(s1, f, 1.8, 2)
    cv2.putText(c, s1, (mid // 2 - w1 // 2, cy + h1 // 2), f, 1.8, GREEN, 2, cv2.LINE_AA)

    # P2 score — right of centre
    s2 = str(score[1])
    (w2, h2), _ = cv2.getTextSize(s2, f, 1.8, 2)
    cv2.putText(c, s2, (mid + mid // 2 - w2 // 2, cy + h2 // 2), f, 1.8, RED, 2, cv2.LINE_AA)

    # Thin separator in score bar
    cv2.line(c, (mid, 4), (mid, FIELD_TOP - 4), (50, 50, 50), 1)

    # ── Rally counter — bottom border zone ───────────────────────────────────
    bot_cy = FIELD_BOT + (GAME_H - FIELD_BOT) // 2   # centre of bottom border

    # Rally label + current hit count
    rally_text = f"RALLY  {rally_hits}"
    (rw, rh), _ = cv2.getTextSize(rally_text, f, 0.7, 1)
    # Colour shifts from grey → yellow → orange as rally grows
    intensity = min(rally_hits / 10.0, 1.0)           # 0 at hit 0, 1.0 at hit 10+
    r_col = (
        int(80  + 175 * intensity),                    # B: fades down
        int(130 + 90  * intensity),                    # G: stays warm
        int(80  + 175 * intensity),                    # R: rises to orange/yellow
    )
    # Clamp BGR values
    r_col = tuple(min(255, v) for v in r_col)
    cv2.putText(c, rally_text,
                (mid - rw // 2, bot_cy + rh // 2),
                f, 0.7, r_col, 1, cv2.LINE_AA)

    # Best rally this game
    if max_rally > 0:
        best_text = f"best  {max_rally}"
        (bw, bh), _ = cv2.getTextSize(best_text, f, 0.45, 1)
        cv2.putText(c, best_text,
                    (mid - bw // 2, bot_cy + rh // 2 + 20),
                    f, 0.45, GREY_LIGHT, 1, cv2.LINE_AA)


def draw_paddle(c, x, y, color, dark_color):
    """Paddle with a simple highlight stripe for a 3-D feel."""
    x2 = x + PADDLE_W
    y2 = y + PADDLE_H
    _rounded_rect(c, x, y, x2, y2, color, r=4)
    # Highlight: thin brighter strip on inner edge
    cv2.line(c, (x + 2, y + 4), (x + 2, y2 - 4), WHITE, 1)


def draw_ball(c, bx, by):
    """Ball with a small highlight dot."""
    cx = bx + BALL_SIZE // 2
    cy = by + BALL_SIZE // 2
    r  = BALL_SIZE // 2
    cv2.circle(c, (cx, cy), r,     WHITE,         -1)
    cv2.circle(c, (cx - 2, cy - 2), max(r // 3, 2), (220, 220, 220), -1)


def draw_overlay(c, text, sub=""):
    """Semi-transparent dark pill behind overlay text."""
    f         = cv2.FONT_HERSHEY_SIMPLEX
    field_mid = (FIELD_TOP + FIELD_BOT) // 2
    (tw, th), _ = cv2.getTextSize(text, f, 1.2, 2)
    pad = 18
    x1 = (GAME_W - tw) // 2 - pad
    y1 = field_mid - th - pad
    x2 = (GAME_W + tw) // 2 + pad
    y2 = field_mid + (40 if sub else 10)
    overlay = c.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, c, 0.45, 0, c)
    tx = (GAME_W - tw) // 2
    cv2.putText(c, text, (tx, field_mid), f, 1.2, WHITE, 2, cv2.LINE_AA)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, f, 0.55, 1)
        cv2.putText(c, sub, ((GAME_W - sw) // 2, field_mid + 36),
                    f, 0.55, GREY_LIGHT, 1, cv2.LINE_AA)


def render_game(state):
    c = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)

    draw_field(c)
    draw_score(c, state["score"], state["rally_hits"], state["max_rally"])

    # Paddles
    p1, p2 = state["p1"], state["p2"]
    p1col  = GREEN if state["p1_detected"] else GREY_LIGHT
    p2col  = RED   if state["p2_detected"] else GREY_LIGHT
    draw_paddle(c, p1["x"], p1["y"], p1col, GREEN_DARK)
    draw_paddle(c, p2["x"], p2["y"], p2col, RED_DARK)

    # Ball
    if state["phase"] in ("playing", "scored"):
        b = state["ball"]
        draw_ball(c, int(b["x"]), int(b["y"]))

    # Overlays
    if state["phase"] == "waiting":
        if not state["p1_detected"] and not state["p2_detected"]:
            draw_overlay(c, "Show both hands to camera",
                            "Left hand = left paddle  |  Right hand = right paddle")
        elif not state["p1_detected"]:
            draw_overlay(c, "P1: show your left hand on the left")
        elif not state["p2_detected"]:
            draw_overlay(c, "P2: show your right hand on the right")
        else:
            draw_overlay(c, "Get ready!", "Launching...")
    elif state["phase"] == "scored":
        draw_overlay(c, "POINT!")
    elif state["phase"] == "won":
        w = "Player 1" if state["winner"] == 0 else "Player 2"
        draw_overlay(c, f"{w} wins!", "Hide hands then show again to restart")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — webcam preview
# ─────────────────────────────────────────────────────────────────────────────

def render_cam(frame, mask, state, hue_high):
    prev = cv2.resize(frame, (CAM_PREV_W, CAM_PREV_H))
    mid  = CAM_PREV_W // 2

    # Highlight detected skin pixels
    mask_small = cv2.resize(mask, (CAM_PREV_W, CAM_PREV_H))
    overlay    = prev.copy()
    overlay[mask_small > 0] = [0, 200, 90]
    cv2.addWeighted(overlay, 0.38, prev, 0.62, 0, prev)

    cv2.line(prev, (mid, 0), (mid, CAM_PREV_H), (60, 60, 60), 1)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(prev, "P1", (8,       CAM_PREV_H - 10), f, 0.6, GREEN, 1, cv2.LINE_AA)
    cv2.putText(prev, "P2", (mid + 8, CAM_PREV_H - 10), f, 0.6, RED,   1, cv2.LINE_AA)

    # Paddle position indicators
    ph_px = int(PADDLE_H / GAME_H * CAM_PREV_H)
    p1y   = int(state["p1"]["y"] / GAME_H * CAM_PREV_H)
    p2y   = int(state["p2"]["y"] / GAME_H * CAM_PREV_H)
    cv2.rectangle(prev, (0,            p1y), (5,           p1y + ph_px), GREEN, -1)
    cv2.rectangle(prev, (CAM_PREV_W-5, p2y), (CAM_PREV_W,  p2y + ph_px), RED,   -1)

    cv2.putText(prev, f"Skin hue max: {hue_high}  (+/-)",
                (8, 16), f, 0.44, YELLOW, 1, cv2.LINE_AA)
    cv2.putText(prev, "Hold OPEN PALM flat toward camera",
                (8, 30), f, 0.38, GREY_LIGHT, 1, cv2.LINE_AA)

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

    state    = initial_state()
    hue_low  = SKIN_HUE_LOW
    hue_high = SKIN_HUE_HIGH

    print("=" * 55)
    print("  PONG — Palm Edition")
    print("=" * 55)
    print("  Hold your OPEN PALM flat toward the camera.")
    print("  Arm tracking is filtered out automatically.")
    print("  Press +/- if your skin tone isn't detected.")
    print("  ESC to quit.")
    print("=" * 55)

    while True:
        ret, raw = cap.read()
        if not ret:
            continue

        frame = cv2.flip(raw, 1)

        left_y, right_y, mask = detect_palms(frame, hue_low, hue_high)
        update_paddles(state, left_y, right_y)
        both = state["p1_detected"] and state["p2_detected"]

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
                reset_ball(state)   # resets rally_hits, uses serve_toward
                state["phase"] = "playing"

        elif state["phase"] == "won":
            if not both:
                state["pause_frames"] = 0
            else:
                state["pause_frames"] += 1
                if state["pause_frames"] >= 90:
                    state = initial_state()

        cv2.imshow(GAME_WIN, render_game(state))
        cv2.imshow(CAM_WIN,  render_cam(frame, mask, state, hue_high))

        key = cv2.waitKey(FRAME_MS) & 0xFF
        if key == 27:
            break
        elif key == ord('+') or key == ord('='):
            hue_high = min(hue_high + 1, 40)
        elif key == ord('-'):
            hue_high = max(hue_high - 1, 5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
