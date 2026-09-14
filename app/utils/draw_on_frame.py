import cv2 as cv
from config import LEFT_ZONE, RIGHT_ZONE, COLOR_GREEN, COLOR_RED, ZONE_W, ZONE_H, BACKSPACE_ZONE, CLEAR_ZONE, \
    COLOR_ORANGE, COLOR_YELLOW, ACTION_BOX_W, ACTION_BOX_H, ACTION_BOX_MARGIN, FRAME_W, FRAME_H


def point_in_zone(x, y, zone):
    zx, zy, zw, zh = zone
    return zx <= x <= zx + zw and zy <= y <= zy + zh

def draw_zones(frame, left_in_zone, right_in_zone):
    lx, ly, lw, lh = LEFT_ZONE
    rx, ry, rw, rh = RIGHT_ZONE
    cv.rectangle(frame, (lx, ly), (lx + lw, ly + lh),
                 COLOR_GREEN if left_in_zone else COLOR_RED, 2)
    cv.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                 COLOR_GREEN if right_in_zone else COLOR_RED, 2)
    return frame

def get_start_zone(frame_w, frame_h):
    zx = int(frame_w / 2 - ZONE_W / 2)
    zy = int(frame_h / 2 - ZONE_H / 2)
    return (zx, zy, ZONE_W, ZONE_H)

def draw_start_zone(frame, zone, is_recording):
    zx, zy, zw, zh = zone
    color = COLOR_GREEN if is_recording else COLOR_RED
    cv.rectangle(frame, (zx, zy), (zx + zw, zy + zh), color, 2)
    return frame


def draw_action_zones(frame, backspace_active, clear_active):
    """Draw BACKSPACE (top-left) and CLEAR (top-right) boxes on the display frame."""
    bx = ACTION_BOX_MARGIN
    by = ACTION_BOX_MARGIN
    bw = ACTION_BOX_W
    bh = ACTION_BOX_H

    from config import FRAME_W as FW
    cx = FW - ACTION_BOX_W - ACTION_BOX_MARGIN
    cy = ACTION_BOX_MARGIN
    cw = ACTION_BOX_W
    ch = ACTION_BOX_H

    # Backspace box (top-left)
    b_color = COLOR_ORANGE if backspace_active else COLOR_RED
    cv.rectangle(frame, (bx, by), (bx + bw, by + bh), b_color, 2)
    cv.putText(frame, 'BACK', (bx + 8, by + 44),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2)
    cv.putText(frame, 'SPACE', (bx + 4, by + 72),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2)

    # Clear box (top-right)
    c_color = COLOR_YELLOW if clear_active else COLOR_RED
    cv.rectangle(frame, (cx, cy), (cx + cw, cy + ch), c_color, 2)
    cv.putText(frame, 'CLEAR', (cx + 8, cy + 58),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, c_color, 2)

    return frame


def wrist_in_zone(hand_arr, detected, zone):
    if not detected:
        return False
    wx = hand_arr[0, 0] * FRAME_W
    wy = hand_arr[0, 1] * FRAME_H
    return point_in_zone(wx, wy, zone)