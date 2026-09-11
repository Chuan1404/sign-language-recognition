import cv2 as cv
from config import LEFT_ZONE, RIGHT_ZONE, COLOR_GREEN, COLOR_RED, ZONE_W, ZONE_H


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