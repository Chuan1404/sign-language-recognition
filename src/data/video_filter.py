"""Shared WLASL video exclusions for training, evaluation and reports."""

import config


def excluded_video_ids():
    return {str(video_id) for video_id in config._REMOVE_VIDEO_IDS}
