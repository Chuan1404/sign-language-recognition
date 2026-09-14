import os, json

def build_gloss_list(label_dir, video_dir):
    all_json = []
    for split in ("train.json", "test.json", "val.json"):
        p = os.path.join(label_dir, split)
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_json.extend(data)

    gloss_video = {}
    for item in all_json:
        g = item["gloss"]
        vid = item["video_id"]
        if g not in gloss_video and os.path.exists(os.path.join(video_dir, f"{vid}.mp4")):
            gloss_video[g] = vid

    return [{"gloss": g, "video_id": v} for g, v in sorted(gloss_video.items())]