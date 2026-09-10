#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("wlasl")

YOUTUBE_HOSTS = ("youtube.com", "youtu.be")


def is_youtube(url: str) -> bool:
    return any(h in url for h in YOUTUBE_HOSTS)


def load_instances(json_path: str):
    """Trả về list các dict instance, có thêm field 'gloss'."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            inst = dict(inst)
            inst["gloss"] = gloss
            instances.append(inst)
    return instances


def download_direct(url: str, dest: str, timeout: int = 30) -> bool:
    """Tải file video trực tiếp (không phải YouTube) bằng requests."""
    try:
        with requests.get(url, stream=True, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0"
        }) as r:
            r.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest)
        return True
    except Exception as e:
        log.warning(f"Lỗi tải trực tiếp {url}: {e}")
        return False


def download_youtube(url: str, dest: str) -> bool:
    """Tải video YouTube bằng yt-dlp (cần: pip install yt-dlp)."""
    try:
        import yt_dlp
    except ImportError:
        log.error("Chưa cài yt-dlp. Chạy: pip install yt-dlp")
        return False

    ydl_opts = {
        "outtmpl": dest,
        "format": "mp4/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return os.path.exists(dest)
    except Exception as e:
        log.warning(f"Lỗi tải YouTube {url}: {e}")
        return False


def download_one(inst: dict, out_dir: str) -> tuple[str, bool, str]:
    video_id = inst["video_id"]
    url = inst["url"]
    dest = os.path.join(out_dir, f"{video_id}.mp4")

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return video_id, True, "đã có sẵn, bỏ qua"

    ok = download_youtube(url, dest) if is_youtube(url) else download_direct(url, dest)
    return video_id, ok, url


def main():
    parser = argparse.ArgumentParser(description="Tải video dataset WLASL")
    parser.add_argument("--json", default=f"{os.path.join(ROOT, 'datasets', 'raw', 'WLASL', 'WLASL_v0_3.json')}", help="Đường dẫn file WLASL_v0_3.json")
    parser.add_argument("--out", default=f"{os.path.join(ROOT, 'datasets', 'raw', 'WLASL', 'videos')}", help="Thư mục lưu video")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng tải song song")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ tải N video đầu (0 = tải hết, để test)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    instances = load_instances(args.json)
    if args.limit:
        instances = instances[: args.limit]

    log.info(f"Tổng số video cần tải: {len(instances)}")

    failed = []
    done = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, inst, args.out): inst for inst in instances}
        for fut in as_completed(futures):
            video_id, ok, info = fut.result()
            done += 1
            status = "OK" if ok else "FAIL"
            log.info(f"[{done}/{len(instances)}] {status} {video_id} ({info})")
            if not ok:
                failed.append((video_id, futures[fut]["url"]))

    elapsed = time.time() - start
    log.info(f"Hoàn tất trong {elapsed:.1f}s. Thành công: {len(instances) - len(failed)}, Lỗi: {len(failed)}")

    if failed:
        fail_path = os.path.join(args.out, "failed.log")
        with open(fail_path, "w", encoding="utf-8") as f:
            for vid, url in failed:
                f.write(f"{vid}\t{url}\n")
        log.info(f"Danh sách video lỗi đã ghi vào: {fail_path}")


if __name__ == "__main__":
    sys.exit(main())