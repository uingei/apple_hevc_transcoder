#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apple_hevc_batch — CLI 入口，直接复用 core.transcoder.convert_video()

转码逻辑统一由 core.transcoder 提供，此处仅负责：
- 命令行参数解析
- 文件发现与并发调度
- 日志输出与 CSV 记录
"""
__version__ = "1.7.1"

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# 统一导入核心模块（CLI 与 GUI 共享同一套管线）
from core.transcoder import convert_video, probe_media
import config

# -------------------- config --------------------
INPUT_EXTS = {
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg',
}
LOG_FILE = "transcode_log.csv"
MAX_WORKERS_SDR = max(1, os.cpu_count() or 1)
MAX_WORKERS_HDR = min(4, max(1, (os.cpu_count() or 4) // 4))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -------------------- tool checks --------------------
def check_tools():
    missing = []
    for tool in ('ffmpeg', 'ffprobe'):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        logger.error("Missing required tools: %s. Install and ensure they are in PATH.", ", ".join(missing))
        raise SystemExit(1)

# -------------------- concurrency helpers --------------------
def dynamic_workers():
    """动态工作线程数，考虑 CPU 数量"""
    return max(1, os.cpu_count() or 1)

# -------------------- batch convert --------------------
def batch_convert(input_dir: Path, output_dir: Path, max_workers: int = 4,
                  debug: bool = False, skip_validator: bool = False,
                  force_cpu: bool = False, force_gpu: bool = False,
                  progress_callback=None):
    """批处理转码，直接调用 core.transcoder.convert_video()"""
    files = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS]
    if not files:
        logger.warning("No input videos found in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(
                convert_video,
                file_path=f,
                out_dir=output_dir,
                debug=debug,
                skip_validator=skip_validator,
                force_cpu=force_cpu,
                force_gpu=force_gpu,
            ): f
            for f in files
        }
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Transcoding"):
            try:
                result = fut.result()
                results.append(result)
            except Exception as e:
                logger.exception("Unexpected error for %s: %s", future_map[fut].name, e)
                results.append({"file": future_map[fut].name, "status": "ERROR", "error": str(e)})

    # write CSV log
    try:
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["file", "status", "quality", "retries", "method", "hdr"])
            writer.writeheader()
            for r in results:
                safe = {k: r.get(k) for k in writer.fieldnames}
                writer.writerow(safe)
    except Exception:
        logger.exception("Failed to write log file")

    return results

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Apple HEVC batch transcoder (v1.7.1) — CLI powered by core.transcoder"
    )
    p.add_argument("-i", "--input", required=True, dest="input_dir", help="Input directory")
    p.add_argument("-o", "--output", required=True, dest="output_dir", help="Output directory")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--skip-validator", action="store_true", help="Skip Apple heuristic validation")
    p.add_argument("--force-cpu", action="store_true", help="Force CPU encoding even if NVENC is available")
    p.add_argument("--force-gpu", action="store_true", help="Force GPU encoding")
    p.add_argument("-j", "--workers", type=int, default=None, help="Max concurrent workers (default: auto)")
    return p.parse_args()


def main():
    """CLI 主入口"""
    print(f"Apple HEVC Transcoder CLI v{config.APP_VERSION}")
    print("https://github.com/uingei/apple_hevc_transcoder")
    print()

    args = parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    input_path = Path(args.input_dir).expanduser()
    output_path = Path(args.output_dir).expanduser()
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        raise SystemExit(1)

    check_tools()

    # 采样探针，判定是否存在 HDR 以调节并发度
    sample_files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS][:6]
    any_hdr = False
    try:
        any_hdr = any(probe_media(s).hdr for s in sample_files)
    except Exception as e:
        logger.warning("Sampling probe failed: %s", e)

    if args.workers is not None:
        max_workers = args.workers
    else:
        max_workers = min(dynamic_workers(), MAX_WORKERS_HDR) if any_hdr else min(dynamic_workers(), MAX_WORKERS_SDR)

    results = batch_convert(
        input_dir=input_path,
        output_dir=output_path,
        max_workers=max_workers,
        debug=args.debug,
        skip_validator=args.skip_validator,
        force_cpu=args.force_cpu,
        force_gpu=args.force_gpu,
    )

    if results:
        success = sum(1 for r in results if r.get("status") == "SUCCESS")
        failed = sum(1 for r in results if r.get("status") != "SUCCESS")
        print(f"\nDone: {success} succeeded, {failed} failed (log: {LOG_FILE})")
    else:
        print("No files processed.")


if __name__ == "__main__":
    main()
