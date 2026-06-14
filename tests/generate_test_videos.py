# tests/generate_test_videos.py
# coding: utf-8
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("tests/sample_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ("1080p_sdr.mp4", "1920x1080", 30, False),
    ("720p_sdr.mp4", "1280x720", 30, False),
    ("4k_sdr.mp4", "3840x2160", 30, False),
    ("1080p_hdr.mp4", "1920x1080", 30, True),
    ("4k_hdr.mp4", "3840x2160", 30, True),
]

for name, res, fps, is_hdr in VIDEOS:
    out_file = OUTPUT_DIR / name
    if out_file.exists():
        print(f"已存在: {out_file}, 跳过生成")
        continue

    width, height = res.split("x")
    if is_hdr:
        # HDR: 10-bit + bt2020 primaries + smpte2084 transfer + bt2020nc matrix
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={res}:rate={fps}:duration=5",
            "-vf", "format=yuv420p10le,format=pix_fmts=yuv420p10le",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
            str(out_file),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={res}:rate={fps}:duration=5",
            str(out_file),
        ]

    print(f"生成: {out_file}")
    subprocess.run(cmd, check=True)

print("Done.")
