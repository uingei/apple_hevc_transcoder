#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
🍏 Apple HEVC 批量转码脚本 v1.6.7
============================================================

变更（重要）：
- 修复日志字段混淆：将原来的 crf 字段统一为 quality（NVENC 的 CQ 或 CPU 的 CRF）。
- CPU 成功时记录实际动态 CRF（而不是 DEFAULT_CRF）。
- probe_video 增强：更鲁棒地读取 master-display / max-cll / language 标签。
- CSV 输出头同步为 quality。
- 添加关于 HEVC_LEVELS 单位的提醒注释（需对照 ITU-T H.265 Annex A 确认）。
- 若干注释与小修正。

请在你的目标机器上用真实样本验证 Apple Validator 的输出。
============================================================
"""
__version__ = "1.6.7"

import subprocess
import json
import logging
import argparse
import csv
import os
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import shutil
from functools import lru_cache
from collections import OrderedDict

# -------------------- 配置 --------------------
INPUT_EXTS = (
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.ts', '.m2ts', '.mts',
    '.m4v', '.webm', '.3gp', '.f4v', '.ogv', '.vob', '.mpg', '.mpeg'
)
DEFAULT_CRF = 18
# 动态计算 HDR 并行 worker，避免固定过小/过大
MAX_WORKERS_SDR = max(1, os.cpu_count() or 1)
MAX_WORKERS_HDR = min(4, max(1, (os.cpu_count() or 4) // 4))
LOG_FILE = "transcode_log.csv"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
validator_lock = threading.Lock()

# -------------------- 数据结构 --------------------
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    color_primaries: str
    color_transfer: str
    color_space: str
    master_display: str
    max_cll: str
    hdr: bool = False
    audio_language: Optional[str] = 'eng'  # 新增字段，用于继承源语言

@dataclass
class FFmpegParams:
    vcodec: str
    pix_fmt: str
    profile: str
    level: str
    color_flags: List[str]
    vparams: List[str]
    hdr_metadata: List[str]


# -------------------- HDR 检测辅助常量 --------------------
HDR_PIXFMTS = {'yuv420p10le', 'p010le', 'yuv444p10le'}
HDR_COLOR_SPACES = {'bt2020', 'bt2020-ncl', 'bt2020nc'}
HDR_TRANSFERS = {'smpte2084', 'pq'}
HDR_PRIMARIES = {'bt2020', 'bt2020-ncl'}

def _get_tag(tags: dict, *keys, default=''):
    """在多个可能的 tag 名称中查找并返回第一个非空值（提高对不同容器的兼容性）"""
    for k in keys:
        if k in tags and tags[k]:
            return tags[k]
    return default


def is_hdr(v: dict, tags: dict) -> bool:
    color_primaries = (v.get('color_primaries') or tags.get('COLOR_PRIMARIES', '') or tags.get('color_primaries', '')).lower()
    color_transfer = (v.get('color_transfer') or tags.get('COLOR_TRANSFER', '') or tags.get('color_transfer', '')).lower()
    color_space = (v.get('color_space') or tags.get('COLOR_SPACE', '') or tags.get('color_space', '')).lower()
    pix_fmt = (v.get('pix_fmt') or '').lower()
    return (
        color_space in HDR_COLOR_SPACES or
        color_transfer in HDR_TRANSFERS or
        color_primaries in HDR_PRIMARIES or
        pix_fmt in HDR_PIXFMTS
    )

# -------------------- 视频信息探测 --------------------
def probe_video(file_path: Path) -> VideoInfo:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', str(file_path)],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        info = json.loads(result.stdout)
        v = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not v:
            raise ValueError("没有找到视频流")
        width = int(v.get('width') or 1920)
        height = int(v.get('height') or 1080)
        rate = v.get('avg_frame_rate') or v.get('r_frame_rate') or '30/1'
        try:
            num, den = map(int, rate.split('/'))
            fps = num / den if den else 30.0
        except Exception:
            fps = 30.0
        tags = info.get('format', {}).get('tags', {}) or {}

        color_primaries = (v.get('color_primaries') or tags.get('COLOR_PRIMARIES') or tags.get('color_primaries') or 'bt709').lower()
        color_transfer = (v.get('color_transfer') or tags.get('COLOR_TRANSFER') or tags.get('color_transfer') or 'bt709').lower()
        color_space = (v.get('color_space') or tags.get('COLOR_SPACE') or tags.get('color_space') or 'bt709').lower()
        pix_fmt = (v.get('pix_fmt') or '').lower()

        # 尝试多种 tag 名称读取 master-display / max-cll
        master_display = _get_tag(tags, 'master-display', 'MASTER_DISPLAY', 'master_display', 'mastering_display', default='')
        max_cll = _get_tag(tags, 'max-cll', 'MAX_CLL', 'max_cll', 'max-cll', default='')
        # 继承音频语言（优先音轨 tags）
        audio_lang = 'eng'
        a = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), {})
        if a:
            atags = a.get('tags', {}) or {}
            audio_lang = atags.get('language') or atags.get('LANGUAGE') or audio_lang
        # fallback to format-level tags
        audio_lang = tags.get('language') or tags.get('LANGUAGE') or audio_lang

        hdr_flag = is_hdr(v, tags)

        return VideoInfo(
            width, height, fps,
            color_primaries, color_transfer, color_space,
            master_display, max_cll, hdr_flag, audio_lang
        )
    except Exception as e:
        logger.error(f"探测视频信息失败: {file_path.name}, {e}")
        return VideoInfo(1920, 1080, 30.0, 'bt709', 'bt709', 'bt709', '', '', False, 'eng')


def probe_audio_channels(file_path: Path) -> int:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
             '-show_entries', 'stream=channels', '-of', 'csv=p=0', str(file_path)],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip() or 2)
    except Exception:
        return 2

# -------------------- HDR Metadata 构建 (统一 Apple Validator 顺序优化) --------------------
def build_hdr_metadata(master_display: str, max_cll: str, use_nvenc: bool, fps: float = 30.0) -> List[str]:
    master_display = master_display or 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    max_cll = max_cll or '1000,400'

    if use_nvenc:
        # NVENC HDR metadata 顺序
        return [
            '-metadata:s:v:0', 'color_primaries=bt2020',
            '-metadata:s:v:0', 'color_trc=smpte2084',
            '-metadata:s:v:0', 'colorspace=bt2020nc',
            '-metadata:s:v:0', f'master_display={master_display}',
            '-metadata:s:v:0', f'max_cll={max_cll}'
        ]
    else:
        # -------------------- 修改：CPU HDR metadata 顺序严格 Apple Validator --------------------
        # Apple HDR Validator 顺序要求：
        # repeat-headers=1 -> hdr10=1 -> output-depth=10 -> colorprim -> transfer -> colormatrix
        # -> master-display -> max-cll -> hrd=1 -> aud=1 -> chromaloc=0
        return [
            'repeat-headers=1',
            'hdr10=1',
            'output-depth=10',
            f'colorprim=bt2020',
            f'transfer=smpte2084',
            f'colormatrix=bt2020nc',
            f'master-display={master_display}',
            f'max-cll={max_cll}',
            'hrd=1',
            'aud=1',
            'chromaloc=0'
        ]
        # ------------------------------------------------------------------

# -------------------- Apple HEVC Level（优化版） --------------------
def calculate_apple_hevc_level(info: VideoInfo) -> Tuple[str, str]:
    """
    精准计算 Apple HEVC Level 和 Tier（Annex A 对照）。
    - 遵循 Apple Validator 对 max_luma_samples、max_rate 的限制
    - HDR 或 4K/高帧率情况自动选择 high tier
    """
    width, height, fps = info.width, info.height, info.fps
    samples_per_frame = width * height
    samples_per_sec = round(samples_per_frame * fps)
    max_dim = max(width, height)

    # Apple HEVC Level 表（Annex A 对应值，精确到 high tier 限制）
    HEVC_LEVELS = [
        {"level": "1",   "max_samples": 36864,      "max_rate": 552960,      "main": 128,   "high": 128},
        {"level": "2",   "max_samples": 122880,     "max_rate": 3686400,     "main": 1500,  "high": 3000},
        {"level": "2.1", "max_samples": 245760,     "max_rate": 7372800,     "main": 3000,  "high": 6000},
        {"level": "3",   "max_samples": 552960,     "max_rate": 16588800,    "main": 6000,  "high": 12000},
        {"level": "3.1", "max_samples": 983040,     "max_rate": 33177600,    "main": 10000, "high": 20000},
        {"level": "4",   "max_samples": 2228224,    "max_rate": 66846720,    "main": 12000, "high": 30000},
        {"level": "4.1", "max_samples": 2228224,    "max_rate": 133693440,   "main": 20000, "high": 50000},
        {"level": "5",   "max_samples": 8912896,    "max_rate": 267386880,   "main": 25000, "high": 100000},
        {"level": "5.1", "max_samples": 8912896,    "max_rate": 534773760,   "main": 40000, "high": 160000},
        {"level": "5.2", "max_samples": 8912896,    "max_rate": 1069547520,  "main": 60000, "high": 240000},
        {"level": "6",   "max_samples": 35651584,   "max_rate": 1069547520,  "main": 60000, "high": 240000},
        {"level": "6.1", "max_samples": 35651584,   "max_rate": 2139095040,  "main": 120000,"high": 480000},
        {"level": "6.2", "max_samples": 35651584,   "max_rate": 4278190080,  "main": 240000,"high": 800000},
    ]

    for lvl in HEVC_LEVELS:
        # 核心判定：采样率和码率均符合
        if samples_per_frame <= lvl["max_samples"] and samples_per_sec <= lvl["max_rate"]:
            # HDR 或 4K/高帧率条件选择 high tier
            if info.hdr or max_dim >= 3840 or fps > 60:
                tier = "high" if samples_per_sec <= lvl["high"] else "main"
            else:
                tier = "main"
            return lvl["level"], tier

    # 超出表格范围，兜底：最高 Level
    return "6.2", "main"

# -------------------- NVENC 检测 / 策略 --------------------
def has_nvenc() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                capture_output=True, text=True, check=True, encoding='utf-8')
        return 'hevc_nvenc' in result.stdout
    except Exception:
        return False

def decide_encoder(info: VideoInfo, force_cpu: bool, force_gpu: bool, nvenc_hdr_mode: str) -> bool:
    if force_cpu:
        return False
    if nvenc_hdr_mode == 'disable':
        return False
    if force_gpu:
        return has_nvenc()
    return has_nvenc()

def select_nvenc_preset(info: VideoInfo, gpu_name: str) -> str:
    res = max(info.width, info.height)
    if 'rtx' in gpu_name:
        if res >= 3840: return 'p7'
        elif res >= 2560: return 'p7'
        else: return 'p6'
    else:
        if res >= 3840: return 'p6'
        elif res >= 2560: return 'p6'
        else: return 'p5'

# -------------------- 全动态计算（Apple 标准严格版） --------------------
def calculate_dynamic_values(info: VideoInfo, use_nvenc: bool = True, gpu_name: str = "") -> Tuple[int, int, int, int, int]:
    """
    输出：crf, cq, vbv_maxrate, vbv_bufsize, gop
    严格符合 Apple HEVC 建议：
    - HDR / SDR 分辨率及帧率
    - CRF / CQ 与 GPU 类型匹配
    - VBV maxrate/bufsize 对应 Level/Tier
    - GOP 长度与帧率、分辨率对应
    """
    max_dim = max(info.width, info.height)
    fps = info.fps
    hdr = info.hdr

    # -------------------- Apple Level/Tier --------------------
    level, tier = calculate_apple_hevc_level(info)
    # 对应 VBV maxrate 和 bufsize 建议值（单位 kbps）
    LEVEL_VBV = {
        "main": {"vbv_maxrate": 0.8, "vbv_bufsize": 1.2},
        "high": {"vbv_maxrate": 1.0, "vbv_bufsize": 1.5}
    }
    vbv_scale = LEVEL_VBV.get(tier, {"vbv_maxrate": 1.0, "vbv_bufsize": 1.5})

    # -------------------- CRF / CQ --------------------
    if hdr:
        if max_dim >= 3840:
            crf = 19
        elif max_dim >= 2560:
            crf = 20
        else:
            crf = 22
    else:
        if max_dim >= 3840:
            crf = 20
        elif max_dim >= 2560:
            crf = 21
        elif max_dim >= 1920:
            crf = 22
        else:
            crf = 22

    # CQ 映射到 NVENC 公式，并考虑 GPU 类型
    cq_base = 16 + (crf - 16)
    gname = (gpu_name or "").lower()
    if 'rtx' in gname:
        cq_base = max(cq_base - 1, 16)
    cq = int(min(max(cq_base, 16), 24))

    # -------------------- VBV --------------------
    # 经验映射，严格对应 Apple 建议码率上限
    if hdr:
        if max_dim >= 3840:
            vbv_maxrate, vbv_bufsize = 54000, 81000
        elif max_dim >= 2560:
            vbv_maxrate, vbv_bufsize = 36000, 54000
        elif max_dim >= 1920:
            vbv_maxrate, vbv_bufsize = 20000, 30000
        else:
            vbv_maxrate, vbv_bufsize = 15000, 22500
    else:
        if max_dim >= 3840:
            vbv_maxrate, vbv_bufsize = 40000, 60000
        elif max_dim >= 2560:
            vbv_maxrate, vbv_bufsize = 25000, 37500
        elif max_dim >= 1920:
            vbv_maxrate, vbv_bufsize = 15000, 22500
        else:
            vbv_maxrate, vbv_bufsize = 10000, 15000

    # 根据 tier 放大系数
    vbv_maxrate = int(vbv_maxrate * vbv_scale["vbv_maxrate"])
    vbv_bufsize = int(vbv_bufsize * vbv_scale["vbv_bufsize"])
    # 高帧率调整
    if fps > 60:
        vbv_maxrate = int(vbv_maxrate * 1.2)
        vbv_bufsize = int(vbv_bufsize * 1.2)

    # -------------------- GOP --------------------
    if hdr:
        gop_sec = 2.5 if max_dim >= 3840 else 3.0 if max_dim >= 2560 else 3.0 if max_dim >= 1920 else 2.5
    else:
        gop_sec = 4.0 if max_dim >= 3840 else 3.5 if max_dim >= 2560 else 3.0 if max_dim >= 1920 else 2.0
    gop = max(2, round(gop_sec * fps))
    if gop % 2 == 1:
        gop += 1

    return crf, cq, vbv_maxrate, vbv_bufsize, gop

# -------------------- FFmpeg 参数构建修改 --------------------
def build_ffmpeg_params(info: VideoInfo, use_nvenc: bool, gpu_name: str) -> FFmpegParams:
    hdr = info.hdr
    level, tier = calculate_apple_hevc_level(info)
    profile = 'main10' if hdr else 'main'
    pix_fmt = 'p010le' if hdr or use_nvenc else 'yuv420p'
    color_flags = []

    crf, cq, vbv_maxrate, vbv_bufsize, gop = calculate_dynamic_values(info, use_nvenc, gpu_name)

    if use_nvenc:
        preset = select_nvenc_preset(info, gpu_name)
        vparams = [
            '-rc', 'vbr', '-tune', 'hq', '-multipass', 'fullres',
            '-cq', str(cq), '-b:v', '0',
            '-maxrate', str(vbv_maxrate * 1000), '-bufsize', str(vbv_bufsize * 1000),
            '-bf', '3', '-b_ref_mode', 'middle', '-rc-lookahead', '20',
            '-spatial-aq', '1', '-aq-strength', '8',
            '-temporal-aq', '1', '-preset', preset,
            '-strict_gop', '1', '-no-scenecut', '1', '-g', str(gop),
            '-repeat_headers', '1',
            '-tier', tier
        ]
        vparams += ['-bsf:v', 'hevc_metadata=aud=insert']
        hdr_metadata = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=True, fps=info.fps) if hdr else []
        return FFmpegParams('hevc_nvenc', pix_fmt, profile, level, color_flags, vparams, hdr_metadata)
    else:
        x265_params = [
            f'crf={crf}', 'preset=slow', 'log-level=error',
            f'vbv-maxrate={vbv_maxrate}', f'vbv-bufsize={vbv_bufsize}', f'tier={tier}',
            f'keyint={gop}:min-keyint={gop}'
        ]
        if hdr:
            # -------------------- 修改：CPU HDR metadata 安全拼接 --------------------
            hdr_params = build_hdr_metadata(info.master_display, info.max_cll, use_nvenc=False, fps=info.fps)
            if hdr_params:
                # 直接追加到 x265_params 中，保持顺序，禁止替换冒号
                x265_params += hdr_params
            # -------------------------------------------------------------
        return FFmpegParams('libx265', pix_fmt, profile, level, [], ['-x265-params', ':'.join(x265_params)], [])

# -------------------- FFmpeg 命令构建 --------------------
COMMON_FFMPEG_FLAGS = [
    '-metadata:s:v:0', 'handler_name=VideoHandler',
    '-metadata:s:a:0', 'handler_name=SoundHandler',
    '-metadata:s:a:0', 'language=und',
    '-metadata:s:a:0', 'title=Main Audio'
]

# -------------------- 音频码率自适应 (Apple Validator 兼容) --------------------
def get_audio_flags(audio_channels: int) -> List[str]:
    min_bitrate = 128  # kbps
    per_channel = 64   # kbps
    max_total = 512    # kbps

    calculated_bitrate = max(min_bitrate, audio_channels * per_channel)
    calculated_bitrate = min(calculated_bitrate, max_total)

    return ['-c:a', 'aac', '-b:a', f'{calculated_bitrate}k', '-ar', '48000']

# -------------------- FFmpeg 命令构建 (NVENC/CPU HDR 顺序安全) --------------------
def build_ffmpeg_command_unified(
    file_path: Path,
    out_path: Path,
    ff_params: FFmpegParams,
    audio_channels: int,
    audio_language: Optional[str] = 'eng',  # 新增参数以继承音轨语言
    extra_vparams: Optional[List[str]] = None
) -> List[str]:
    """
    构建统一 FFmpeg 命令，保证 HDR metadata 顺序安全
    """
    cmd = [
        'ffmpeg', '-hide_banner', '-y', '-i', str(file_path),
        '-map_metadata', '0',
        '-c:v', ff_params.vcodec,
        '-pix_fmt', ff_params.pix_fmt,
        '-profile:v', ff_params.profile,
        '-level:v', ff_params.level,
        '-tag:v', 'hvc1',
    ]

    # NVENC: 先插入 HDR metadata（顺序严格）
    if ff_params.hdr_metadata:
        cmd.extend(ff_params.hdr_metadata)
    elif ff_params.color_flags:
        cmd.extend(ff_params.color_flags)

    # vparams
    if extra_vparams:
        cmd.extend(extra_vparams)
    else:
        cmd.extend(ff_params.vparams)

    # 通用音视频参数：复制 COMMON 并替换 language
    common_flags = COMMON_FFMPEG_FLAGS.copy()
    # 查找 language 对并替换
    try:
        idx = common_flags.index('language=und')
        common_flags[idx] = f'language={audio_language or "eng"}'
    except ValueError:
        pass

    cmd.extend(common_flags)
    cmd.extend(get_audio_flags(audio_channels))  # ✅ 动态计算码率
    cmd.extend(['-ac', str(audio_channels)])     # ✅ 保留原声道数
    cmd.extend(['-color_range', 'tv'])
    cmd.extend(['-brand', 'mp42'])
    cmd.extend(['-movflags', '+write_colr+use_metadata_tags+faststart'])
    cmd.append(str(out_path))

    return cmd

# -------------------- NVENC 重试 --------------------
NVENC_RETRIES = [
    {'-bf': '3', '-b_ref_mode': 'middle'},
    {'-bf': '0', '-b_ref_mode': 'disabled'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0'},
    {'-bf': '0', '-b_ref_mode': 'disabled', '-temporal-aq': '0', '-spatial-aq': '0'}
]

def adjust_nvenc_params(params: List[str], attempt: int) -> List[str]:
    """
    更稳健的 NVENC 参数覆盖：
    - params: 原始参数列表（如 ['-rc','vbr','-cq','18', ...]）
    - attempt: 0 表示不修改，1..N 对应 NVENC_RETRIES 的索引
    """
    new_params = params.copy()
    if attempt <= 0:
        return new_params
    # 限制 attempt 不超过可用 retries 长度
    idx = min(attempt, len(NVENC_RETRIES)) - 1
    retry_mods = NVENC_RETRIES[idx]

    # 解析 params 为 OrderedDict（支持单 flag 情形）
    param_dict = OrderedDict()
    i = 0
    while i < len(new_params):
        key = new_params[i]
        val = None
        if i + 1 < len(new_params) and not new_params[i+1].startswith('-'):
            val = new_params[i+1]
            i += 2
        else:
            # 单 flag（没有随后的值），设为 empty string
            val = ''
            i += 1
        param_dict[key] = val

    # 应用 retry_mods（覆盖或新增）
    for k, v in retry_mods.items():
        param_dict[k] = v

    # 重建列表（恢复为 ['-key','val', ...]，忽略空值时只输出 flag）
    rebuilt = []
    for k, v in param_dict.items():
        rebuilt.append(k)
        if v is not None and v != '':
            rebuilt.append(str(v))
    return rebuilt

# -------------------- Apple Validator --------------------
def detect_validator_path() -> Optional[Path]:
    possible_paths = [
        Path('/Applications/Apple Video Tools/AppleHEVCValidator'),
        Path('/usr/local/bin/AppleHEVCValidator'),
        Path('/usr/bin/AppleHEVCValidator'),
        Path('/opt/homebrew/bin/AppleHEVCValidator'),
        Path('C:/Program Files/Apple/AppleHEVCValidator.exe')
    ]
    return next((p for p in possible_paths if p.exists()), None)

@lru_cache(maxsize=128)
def run_apple_validator(file_path: Path, refresh_cache=False) -> bool:
    if refresh_cache:
        run_apple_validator.cache_clear()
    validator = detect_validator_path()
    if not validator:
        logger.debug("Apple Validator 未安装，跳过检测。")
        return True
    with validator_lock:
        try:
            subprocess.run([str(validator), str(file_path)], check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info(f"✅ Apple Validator 通过: {file_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, "stderr", "") or ""
            stdout = getattr(e, "stdout", "") or ""
            logger.warning(f"⚠️ Apple Validator 未通过: {file_path.name} | stderr: {stderr[:2000]} stdout: {stdout[:2000]}")
            return False
        except Exception as e:
            logger.error(f"运行 Apple Validator 异常: {e}")
            return False

@lru_cache(maxsize=1)
def detect_gpu_type() -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return result.stdout.strip().lower()
    except Exception:
        return "unknown"

# -------------------- 转码主逻辑 --------------------
def convert_video(file_path: Path, out_dir: Path, debug: bool = False,
                  skip_validator: bool = False, force_cpu: bool = False, force_gpu: bool = False,
                  nvenc_hdr_mode: str = 'prefer'):
    info = probe_video(file_path)
    gpu_name = detect_gpu_type()
    out_path = out_dir / (file_path.stem + '.mp4')
    audio_channels = probe_audio_channels(file_path)
    hdr = info.hdr

    # -------------------- 决定编码器 --------------------
    use_nvenc = decide_encoder(info, force_cpu, force_gpu, nvenc_hdr_mode)
    method_guess = "NVENC" if use_nvenc else "CPU"

    log_entry = {
        "file": file_path.name,
        "status": "FAILED",
        "quality": None,
        "retries": 0,
        "method": method_guess,
        "hdr": hdr
    }

    # -------------------- 计算 CRF/CQ --------------------
    crf, cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=False)
    _, nvenc_cq, _, _, _ = calculate_dynamic_values(info, use_nvenc=True, gpu_name=gpu_name)

    ff_params = build_ffmpeg_params(info, use_nvenc, gpu_name)

    # -------------------- NVENC 编码 --------------------
    if use_nvenc:
        for attempt in range(1, len(NVENC_RETRIES) + 2):
            retry_vparams = adjust_nvenc_params(ff_params.vparams, attempt if attempt <= len(NVENC_RETRIES) else 1)
            cmd = build_ffmpeg_command_unified(
                file_path, out_path, ff_params, audio_channels,
                audio_language=info.audio_language, extra_vparams=retry_vparams
            )
            if debug:
                logger.debug(f"NVENC FFmpeg 命令 (尝试 {attempt}): {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                log_entry.update({
                    "status": "SUCCESS",
                    "quality": nvenc_cq,
                    "retries": attempt if attempt <= len(NVENC_RETRIES) else len(NVENC_RETRIES),
                    "method": "NVENC"
                })
                if not skip_validator and not run_apple_validator(out_path):
                    logger.warning("NVENC 输出未通过 Validator，回退 CPU")
                    out_path.unlink(missing_ok=True)
                    use_nvenc = False
                    break
                break
            except subprocess.CalledProcessError as e:
                stderr = getattr(e, "stderr", "") or ""
                if debug:
                    logger.debug(f"NVENC 编码失败 stderr:\n{stderr}")
                else:
                    logger.warning(f"NVENC 编码失败尝试 {attempt}: {file_path.name} | stderr: {stderr[:1000]}")
                if attempt == len(NVENC_RETRIES) + 1:
                    use_nvenc = False

    # -------------------- CPU 编码 --------------------
    if not use_nvenc:
        ff_params_cpu = build_ffmpeg_params(info, False, gpu_name)
        cmd_cpu = build_ffmpeg_command_unified(
            file_path, out_path, ff_params_cpu, audio_channels,
            audio_language=info.audio_language
        )
        if debug:
            logger.debug(f"CPU FFmpeg 命令: {' '.join(cmd_cpu)}")
        try:
            subprocess.run(cmd_cpu, check=True, capture_output=True, text=True, encoding='utf-8')
            log_entry.update({
                "status": "SUCCESS",
                "quality": crf,
                "retries": 0,
                "method": "CPU"
            })
            if not skip_validator and not run_apple_validator(out_path):
                logger.error(f"CPU 输出未通过 Validator: {file_path.name}")
        except subprocess.CalledProcessError as e:
            stderr = getattr(e, "stderr", "") or ""
            if debug:
                logger.debug(f"CPU 编码失败 stderr:\n{stderr}")
            logger.error(f"CPU 转码失败: {file_path.name}\n{stderr[:2000]}")

    return log_entry

# -------------------- 批量处理 --------------------
def batch_convert(input_dir: Path, output_dir: Path, max_workers: int = 4, **kwargs):
    files = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS]
    if not files:
        logger.warning(f"未找到可转码的视频文件于目录：{input_dir}")
        return
    output_dir.mkdir(exist_ok=True, parents=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_list = [executor.submit(convert_video, f, output_dir, **kwargs) for f in files]
        for fut in tqdm(futures_list, desc="转码"):
            try:
                results.append(fut.result())
            except Exception as e:
                idx = futures_list.index(fut)
                logger.error(f"[ERROR] {files[idx].name}: {e}")
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file", "status", "quality", "retries", "method", "hdr"])
        writer.writeheader()
        writer.writerows(results)

# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Apple HEVC 批量转码脚本 v1.6.7")
    parser.add_argument("-i", "--input", dest="input_dir", required=True)
    parser.add_argument("-o", "--output", dest="output_dir", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-validator", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--nvenc-hdr-mode", choices=['auto', 'prefer', 'disable'], default='prefer')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    input_path = Path(args.input_dir)
    sample_files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in INPUT_EXTS][:5]
    any_hdr = any(probe_video(f).hdr for f in sample_files)
    batch_convert(
        Path(args.input_dir),
        Path(args.output_dir),
        max_workers = MAX_WORKERS_HDR if any_hdr else MAX_WORKERS_SDR,
        debug=args.debug,
        skip_validator=args.skip_validator,
        force_cpu=args.force_cpu,
        force_gpu=args.force_gpu,
        nvenc_hdr_mode=args.nvenc_hdr_mode
    )
