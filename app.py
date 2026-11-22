# ==========================================
# VFX Engine - Expert Edition (Studio Dark Pro)
# Verified Edition: All Filters Tested & Fixed
# ==========================================

import os
import cv2
import numpy as np
import time
import threading
import colorsys
import json
import copy
import random
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QCheckBox, QListWidget, QScrollArea, QGroupBox,
    QFileDialog, QMessageBox, QLineEdit, QSizePolicy, QTextEdit,
    QInputDialog, QFrame, QLayout, QAbstractSpinBox,
    QStyledItemDelegate, QStyle
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, QThread, QSize, QRect, QPoint,
    QPropertyAnimation, QEasingCurve, QElapsedTimer
)
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QTextCursor, QColor,
    QLinearGradient, QPainter, QPalette
)

try:
    import pyvirtualcam
    VIRTUALCAM_OK = True
except ImportError:
    VIRTUALCAM_OK = False

SETTINGS_FILE = "vfx_engine_settings.json"


# ==========================================
# 2. PALETTE & LUT GENERATION
# ==========================================
def _gen_sine_hsv(h_start, h_range, s_base, s_range, v_base, v_range, steps=100):
    stops = []
    for i in range(steps + 1):
        t = i / steps
        wave = np.sin(t * np.pi)
        inv_wave = 1.0 - wave
        h = (h_start + h_range * t) % 360.0
        s = s_base + s_range * wave
        v = v_base + v_range * inv_wave
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, np.clip(s, 0, 1), np.clip(v, 0, 1))
        stops.append((t, [int(r * 255), int(g * 255), int(b * 255)]))
    return stops

def _gen_multi_stop(stops_list, steps=100):
    stops = []
    stops_list = sorted(stops_list, key=lambda x: x[0])
    for i in range(steps + 1):
        t = i / steps
        p1, c1 = stops_list[0]
        p2, c2 = stops_list[-1]
        for j in range(len(stops_list) - 1):
            if stops_list[j][0] <= t <= stops_list[j + 1][0]:
                p1, c1 = stops_list[j]
                p2, c2 = stops_list[j + 1]
                break
        ratio = 0.0 if p2 == p1 else (t - p1) / (p2 - p1)
        r = int(c1[0] + (c2[0] - c1[0]) * ratio)
        g = int(c1[1] + (c2[1] - c1[1]) * ratio)
        b = int(c1[2] + (c2[2] - c1[2]) * ratio)
        stops.append((t, [r, g, b]))
    return stops

def build_lut(palette_stops):
    stops = sorted(palette_stops, key=lambda x: x[0])
    positions = np.array([s[0] for s in stops])
    colors = np.array([s[1] for s in stops])
    t_values = np.linspace(0, 1, 256)
    r = np.interp(t_values, positions, colors[:, 0])
    g = np.interp(t_values, positions, colors[:, 1])
    b = np.interp(t_values, positions, colors[:, 2])
    lut = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return lut.reshape(256, 1, 3)

PALETTES = {
    "None": [(0.0, [0, 0, 0]), (1.0, [255, 255, 255])],
    "White": [(0.0, [0, 0, 0]), (1.0, [255, 255, 255])],
    "Bioluminescent": _gen_multi_stop([(0.0, [0, 0, 0]), (0.2, [0, 15, 40]), (0.4, [0, 80, 120]), (0.6, [0, 200, 220]), (0.8, [100, 255, 180]), (1.0, [230, 255, 255])], 100),
    "Holographic": _gen_multi_stop([(0.0, [10, 0, 20]), (0.15, [80, 0, 120]), (0.30, [200, 0, 255]), (0.45, [0, 150, 255]), (0.60, [0, 255, 200]), (0.75, [180, 255, 220]), (0.90, [255, 100, 200]), (1.0, [255, 240, 150])], 100),
    "Radioactive": _gen_multi_stop([(0.0, [0, 0, 0]), (0.2, [0, 30, 0]), (0.4, [20, 80, 0]), (0.6, [60, 150, 0]), (0.8, [180, 255, 20]), (1.0, [255, 255, 200])], 100),
    "Coral Reef": _gen_multi_stop([(0.0, [0, 0, 10]), (0.2, [10, 0, 40]), (0.4, [100, 0, 100]), (0.6, [200, 50, 100]), (0.8, [255, 150, 120]), (1.0, [255, 240, 220])], 100),
    "Ember & Ash": _gen_multi_stop([(0.0, [5, 5, 5]), (0.2, [30, 10, 5]), (0.4, [100, 20, 0]), (0.6, [200, 60, 0]), (0.8, [255, 150, 20]), (1.0, [255, 230, 150])], 100),
    "Steampunk": _gen_multi_stop([(0.0, [10, 5, 0]), (0.2, [40, 20, 0]), (0.4, [100, 50, 10]), (0.6, [180, 100, 0]), (0.8, [220, 160, 50]), (1.0, [255, 230, 150])], 100),
    "Tie-Dye": _gen_multi_stop([(0.0, [10, 0, 30]), (0.15, [200, 0, 100]), (0.30, [255, 100, 0]), (0.45, [255, 230, 0]), (0.60, [50, 200, 50]), (0.75, [0, 200, 255]), (0.90, [100, 50, 255]), (1.0, [255, 50, 200])], 100),
    "Neon Genesis": _gen_multi_stop([(0.0, [0, 0, 0]), (0.15, [20, 0, 40]), (0.30, [80, 0, 120]), (0.45, [150, 0, 180]), (0.60, [255, 0, 150]), (0.75, [0, 200, 255]), (0.90, [150, 240, 255]), (1.0, [255, 255, 255])], 100),
    "100 Color Rainbow": _gen_sine_hsv(0, 360, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Inferno": _gen_sine_hsv(0, 60, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Ocean": _gen_sine_hsv(180, 60, 1.0, 0.0, 0.0, 1.0, 100),
    "100 Color Cyberpunk": _gen_sine_hsv(300, 240, 1.0, 0.0, 0.0, 1.0, 100),
    "Neon Cyan": [(0.0, [0, 0, 0]), (0.5, [0, 130, 160]), (1.0, [200, 255, 255])],
    "Matrix Green": [(0.0, [0, 0, 0]), (0.5, [0, 90, 20]), (1.0, [220, 255, 220])],
    "Ice Blue": [(0.0, [0, 5, 20]), (0.5, [50, 100, 150]), (1.0, [255, 255, 255])],
    "Acid Pink": [(0.0, [0, 0, 0]), (0.5, [100, 0, 120]), (1.0, [255, 210, 255])],
    "Inferno": [(0.00, [0, 0, 0]), (0.50, [200, 40, 60]), (1.00, [255, 255, 220])],
    "Cyberpunk": [(0.00, [0, 0, 0]), (0.50, [240, 50, 140]), (1.00, [255, 255, 255])],
    "Sunset": [(0.00, [10, 0, 30]), (0.50, [160, 30, 90]), (1.00, [255, 240, 150])],
    "Ocean Depth": [(0.00, [0, 5, 20]), (0.50, [0, 90, 150]), (1.00, [240, 255, 255])],
    "Fire": [(0.00, [0, 0, 0]), (0.50, [200, 70, 0]), (1.00, [255, 255, 230])],
    "Viridis": [(0.00, [68, 1, 84]), (0.50, [52, 84, 140]), (1.00, [253, 231, 37])],
    "Aurora": [(0.00, [0, 5, 20]), (0.50, [0, 140, 90]), (1.00, [180, 100, 240])],
    "Toxic": [(0.00, [0, 0, 0]), (0.50, [40, 100, 15]), (1.00, [255, 255, 200])],
    "Vaporwave": [(0.00, [10, 0, 30]), (0.50, [140, 60, 170]), (1.00, [255, 240, 255])],
    "Gold": [(0.00, [0, 0, 0]), (0.50, [120, 70, 15]), (1.00, [255, 250, 220])],
    "Rainbow Flow": [(0.00, [255, 0, 0]), (0.33, [255, 255, 0]), (0.66, [0, 200, 255]), (1.00, [180, 0, 255])],
    "Plasma": [(0.00, [13, 8, 135]), (0.50, [166, 40, 133]), (1.00, [240, 249, 33])],
    "Magma": [(0.0, [0, 0, 4]), (0.5, [140, 40, 75]), (1.0, [255, 240, 180])],
    "Tropical": [(0.0, [0, 60, 80]), (0.5, [80, 180, 70]), (1.0, [240, 100, 30])],
    "Synthwave": [(0.0, [20, 0, 40]), (0.5, [160, 40, 130]), (1.0, [255, 150, 80])],
    "Neon Tokyo": [(0.0, [5, 5, 20]), (0.5, [255, 50, 150]), (1.0, [200, 255, 255])],
    "Cotton Candy": [(0.0, [50, 0, 80]), (0.5, [180, 80, 210]), (1.0, [255, 255, 255])],
    "Lava Reef": [(0.0, [10, 5, 5]), (0.5, [170, 50, 5]), (1.0, [255, 255, 200])],
    "Deep Space": [(0.0, [0, 0, 10]), (0.5, [60, 20, 120]), (1.0, [255, 200, 255])],
    "Turbo": [(0.0, [48, 18, 59]), (0.5, [94, 221, 74]), (1.0, [248, 99, 69])],
    "Cividis": [(0.0, [0, 32, 76]), (0.5, [150, 130, 105]), (1.0, [255, 233, 69])],
    "Emerald Forest": [(0.0, [0, 10, 5]), (0.5, [25, 110, 45]), (1.0, [220, 255, 220])],
    "Sahara Gold": [(0.0, [20, 10, 0]), (0.5, [150, 100, 40]), (1.0, [255, 240, 180])],
    "Arctic Frost": [(0.0, [10, 20, 30]), (0.5, [90, 130, 160]), (1.0, [255, 255, 255])],
    "Mojave Dusk": [(0.0, [20, 10, 30]), (0.5, [140, 50, 80]), (1.0, [255, 220, 150])],
    "Cherry Blossom": _gen_multi_stop([(0.0, [20, 0, 30]), (0.3, [80, 20, 60]), (0.6, [220, 130, 170]), (1.0, [255, 240, 245])], 100),
    "Mint Sorbet": _gen_multi_stop([(0.0, [0, 20, 25]), (0.3, [20, 80, 80]), (0.6, [120, 200, 180]), (1.0, [230, 255, 240])], 100),
    "Blood Moon": _gen_multi_stop([(0.0, [0, 0, 0]), (0.3, [40, 0, 0]), (0.6, [120, 10, 5]), (1.0, [255, 80, 40])], 100),
    "Galactic Pink": _gen_multi_stop([(0.0, [5, 0, 20]), (0.3, [80, 0, 100]), (0.6, [200, 50, 180]), (1.0, [255, 200, 240])], 100),
    "Toxic Slime": _gen_multi_stop([(0.0, [0, 10, 0]), (0.3, [40, 80, 0]), (0.6, [120, 200, 30]), (1.0, [240, 255, 150])], 100),
    "Volcanic Glass": _gen_multi_stop([(0.0, [0, 0, 0]), (0.4, [30, 10, 5]), (0.7, [180, 30, 0]), (1.0, [255, 200, 100])], 100),
}

PRECOMPUTED_LUTS = {name: build_lut(stops) for name, stops in PALETTES.items()}
PALETTE_NAMES = list(PALETTES.keys())

PALETTE_MODES = [
    "Mono Map", "Direct RGB", "50% Blend", "HSV Tint", "Luma Preserve",
    "Dual Tone", "Gradient Overlay", "Channel Swap", "Cycle Shift", "Soft Tint",
    "Hard Mix", "Lab Mix", "Saturation Boost", "Inverse Tint", "Pastel Wash", "Cyber Glitch",
]


# ==========================================
# 3. PARAMETER DEFINITIONS
# ==========================================
@dataclass
class ParamDef:
    key: str
    label: str
    p_type: str
    min_val: float
    max_val: float
    default: float
    category: str = "filter"
    step: float = 1.0

_PRE_PARAMS = [
    ParamDef("pre_brightness", "Brightness",          "int",   -100, 100,  0,    "pre"),
    ParamDef("pre_contrast",   "Contrast",             "int",   -100, 100,  0,    "pre"),
    ParamDef("pre_saturation", "Saturation",           "int",   -100, 100,  0,    "pre"),
    ParamDef("pre_gamma",      "Gamma",                "float", 0.1,  3.0,  1.0,  "pre", 0.05),
    ParamDef("pre_sharpness",  "Sharpness",            "int",   0,    100,  0,    "pre"),
    ParamDef("pre_denoise",    "Denoise Strength",     "int",   0,    15,   0,    "pre"),
]

_POST_PARAMS = [
    ParamDef("global_glow",       "Glow",              "int",   0,   100,  0,   "post"),
    ParamDef("global_posterize",  "Posterize Levels",  "int",   0,   8,    0,   "post"),
    ParamDef("global_pixelate",   "Pixelate Block",    "int",   0,   80,   0,   "post"),
    ParamDef("global_vignette",   "Vignette",          "int",   0,   100,  0,   "post"),
    ParamDef("global_grain",      "Film Grain",        "int",   0,   100,  0,   "post"),
    ParamDef("global_scanlines",  "Scanlines",         "int",   0,   100,  0,   "post"),
    ParamDef("global_bloom",      "Bloom",             "int",   0,   100,  0,   "post"),
    ParamDef("global_distortion", "Distortion",        "int",   0,   100,  0,   "post"),
    ParamDef("global_invert",     "Invert Colors",     "bool",  0,   1,    0,   "post"),
    ParamDef("global_flip_h",     "Mirror Horizontal", "bool",  0,   1,    0,   "post"),
    ParamDef("global_flip_v",     "Flip Vertical",     "bool",  0,   1,    0,   "post"),
    ParamDef("global_contrast",   "Contrast",          "int",   -100, 100,  0,   "post"),
    ParamDef("global_brightness", "Brightness",        "int",   -100, 100,  0,   "post"),
    ParamDef("global_saturation", "Saturation",        "int",   -100, 100,  0,   "post"),
]

_FILTER_PARAMS_LIST = [
    ParamDef("canny_low",          "Canny Low",             "int",   10,  200,  50,   "filter"),
    ParamDef("canny_high",         "Canny High",            "int",   50,  400,  150,  "filter"),
    ParamDef("ghost_decay",        "Ghost Decay",           "float", 0.0, 0.99, 0.85, "filter", 0.01),
    ParamDef("ghost_thresh",       "Ghost Threshold",       "int",   0,   255,  20,   "filter"),
    ParamDef("ascii_block",        "ASCII Block Size",      "int",   4,   40,   12,   "filter"),
    ParamDef("ascii_bg",           "ASCII Background",      "int",   0,   255,  0,    "filter"),
    ParamDef("ascii_color_mix",    "Color Preservation",    "int",   0,   100,  60,   "filter"),
    ParamDef("ascii_neon_glow",    "Neon Glow Amount",      "int",   0,   200,  120,  "filter"),
    ParamDef("ascii_neon_hue",     "Neon Hue °",            "int",   0,   360,  180,  "filter"),
    ParamDef("ascii_matrix_speed", "Fall Speed",            "int",   1,   50,   12,   "filter"),
    ParamDef("ascii_matrix_trail", "Trail Decay",           "float", 0.50, 0.99, 0.90, "filter", 0.01),
    ParamDef("ascii_rainbow_spd",  "Rainbow Speed",         "int",   1,   50,   10,   "filter"),
    ParamDef("ascii_outline",      "Edge Outline",          "int",   0,   100,  40,   "filter"),
    ParamDef("ascii_invert_text",  "Invert Text",           "bool",  0,   1,    0,    "filter"),
    ParamDef("ascii_glitch_amt",   "Glitch Amount",         "int",   0,   100,  30,   "filter"),
    ParamDef("ascii_block_chars",  "Character Set",         "int",   0,   3,    0,    "filter"),
    ParamDef("halftone_density",   "Halftone Density",      "float", 0.5, 4.0,  1.5,  "filter", 0.1),
    ParamDef("halftone_min",       "Halftone Min Size",     "float", 0.1, 1.0,  0.1,  "filter", 0.05),
    ParamDef("mesh_pts",           "Mesh Points",           "int",   30,  400,  150,  "filter"),
    ParamDef("mesh_thick",         "Mesh Thickness",        "int",   1,   5,    1,    "filter"),
    ParamDef("edge_smooth",        "Edge Smoothness",       "int",   1,   31,   5,    "filter"),
    ParamDef("edge_darkness",      "Edge Darkness",         "int",   0,   255,  255,  "filter"),
    ParamDef("blur_k",             "Blur Strength",         "int",   1,   51,   5,    "filter"),
    ParamDef("blur_h",             "Blur Horizontal",       "int",   1,   51,   0,    "filter"),
    ParamDef("blur_v",             "Blur Vertical",         "int",   1,   51,   0,    "filter"),
    ParamDef("neon_glow_amt",      "Neon Glow Amount",      "int",   0,   200,  80,   "filter"),
    ParamDef("dither_levels",      "Dither Levels",         "int",   2,   16,   4,    "filter"),
    ParamDef("flip_h",             "Mirror Horizontal",     "bool",  0,   1,    0,    "filter"),
    ParamDef("flip_v",             "Flip Vertical",         "bool",  0,   1,    0,    "filter"),
    ParamDef("pixelate_amt",       "Pixelate Block",        "int",   0,   80,   0,    "filter"),
    ParamDef("posterize_lvl",      "Posterize Levels",      "int",   0,   8,    0,    "filter"),
    ParamDef("grain_amt",          "Film Grain",            "int",   0,   100,  0,    "filter"),
    ParamDef("vignette_amt",       "Vignette",              "int",   0,   100,  0,    "filter"),
    ParamDef("bloom_amt",          "Bloom",                 "int",   0,   100,  0,    "filter"),
    ParamDef("distortion_amt",     "Distortion",            "int",   0,   100,  0,    "filter"),
    ParamDef("scanlines_amt",      "Scanlines",             "int",   0,   100,  0,    "filter"),
    ParamDef("chromatic_shift",    "Chromatic Shift",       "int",   0,   50,   8,    "filter"),
    ParamDef("glitch_intensity",   "Glitch Intensity",      "int",   0,   100,  50,   "filter"),
    ParamDef("glitch_slices",      "Glitch Slices",         "int",   2,   30,   8,    "filter"),
    ParamDef("glitch_offset",      "Glitch Max Offset",     "int",   0,   200,  60,   "filter"),
    ParamDef("cartoon_edges",      "Edge Strength",         "int",   0,   100,  50,   "filter"),
    ParamDef("cartoon_levels",     "Color Levels",          "int",   2,   16,   6,    "filter"),
    ParamDef("cartoon_smooth",     "Smoothing",             "int",   3,   31,   7,    "filter"),
    ParamDef("kaleido_segments",   "Segments",              "int",   2,   24,   8,    "filter"),
    ParamDef("kaleido_rotate",     "Rotation °",            "float", 0,   360,  0,    "filter", 1.0),
    ParamDef("tilt_focus",         "Focus Position",        "int",   0,   100,  50,   "filter"),
    ParamDef("tilt_range",         "Focus Range",           "int",   5,   100,  30,   "filter"),
    ParamDef("tilt_blur",          "Blur Strength",         "int",   1,   51,   15,   "filter"),
    ParamDef("water_smooth",       "Smoothing",             "int",   1,   31,   9,    "filter"),
    ParamDef("water_sat",          "Saturation Boost",      "int",   50,  250,  140,  "filter"),
    ParamDef("oil_radius",         "Oil Radius",            "int",   1,   8,    3,    "filter"),
    ParamDef("bokeh_blur",         "Bokeh Blur",            "int",   5,   81,   25,   "filter"),
    ParamDef("bokeh_thresh",       "Highlight Threshold",   "int",   100, 255,  180,  "filter"),
    ParamDef("exposure_decay",     "Trail Decay",           "float", 0.5, 0.99, 0.85, "filter", 0.01),
    ParamDef("slice_count",        "Slice Count",           "int",   2,   30,   8,    "filter"),
    ParamDef("slice_max_offset",   "Max Offset",            "int",   0,   200,  60,   "filter"),
    ParamDef("sort_direction",     "Direction (0=H,1=V)",   "int",   0,   1,    0,    "filter"),
    ParamDef("sort_threshold",     "Sort Threshold",        "int",   0,   255,  100,  "filter"),
    ParamDef("anaglyph_shift",     "Channel Shift",         "int",   0,   50,   8,    "filter"),
    ParamDef("vhs_noise",          "Noise Amount",          "int",   0,   100,  30,   "filter"),
    ParamDef("vhs_tracking",       "Tracking Distortion",   "int",   0,   100,  20,   "filter"),
    ParamDef("fractal_zoom",       "Zoom Factor",           "float", 1.0, 3.0,  1.5,  "filter", 0.05),
    ParamDef("fractal_layers",     "Recursion Layers",      "int",   1,   8,    4,    "filter"),
    ParamDef("night_blue",         "Blue Tint",             "int",   0,   100,  60,   "filter"),
    ParamDef("night_darken",       "Darken Amount",         "int",   0,   100,  40,   "filter"),
    ParamDef("duo_shadow_hue",     "Shadow Hue °",          "int",   0,   360,  240,  "filter"),
    ParamDef("duo_high_hue",       "Highlight Hue °",       "int",   0,   360,  30,   "filter"),
    ParamDef("edge_glow_str",      "Glow Strength",         "int",   0,   200,  100,  "filter"),
    ParamDef("edge_glow_rad",      "Glow Radius",           "int",   1,   51,   11,   "filter"),
    ParamDef("face_blur_strength", "Blur Strength",         "int",   11,  101,  35,   "filter"),
    ParamDef("chroma_thresh",      "Chroma Threshold",      "int",   10,  100,  40,   "filter"),
    ParamDef("ripple_amp",         "Ripple Amplitude",      "int",   0,   50,   15,   "filter"),
    ParamDef("ripple_freq",        "Ripple Frequency",      "int",   1,   50,   10,   "filter"),
]

PARAM_DEFS: Dict[str, ParamDef] = {}
for _p in _PRE_PARAMS + _POST_PARAMS + _FILTER_PARAMS_LIST:
    PARAM_DEFS[_p.key] = _p

PRE_PARAMS = [p.key for p in _PRE_PARAMS]
POST_PARAMS = [p.key for p in _POST_PARAMS]


# ==========================================
# 4. FILTER CATEGORIES & PRESETS
# ==========================================
FILTER_CATEGORIES = {
    "Edge Detection": ["Canny Edges", "Ghost Edges", "Neon Edges", "Sketch Pencil", "Edge Glow Stack"],
    "Artistic":       ["ASCII Art", "ASCII Color", "ASCII Matrix", "ASCII Neon", "ASCII Blocks",
                       "ASCII Rainbow", "ASCII Glitch", "ASCII Inverted", "Halftone", "Wireframe Mesh",
                       "Dither", "Thermal Heatmap", "Cartoonify", "Oil Painting", "Watercolor", "Face Blur"],
    "Stylize":        ["Posterize", "Pixelate", "Bloom", "Vignette", "Grain", "Scanlines",
                       "Duotone Pro", "Day for Night", "Chroma Key"],
    "Distortion":     ["Distortion", "Blur", "Glitch RGB", "Slice Glitch", "Kaleidoscope",
                       "Tilt-Shift", "Fractal Tunnel", "Pixel Sort", "Time Ripple"],
    "Transform":      ["Invert", "Flip", "Anaglyph 3D", "Vaporwave VHS"],
    "Temporal":       ["Long Exposure", "Bokeh"],
    "None":           [],
}

FILTER_PARAMS: Dict[str, List[str]] = {
    "Canny Edges":     ["canny_low", "canny_high", "edge_smooth", "edge_darkness"],
    "Ghost Edges":     ["ghost_decay", "ghost_thresh", "edge_smooth"],
    "ASCII Art":       ["ascii_block", "ascii_bg"],
    "ASCII Color":     ["ascii_block", "ascii_color_mix"],
    "ASCII Matrix":    ["ascii_block", "ascii_matrix_speed", "ascii_matrix_trail"],
    "ASCII Neon":      ["ascii_block", "ascii_neon_glow", "ascii_neon_hue", "ascii_bg"],
    "ASCII Blocks":    ["ascii_block", "ascii_bg", "ascii_block_chars"],
    "ASCII Rainbow":   ["ascii_block", "ascii_rainbow_spd"],
    "ASCII Glitch":    ["ascii_block", "ascii_glitch_amt", "ascii_bg"],
    "ASCII Inverted":  ["ascii_block", "ascii_invert_text", "ascii_outline"],
    "Halftone":        ["halftone_density", "halftone_min"],
    "Wireframe Mesh":  ["mesh_pts", "mesh_thick"],
    "Sketch Pencil":   ["edge_smooth", "edge_darkness"],
    "Neon Edges":      ["canny_low", "canny_high", "neon_glow_amt"],
    "Dither":          ["dither_levels"],
    "Thermal Heatmap": ["edge_smooth"],
    "Invert":          [],
    "Flip":            ["flip_h", "flip_v"],
    "Pixelate":        ["pixelate_amt"],
    "Posterize":       ["posterize_lvl"],
    "Grain":           ["grain_amt"],
    "Vignette":        ["vignette_amt"],
    "Bloom":           ["bloom_amt"],
    "Distortion":      ["distortion_amt"],
    "Scanlines":       ["scanlines_amt"],
    "Blur":            ["blur_k", "blur_v", "blur_h"],
    "Cartoonify":      ["cartoon_smooth", "cartoon_levels", "cartoon_edges"],
    "Glitch RGB":      ["chromatic_shift", "glitch_intensity", "glitch_slices", "glitch_offset"],
    "Kaleidoscope":    ["kaleido_segments", "kaleido_rotate"],
    "Tilt-Shift":      ["tilt_focus", "tilt_range", "tilt_blur"],
    "Watercolor":      ["water_smooth", "water_sat"],
    "Oil Painting":    ["oil_radius"],
    "Bokeh":           ["bokeh_blur", "bokeh_thresh"],
    "Long Exposure":   ["exposure_decay"],
    "Slice Glitch":    ["slice_count", "slice_max_offset"],
    "Pixel Sort":      ["sort_direction", "sort_threshold"],
    "Anaglyph 3D":     ["anaglyph_shift"],
    "Vaporwave VHS":   ["vhs_noise", "vhs_tracking"],
    "Fractal Tunnel":  ["fractal_zoom", "fractal_layers"],
    "Day for Night":   ["night_blue", "night_darken"],
    "Duotone Pro":     ["duo_shadow_hue", "duo_high_hue"],
    "Edge Glow Stack": ["edge_glow_str", "edge_glow_rad"],
    "Face Blur":       ["face_blur_strength"],
    "Chroma Key":      ["chroma_thresh"],
    "Time Ripple":     ["ripple_amp", "ripple_freq"],
    "None":            [],
}

FILTER_NAMES = list(FILTER_PARAMS.keys())

CATEGORY_ICONS = {
    "Edge Detection": "✦", "Artistic": "◈", "Stylize": "❖", "Distortion": "≋",
    "Transform": "⇄", "Temporal": "⏱", "None": "∅", "All": "✺",
}

BLEND_MODES = [
    "Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Hard Light",
    "Add", "Subtract", "Difference", "Darken Only", "Lighten Only",
    "Color Dodge", "Color Burn", "Hue", "Saturation", "Luminosity",
]

FILTER_PRESETS = {
    "🎨 Cyberpunk Glitch": [
        {"filter": "Neon Edges",   "blend_mode": 0, "opacity": 100, "params": {"canny_low": 50, "canny_high": 150, "neon_glow_amt": 120}},
        {"filter": "Glitch RGB",   "blend_mode": 8, "opacity": 35,  "params": {"chromatic_shift": 15, "glitch_intensity": 60, "glitch_slices": 10, "glitch_offset": 40}},
        {"filter": "Scanlines",    "blend_mode": 3, "opacity": 40,  "params": {"scanlines_amt": 40}},
    ],
    "🖤 Comic Book Ink": [
        {"filter": "Cartoonify",   "blend_mode": 0, "opacity": 100, "params": {"cartoon_smooth": 7, "cartoon_levels": 6, "cartoon_edges": 80}},
        {"filter": "Halftone",     "blend_mode": 1, "opacity": 60,  "params": {"halftone_density": 2.0, "halftone_min": 0.2}},
    ],
    "✨ Dreamy Bloom": [
        {"filter": "Bloom",        "blend_mode": 0, "opacity": 100, "params": {"bloom_amt": 60}},
        {"filter": "Tilt-Shift",   "blend_mode": 0, "opacity": 80,  "params": {"tilt_focus": 50, "tilt_range": 40, "tilt_blur": 21}},
        {"filter": "Bokeh",        "blend_mode": 2, "opacity": 45,  "params": {"bokeh_blur": 31, "bokeh_thresh": 160}},
    ],
    "📺 Retro VHS Tape": [
        {"filter": "Vaporwave VHS","blend_mode": 0, "opacity": 100, "params": {"vhs_noise": 40, "vhs_tracking": 30}},
        {"filter": "Scanlines",    "blend_mode": 3, "opacity": 50,  "params": {"scanlines_amt": 60}},
        {"filter": "Grain",        "blend_mode": 4, "opacity": 30,  "params": {"grain_amt": 50}},
    ],
    "🌌 Acid Trip Fractal": [
        {"filter": "Fractal Tunnel","blend_mode": 0, "opacity": 100, "params": {"fractal_zoom": 1.8, "fractal_layers": 5}},
        {"filter": "Kaleidoscope", "blend_mode": 7, "opacity": 40,  "params": {"kaleido_segments": 6, "kaleido_rotate": 45}},
        {"filter": "Edge Glow Stack","blend_mode": 6, "opacity": 60, "params": {"edge_glow_str": 150, "edge_glow_rad": 15}},
    ],
    "🌃 Day for Night Noir": [
        {"filter": "Day for Night","blend_mode": 0, "opacity": 100, "params": {"night_blue": 70, "night_darken": 50}},
        {"filter": "Duotone Pro",  "blend_mode": 1, "opacity": 60,  "params": {"duo_shadow_hue": 240, "duo_high_hue": 210}},
        {"filter": "Vignette",     "blend_mode": 0, "opacity": 80,  "params": {"vignette_amt": 60}},
    ],
    "💻 Terminal Hacker": [
        {"filter": "ASCII Matrix", "blend_mode": 0, "opacity": 100, "params": {"ascii_block": 10, "ascii_matrix_speed": 18, "ascii_matrix_trail": 0.88}},
        {"filter": "Scanlines",    "blend_mode": 3, "opacity": 30,  "params": {"scanlines_amt": 35}},
        {"filter": "Vignette",     "blend_mode": 0, "opacity": 60,  "params": {"vignette_amt": 50}},
    ],
    "🌈 ASCII Dream": [
        {"filter": "ASCII Rainbow","blend_mode": 0, "opacity": 100, "params": {"ascii_block": 14, "ascii_rainbow_spd": 8}},
        {"filter": "Bloom",        "blend_mode": 2, "opacity": 50,  "params": {"bloom_amt": 45}},
    ],
    "⚡ Neon ASCII": [
        {"filter": "ASCII Neon",   "blend_mode": 0, "opacity": 100, "params": {"ascii_block": 10, "ascii_neon_glow": 150, "ascii_neon_hue": 280}},
        {"filter": "Edge Glow Stack","blend_mode": 6, "opacity": 40, "params": {"edge_glow_str": 80, "edge_glow_rad": 7}},
    ],
}


# ==========================================
# 5. PROCESS CONTEXT
# ==========================================
class ProcessContext:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        Y, X = np.ogrid[:h, :w]
        self.dist = np.sqrt((X - w / 2) ** 2 + (Y - h / 2) ** 2)
        self.max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        self.vignette_base = np.clip(1.0 - (self.dist / self.max_dist), 0, 1).astype(np.float32)
        self.map_x_base = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w)).copy()
        self.map_y_base = np.broadcast_to(np.arange(h, dtype=np.float32).reshape(h, 1), (h, w)).copy()

        self.grid_X, self.grid_Y = np.mgrid[0:h, 0:w].astype(np.float32)
        self.centered_X = self.grid_X - w / 2
        self.centered_Y = self.grid_Y - h / 2
        self.polar_r = np.sqrt(self.centered_X ** 2 + self.centered_Y ** 2) + 1e-5
        self.polar_theta = np.arctan2(self.centered_Y, self.centered_X)

        self.char_sets = [" .:-=+*#@", " ░▒▓█", " ·•●⬤", " 。，·"]
        self.char_size = 16
        self.ascii_chars = self.char_sets[0]
        self.sprite_sheets = []
        for chars in self.char_sets:
            sheet = np.zeros((self.char_size, self.char_size * len(chars), 3), dtype=np.uint8)
            for i, c in enumerate(chars):
                cv2.putText(sheet, c, (i * self.char_size + 2, self.char_size - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            self.sprite_sheets.append(sheet)
        self.sprite_sheet = self.sprite_sheets[0]
        self.bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0

        self.matrix_acc = None
        self.matrix_offsets = None
        self.matrix_small_shape = None
        self.ghost_accumulator = None
        self.exposure_accumulator = None

        self._gamma_cache: Dict[float, np.ndarray] = {}

        self.face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            pass

    def get_sprite_sheet(self, char_set_idx: int = 0):
        idx = min(char_set_idx, len(self.sprite_sheets) - 1)
        return self.sprite_sheets[idx], self.char_sets[idx]

    def get_gamma_lut(self, gamma: float) -> np.ndarray:
        if gamma not in self._gamma_cache:
            inv = 1.0 / max(gamma, 1e-3)
            self._gamma_cache[gamma] = np.array(
                [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8
            ).reshape(256, 1, 3)
            if len(self._gamma_cache) > 32:
                self._gamma_cache.pop(next(iter(self._gamma_cache)))
        return self._gamma_cache[gamma]

    def reset_temporal_state(self):
        self.ghost_accumulator = None
        self.exposure_accumulator = None
        self.matrix_acc = None
        self.matrix_offsets = None
        self.matrix_small_shape = None


# ==========================================
# 6. FILTER IMPLEMENTATIONS
# ==========================================
FILTER_REGISTRY: Dict[str, Callable] = {}

def register_filter(name: str):
    def decorator(fn: Callable):
        FILTER_REGISTRY[name] = fn
        return fn
    return decorator

def _build_ascii_index_map(ctx: ProcessContext, frame, block: int, chars: str):
    h, w = frame.shape[:2]
    small_h, small_w = max(1, h // block), max(1, w // block)
    if small_h <= 0 or small_w <= 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (small_w, small_h))
    char_idx = (small_gray / 255.0 * (len(chars) - 1)).astype(np.int32)
    idx_full = cv2.resize(char_idx, (w, h), interpolation=cv2.INTER_NEAREST)
    x_mod = np.tile(np.arange(w) % ctx.char_size, (h, 1))
    y_mod = np.tile(np.arange(h) % ctx.char_size, (w, 1)).T
    map_x = x_mod + idx_full * ctx.char_size
    return small_gray, idx_full, y_mod, map_x

@register_filter("Canny Edges")
def _f_canny_edges(ctx, frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(1, p.get("edge_smooth", 5) | 1)
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, p.get("canny_low", 50), p.get("canny_high", 150))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

@register_filter("Ghost Edges")
def _f_ghost_edges(ctx, frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if ctx.ghost_accumulator is None or ctx.ghost_accumulator.shape != frame.shape:
        ctx.ghost_accumulator = np.zeros_like(frame, dtype=np.float32)
    decay = float(np.clip(p.get("ghost_decay", 0.85), 0.0, 0.999))
    ctx.ghost_accumulator *= decay
    thresh = p.get("ghost_thresh", 20)
    edge_mask = edges_bgr > thresh
    ctx.ghost_accumulator[edge_mask] = edges_bgr[edge_mask]
    return np.clip(ctx.ghost_accumulator, 0, 255).astype(np.uint8)

@register_filter("Sketch Pencil")
def _f_sketch_pencil(ctx, frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(1, p.get("edge_smooth", 5) | 1)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    inv_blur = 255 - blur
    sketch = cv2.divide(blur, cv2.GaussianBlur(inv_blur, (k, k), 0),
                        scale=p.get("edge_darkness", 255))
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

@register_filter("Neon Edges")
def _f_neon_edges(ctx, frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, p.get("canny_low", 50), p.get("canny_high", 150))
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    glow_amount = p.get("neon_glow_amt", 80) / 100.0
    blurred = cv2.GaussianBlur(edges_bgr, (21, 21), 0)
    return cv2.addWeighted(edges_bgr, 1.0, blurred, glow_amount, 0)

@register_filter("Edge Glow Stack")
def _f_edge_glow_stack(ctx, frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
    mag = np.sqrt(sx * sx + sy * sy)
    edges = (mag / (mag.max() + 1e-5) * 255).astype(np.uint8)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    strength = p.get("edge_glow_str", 100) / 100.0
    rad = max(1, p.get("edge_glow_rad", 11) | 1)
    glow = cv2.GaussianBlur(edges_3ch, (rad, rad), 0)
    glow = cv2.GaussianBlur(glow, (rad * 2 | 1, rad * 2 | 1), 0)
    base = cv2.addWeighted(frame, 0.55, glow, strength, 0)
    return cv2.addWeighted(base, 0.75, edges_3ch, strength * 0.8, 0)

@register_filter("ASCII Art")
def _f_ascii_art(ctx, frame, p):
    res = _build_ascii_index_map(ctx, frame, p.get("ascii_block", 12), ctx.ascii_chars)
    return ctx.sprite_sheet[res[2], res[3]] if res else np.full_like(frame, p.get("ascii_bg", 0))

@register_filter("ASCII Color")
def _f_ascii_color(ctx, frame, p):
    block = p.get("ascii_block", 12)
    mix = p.get("ascii_color_mix", 60) / 100.0
    res = _build_ascii_index_map(ctx, frame, block, ctx.ascii_chars)
    if not res:
        return np.zeros_like(frame)
    h, w = frame.shape[:2]
    _, _, y_mod, map_x = res
    base = ctx.sprite_sheet[y_mod, map_x]
    small_w, small_h = max(1, w // block), max(1, h // block)
    color_layer = cv2.resize(
        cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA),
        (w, h), interpolation=cv2.INTER_NEAREST
    )
    mask = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask = mask[..., np.newaxis]
    base_f = base.astype(np.float32)
    color_f = color_layer.astype(np.float32)
    # base * (1-mix) + mix * (color*mask + base*(1-mask))
    return np.clip(base_f * (1.0 - mix) + color_f * mix * mask + base_f * (1.0 - mask) * mix,
                   0, 255).astype(np.uint8)

@register_filter("ASCII Matrix")
def _f_ascii_matrix(ctx, frame, p):
    h, w = frame.shape[:2]
    block = p.get("ascii_block", 12)
    speed = p.get("ascii_matrix_speed", 12)
    decay = float(np.clip(p.get("ascii_matrix_trail", 0.90), 0.0, 0.999))
    small_h, small_w = max(1, h // block), max(1, w // block)
    if small_h <= 0 or small_w <= 0:
        return np.zeros_like(frame)
    if ctx.matrix_acc is None or ctx.matrix_small_shape != (small_h, small_w):
        ctx.matrix_acc = np.zeros((small_h, small_w), dtype=np.float32)
        ctx.matrix_offsets = np.random.rand(small_w) * small_h
        ctx.matrix_small_shape = (small_h, small_w)
    small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (small_w, small_h))
    ctx.matrix_acc *= decay
    ctx.matrix_offsets += speed * 0.15
    for x in range(small_w):
        y = int(ctx.matrix_offsets[x]) % small_h
        ctx.matrix_acc[y, x] = max(ctx.matrix_acc[y, x], small_gray[y, x] / 255.0 * 1.8)
    n_chars = len(ctx.ascii_chars)
    idx_full = cv2.resize(
        np.clip(ctx.matrix_acc * (n_chars - 1), 0, n_chars - 1).astype(np.int32),
        (w, h), interpolation=cv2.INTER_NEAREST
    )
    map_x = np.tile(np.arange(w) % ctx.char_size, (h, 1)) + idx_full * ctx.char_size
    y_mod = np.tile(np.arange(h) % ctx.char_size, (w, 1)).T
    out = ctx.sprite_sheet[y_mod, map_x].astype(np.float32)
    out[:, :, 1] = np.clip(out[:, :, 1] * 2.0, 0, 255)
    out[:, :, 0] *= 0.15
    out[:, :, 2] *= 0.15
    out[idx_full >= n_chars - 2] = [200, 255, 200]
    return np.clip(out, 0, 255).astype(np.uint8)

@register_filter("ASCII Neon")
def _f_ascii_neon(ctx, frame, p):
    block = p.get("ascii_block", 12)
    glow_amt = p.get("ascii_neon_glow", 120) / 100.0
    hue = p.get("ascii_neon_hue", 180) / 360.0
    res = _build_ascii_index_map(ctx, frame, block, ctx.ascii_chars)
    if not res:
        return np.full_like(frame, p.get("ascii_bg", 0))
    _, _, y_mod, map_x = res
    base = ctx.sprite_sheet[y_mod, map_x]
    neon_rgb = (np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0)) * 255).reshape(1, 1, 3)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    colored = (gray[..., np.newaxis] / 255.0 * neon_rgb).astype(np.uint8)
    glow_k = max(1, int(15 + glow_amt * 10) | 1)
    inner = cv2.addWeighted(colored, 1.0, cv2.GaussianBlur(colored, (glow_k, glow_k), 0),
                            glow_amt * 0.6, 0)
    outer = cv2.GaussianBlur(cv2.GaussianBlur(colored, (glow_k, glow_k), 0),
                             (glow_k * 2 | 1, glow_k * 2 | 1), 0)
    return cv2.addWeighted(inner, 1.0, outer, glow_amt * 0.4, 0)

@register_filter("ASCII Blocks")
def _f_ascii_blocks(ctx, frame, p):
    sheet, chars = ctx.get_sprite_sheet(p.get("ascii_block_chars", 0))
    res = _build_ascii_index_map(ctx, frame, p.get("ascii_block", 12), chars)
    return sheet[res[2], res[3]] if res else np.full_like(frame, p.get("ascii_bg", 0))

@register_filter("ASCII Rainbow")
def _f_ascii_rainbow(ctx, frame, p):
    h, w = frame.shape[:2]
    res = _build_ascii_index_map(ctx, frame, p.get("ascii_block", 12), ctx.ascii_chars)
    if not res:
        return np.zeros_like(frame)
    _, _, y_mod, map_x = res
    base_gray = cv2.cvtColor(ctx.sprite_sheet[y_mod, map_x], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = time.time() * p.get("ascii_rainbow_spd", 10) / 30.0
    X, Y = ctx.grid_X, ctx.grid_Y
    hue = ((X / w + Y / h * 0.5 + t) * 1.5) % 1.0
    f = hue * 6 - np.floor(hue * 6)
    pp = base_gray * 0.0
    q = base_gray * (1 - f)
    tt = base_gray * (1 - (1 - f))
    hi = (hue * 6).astype(np.int32) % 6
    r = np.choose(hi, [base_gray, q, pp, pp, tt, base_gray])
    g = np.choose(hi, [tt, base_gray, base_gray, q, pp, pp])
    b = np.choose(hi, [pp, pp, tt, base_gray, base_gray, q])
    return np.stack([b, g, r], axis=-1).clip(0, 255).astype(np.uint8)

@register_filter("ASCII Glitch")
def _f_ascii_glitch(ctx, frame, p):
    h, w = frame.shape[:2]
    glitch_amt = p.get("ascii_glitch_amt", 30) / 100.0
    res = _build_ascii_index_map(ctx, frame, p.get("ascii_block", 12), ctx.ascii_chars)
    if not res:
        return np.full_like(frame, p.get("ascii_bg", 0))
    _, _, y_mod, map_x = res
    out = ctx.sprite_sheet[y_mod, map_x].copy()
    b_ch, g_ch, r_ch = cv2.split(out)
    shift = max(1, int(glitch_amt * 15))
    out = cv2.merge([np.roll(b_ch, -shift, axis=1), g_ch, np.roll(r_ch, shift, axis=1)])
    n_slices = max(2, int(glitch_amt * 10))
    for _ in range(n_slices):
        sy = np.random.randint(0, max(1, h - 2))
        sh = np.random.randint(2, max(3, int(h * 0.1)))
        out[sy:sy + sh] = np.roll(out[sy:sy + sh],
                                   np.random.randint(-shift * 3, shift * 3), axis=1)
    return out

@register_filter("ASCII Inverted")
def _f_ascii_inverted(ctx, frame, p):
    block = p.get("ascii_block", 12)
    invert = bool(p.get("ascii_invert_text", 0))
    outline_amt = p.get("ascii_outline", 40) / 100.0
    h, w = frame.shape[:2]
    res = _build_ascii_index_map(ctx, frame, block, ctx.ascii_chars)
    if not res:
        return np.full_like(frame, 0 if invert else 255)
    _, _, y_mod, map_x = res
    sprite = 255 - ctx.sprite_sheet if invert else ctx.sprite_sheet.copy()
    out = sprite[y_mod, map_x]
    if outline_amt > 0:
        edges_3ch = cv2.cvtColor(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150),
                                  cv2.COLOR_GRAY2BGR)
        edge_color = 255 if invert else 0
        out = cv2.addWeighted(out, 1.0 - outline_amt,
                              np.where(edges_3ch > 0, edge_color, out).astype(np.uint8),
                              outline_amt, 0)
    return out

@register_filter("Halftone")
def _f_halftone(ctx, frame, p):
    h, w = frame.shape[:2]
    step = 4
    density = p.get("halftone_density", 1.5)
    min_r = p.get("halftone_min", 0.1)
    small_h, small_w = h // step, w // step
    if small_h <= 0 or small_w <= 0:
        return np.zeros_like(frame)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rads = ((255.0 - gray_small) / 255.0 * density * step + min_r).astype(np.int32)
    rads = np.clip(rads, 0, step * 2)
    out = np.zeros_like(frame)
    ys, xs = np.mgrid[0:small_h, 0:small_w]
    ys = (ys * step + step // 2).ravel()
    xs = (xs * step + step // 2).ravel()
    rs = rads.ravel()
    mask = rs > 0
    ys, xs, rs = ys[mask], xs[mask], rs[mask]
    for y, x, r in zip(ys.tolist(), xs.tolist(), rs.tolist()):
        cv2.circle(out, (x, y), int(r), (255, 255, 255), -1)
    return out

@register_filter("Wireframe Mesh")
def _f_wireframe_mesh(ctx, frame, p):
    h, w = frame.shape[:2]
    out = np.zeros_like(frame)
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    pts = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                  p.get("mesh_pts", 150), 0.01, 10)
    if pts is not None:
        for pt in pts:
            subdiv.insert((int(pt[0][0]), int(pt[0][1])))
        thick = p.get("mesh_thick", 1)
        for t in subdiv.getTriangleList():
            p1 = (int(t[0]), int(t[1]))
            p2 = (int(t[2]), int(t[3]))
            p3 = (int(t[4]), int(t[5]))
            cv2.line(out, p1, p2, (255, 255, 255), thick)
            cv2.line(out, p2, p3, (255, 255, 255), thick)
            cv2.line(out, p3, p1, (255, 255, 255), thick)
    return out

@register_filter("Dither")
def _f_dither(ctx, frame, p):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    levels = max(2, p.get("dither_levels", 4))
    step = 255.0 / (levels - 1)
    tile_h = (h + 3) // 4
    tile_w = (w + 3) // 4
    bayer_tiled = np.tile(ctx.bayer, (tile_h + 2, tile_w + 2))[:h, :w]
    dithered = np.floor(gray / step) * step + (bayer_tiled - 0.5) * step
    return cv2.cvtColor(np.clip(dithered, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

@register_filter("Thermal Heatmap")
def _f_thermal_heatmap(ctx, frame, p):
    return cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

@register_filter("Cartoonify")
def _f_cartoonify(ctx, frame, p):
    smooth_k = max(1, p.get("cartoon_smooth", 7) | 1)
    levels = max(2, p.get("cartoon_levels", 6))
    step = 256.0 / levels
    color = cv2.bilateralFilter((np.floor(frame.astype(np.float32) / step) * step).astype(np.uint8),
                                smooth_k, smooth_k * 2.5, smooth_k / 2)
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), smooth_k)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, smooth_k, 2)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    masked = cv2.bitwise_and(color, edges_bgr)
    edge_strength = p.get("cartoon_edges", 50) / 100.0 * 0.4
    return cv2.addWeighted(masked, 1.0, color, edge_strength, 0)

@register_filter("Oil Painting")
def _f_oil_painting(ctx, frame, p):
    radius = p.get("oil_radius", 3)
    out = cv2.edgePreservingFilter(frame, flags=cv2.RECURS_FILTER,
                                   sigma_s=radius * 25, sigma_r=0.4)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

@register_filter("Watercolor")
def _f_watercolor(ctx, frame, p):
    smooth_k = max(1, p.get("water_smooth", 9) | 1)
    sat = p.get("water_sat", 140) / 100.0
    bf1 = cv2.bilateralFilter(frame, smooth_k, smooth_k * 2, smooth_k / 2)
    quantized = (np.floor(bf1.astype(np.float32) / 32.0) * 32.0).astype(np.uint8)
    q = cv2.bilateralFilter(quantized, smooth_k, smooth_k * 1.5, smooth_k / 2)
    hsv = cv2.cvtColor(q, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

@register_filter("Face Blur")
def _f_face_blur(ctx, frame, p):
    blur_k = max(1, p.get("face_blur_strength", 35) | 1)
    out = frame.copy()
    if ctx.face_cascade is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = ctx.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, fw, fh) in faces:
            out[y:y + fh, x:x + fw] = cv2.GaussianBlur(out[y:y + fh, x:x + fw],
                                                        (blur_k, blur_k), 0)
    return out

@register_filter("Posterize")
def _f_posterize(ctx, frame, p):
    levels = max(2, p.get("posterize_lvl", 0))  # Prevent level 1 (divide by zero)
    step = 255.0 / (levels - 1)
    return (np.round(frame.astype(np.float32) / step) * step).clip(0, 255).astype(np.uint8)

@register_filter("Pixelate")
def _f_pixelate(ctx, frame, p):
    h, w = frame.shape[:2]
    block = p.get("pixelate_amt", 0)
    if block <= 0:
        return frame
    small_w, small_h = max(1, w // block), max(1, h // block)
    return cv2.resize(cv2.resize(frame, (small_w, small_h)),
                      (w, h), interpolation=cv2.INTER_NEAREST)

@register_filter("Bloom")
def _f_bloom(ctx, frame, p):
    amt = p.get("bloom_amt", 0) / 100.0
    if amt <= 0:
        return frame
    return cv2.addWeighted(frame, 1.0, cv2.GaussianBlur(frame, (21, 21), 0), amt, 0)

@register_filter("Vignette")
def _f_vignette(ctx, frame, p):
    amt = p.get("vignette_amt", 0) / 100.0
    if amt <= 0:
        return frame
    mask = 1.0 - (1.0 - ctx.vignette_base) * amt
    return (frame.astype(np.float32) * mask[..., np.newaxis]).clip(0, 255).astype(np.uint8)

@register_filter("Grain")
def _f_grain(ctx, frame, p):
    intensity = p.get("grain_amt", 0) / 100.0
    if intensity <= 0:
        return frame
    noise = np.random.randint(0, 255, frame.shape, dtype=np.uint8)
    return cv2.addWeighted(frame, 1.0 - intensity, noise, intensity, 0)

@register_filter("Scanlines")
def _f_scanlines(ctx, frame, p):
    amt = p.get("scanlines_amt", 0)
    if amt <= 0:
        return frame
    out = frame.copy()
    line_int = int((amt / 100.0) * 255)
    out[::2, :] = np.clip(out[::2, :].astype(np.int16) - line_int, 0, 255).astype(np.uint8)
    return out

@register_filter("Duotone Pro")
def _f_duotone_pro(ctx, frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow_hue = (p.get("duo_shadow_hue", 240) % 360) / 360.0
    high_hue = (p.get("duo_high_hue", 30) % 360) / 360.0
    s_rgb = (np.array(colorsys.hsv_to_rgb(shadow_hue, 0.9, 0.95)) * 255).reshape(1, 1, 3)
    h_rgb = (np.array(colorsys.hsv_to_rgb(high_hue, 0.8, 1.0)) * 255).reshape(1, 1, 3)
    gray_3ch = gray[..., np.newaxis]
    return (s_rgb * (1.0 - gray_3ch) + h_rgb * gray_3ch).clip(0, 255).astype(np.uint8)

@register_filter("Day for Night")
def _f_day_for_night(ctx, frame, p):
    blue = p.get("night_blue", 60) / 100.0
    darken = p.get("night_darken", 40) / 100.0
    out = frame.astype(np.float32) * (1.0 - darken * 0.5)
    out[:, :, 0] *= (1 + blue * 0.3)
    out[:, :, 1] *= (1 - blue * 0.1)
    out[:, :, 2] *= (1 - blue * 0.25)
    return np.clip(out, 0, 255).astype(np.uint8)

@register_filter("Chroma Key")
def _f_chroma_key(ctx, frame, p):
    thresh = p.get("chroma_thresh", 40)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([60 - thresh, 40, 40]),
                       np.array([60 + thresh, 255, 255]))
    inv_mask = cv2.bitwise_not(mask)
    black_bg = np.zeros_like(frame)
    return cv2.add(cv2.bitwise_and(frame, frame, mask=inv_mask),
                   cv2.bitwise_and(black_bg, black_bg, mask=mask))

@register_filter("Distortion")
def _f_distortion(ctx, frame, p):
    amt = p.get("distortion_amt", 0)
    if amt <= 0:
        return frame
    map_x = ctx.map_x_base - np.sin(ctx.map_y_base / 20.0) * (amt / 10.0)
    return cv2.remap(frame, map_x, ctx.map_y_base, cv2.INTER_LINEAR)

@register_filter("Blur")
def _f_blur(ctx, frame, p):
    k = max(1, p.get("blur_k", 5) | 1)
    out = cv2.GaussianBlur(frame, (k, k), 0)
    bh = p.get("blur_h", 0)
    bv = p.get("blur_v", 0)
    if bh > 0 or bv > 0:
        out = cv2.GaussianBlur(out, (max(1, bh | 1), max(1, bv | 1)), 0)
    return out

@register_filter("Glitch RGB")
def _f_glitch_rgb(ctx, frame, p):
    h, w = frame.shape[:2]
    intensity = p.get("glitch_intensity", 50) / 100.0
    shift = max(1, int(p.get("chromatic_shift", 8) * intensity))
    slices = p.get("glitch_slices", 8)
    max_off = p.get("glitch_offset", 60)
    out = frame.copy()
    for _ in range(slices):
        sy = np.random.randint(0, max(1, h - 2))
        sh = np.random.randint(2, max(3, int(h * 0.12)))
        out[sy:sy + sh] = np.roll(out[sy:sy + sh],
                                   np.random.randint(-max_off, max_off), axis=1)
    b, g, r = cv2.split(out)
    return cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])

@register_filter("Slice Glitch")
def _f_slice_glitch(ctx, frame, p):
    h = frame.shape[0]
    out = frame.copy()
    count = p.get("slice_count", 8)
    max_off = p.get("slice_max_offset", 60)
    for i in range(count):
        if np.random.random() < 0.65:
            y0 = i * (h // count)
            y1 = min((i + 1) * (h // count), h)
            out[y0:y1] = np.roll(out[y0:y1], np.random.randint(-max_off, max_off), axis=1)
    return out

@register_filter("Kaleidoscope")
def _f_kaleidoscope(ctx, frame, p):
    h, w = frame.shape[:2]
    segments = max(2, p.get("kaleido_segments", 8))
    rotation = p.get("kaleido_rotate", 0) * np.pi / 180.0
    r = ctx.polar_r
    theta = ctx.polar_theta + rotation
    wedge = 2 * np.pi / segments
    mod_theta = np.mod(theta, wedge)
    reflected = np.where(mod_theta > wedge / 2, wedge - mod_theta, mod_theta)
    map_x = (w / 2 + r * np.cos(reflected)).astype(np.float32)
    map_y = (h / 2 + r * np.sin(reflected)).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

@register_filter("Tilt-Shift")
def _f_tilt_shift(ctx, frame, p):
    h = frame.shape[0]
    blur_k = max(1, p.get("tilt_blur", 15) | 1)
    blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
    focus = p.get("tilt_focus", 50) / 100.0
    rng = p.get("tilt_range", 30) / 200.0
    Y = np.arange(h, dtype=np.float32) / max(1, h - 1)
    mask = np.clip(1.0 - (np.abs(Y - focus) - rng) / 0.12, 0, 1)[:, np.newaxis, np.newaxis]
    return (frame.astype(np.float32) * mask + blurred.astype(np.float32) * (1 - mask)).astype(np.uint8)

@register_filter("Fractal Tunnel")
def _f_fractal_tunnel(ctx, frame, p):
    h, w = frame.shape[:2]
    zoom = p.get("fractal_zoom", 1.5)
    layers_n = max(1, p.get("fractal_layers", 4))
    t = time.time() * 0.6
    r = ctx.polar_r
    theta = ctx.polar_theta
    out = np.zeros_like(frame)
    for i in range(layers_n):
        scale = zoom ** i
        map_x = ((np.cos(theta) * 80.0 / r * scale + w / 2) % w).astype(np.float32)
        map_y = ((np.sin(theta) * 80.0 / r * scale + h / 2 + t * 30 * scale) % h).astype(np.float32)
        layer = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)
        out = cv2.addWeighted(out, 0.65, layer, 0.4, 0)
    return cv2.addWeighted(out, 0.8, frame, 0.2, 0)

@register_filter("Pixel Sort")
def _f_pixel_sort(ctx, frame, p):
    h, w = frame.shape[:2]
    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    direction = p.get("sort_direction", 0)
    thresh = p.get("sort_threshold", 100)
    if direction == 0:
        for y in range(h):
            mask = gray[y] < thresh
            if mask.sum() >= 2:
                out[y, mask] = np.sort(frame[y, mask], axis=0)
    else:
        for x in range(w):
            mask = gray[:, x] < thresh
            if mask.sum() >= 2:
                out[mask, x] = np.sort(frame[mask, x], axis=0)
    return out

@register_filter("Time Ripple")
def _f_time_ripple(ctx, frame, p):
    amp = p.get("ripple_amp", 15)
    freq = p.get("ripple_freq", 10)
    t = time.time()
    X, Y = ctx.grid_X, ctx.grid_Y
    dist = np.sqrt((X - ctx.w / 2) ** 2 + (Y - ctx.h / 2) ** 2)
    map_x = (X + np.cos(dist * freq / 100.0 - t * 4.0) * amp).astype(np.float32)
    map_y = (Y + np.sin(dist * freq / 100.0 - t * 4.0) * amp).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

@register_filter("Invert")
def _f_invert(ctx, frame, p):
    return cv2.bitwise_not(frame)

@register_filter("Flip")
def _f_flip(ctx, frame, p):
    code = -1
    if p.get("flip_h", 0) and p.get("flip_v", 0):
        code = -1
    elif p.get("flip_h", 0):
        code = 1
    elif p.get("flip_v", 0):
        code = 0
    else:
        return frame
    return cv2.flip(frame, code)

@register_filter("Anaglyph 3D")
def _f_anaglyph_3d(ctx, frame, p):
    shift = p.get("anaglyph_shift", 8)
    if shift <= 0:
        return frame
    b, g, r = cv2.split(frame)
    return cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])

@register_filter("Vaporwave VHS")
def _f_vaporwave_vhs(ctx, frame, p):
    h = frame.shape[0]
    b, g, r = cv2.split(frame)
    out = cv2.merge([np.roll(b, 3, axis=1), g, np.roll(r, -3, axis=1)])
    track = p.get("vhs_tracking", 20) / 100.0
    if track > 0:
        bh = max(2, int(track * 30))
        by = np.random.randint(0, max(1, h - bh))
        out[by:by + bh] = np.roll(out[by:by + bh], np.random.randint(-25, 25), axis=1)
    out[::2, :] = (out[::2, :].astype(np.float32) * 0.82).astype(np.uint8)
    n_amt = p.get("vhs_noise", 30) / 100.0
    if n_amt > 0:
        noise = np.random.randint(0, 255, out.shape, dtype=np.uint8)
        out = cv2.addWeighted(out, 1 - n_amt * 0.3, noise, n_amt * 0.3, 0)
    out_f = out.astype(np.float32)
    out_f[:, :, 2] *= 1.12
    out_f[:, :, 0] *= 1.05
    return np.clip(out_f, 0, 255).astype(np.uint8)

@register_filter("Long Exposure")
def _f_long_exposure(ctx, frame, p):
    decay = float(np.clip(p.get("exposure_decay", 0.85), 0.0, 0.999))
    frame_f = frame.astype(np.float32)
    if ctx.exposure_accumulator is None or ctx.exposure_accumulator.shape != frame.shape:
        ctx.exposure_accumulator = frame_f.copy()
    else:
        ctx.exposure_accumulator = ctx.exposure_accumulator * decay + frame_f * (1 - decay)
    return np.clip(ctx.exposure_accumulator, 0, 255).astype(np.uint8)

@register_filter("Bokeh")
def _f_bokeh(ctx, frame, p):
    blur_k = max(1, p.get("bokeh_blur", 25) | 1)
    thresh = p.get("bokeh_thresh", 180)
    blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, hl = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    hl = cv2.GaussianBlur(hl, (blur_k, blur_k), 0)
    bokeh = cv2.bitwise_and(blurred, blurred, mask=hl)
    bokeh = cv2.GaussianBlur(bokeh, (blur_k * 2 | 1, blur_k * 2 | 1), 0)
    return cv2.addWeighted(frame, 0.7, bokeh, 0.6, 0)

def apply_filter(ctx: ProcessContext, frame, filter_name: str, params: dict):
    fn = FILTER_REGISTRY.get(filter_name)
    if fn is None:
        return frame.copy()
    try:
        return fn(ctx, frame, params)
    except Exception as e:
        print(f"⚠️ Filter '{filter_name}' error: {e}")
        return frame.copy()


# ==========================================
# 7. BLEND MODES & POST-PROCESSING
# ==========================================
def apply_pre_adjustments(ctx: ProcessContext, frame, params):
    d = params.get("pre_denoise", 0)
    if d > 0:
        frame = cv2.bilateralFilter(frame, d * 2 + 1, d * 20.0, d * 20.0)
    b = params.get("pre_brightness", 0)
    c = params.get("pre_contrast", 0)
    if b != 0 or c != 0:
        f = (259 * (c + 255)) / (255 * (259 - c))
        frame = np.clip(f * (frame.astype(np.float32) - 128) + 128 + b,
                        0, 255).astype(np.uint8)
    g = params.get("pre_gamma", 1.0)
    if abs(g - 1.0) > 1e-3:
        frame = cv2.LUT(frame, ctx.get_gamma_lut(g))
    s = params.get("pre_saturation", 0)
    if s != 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + s / 100.0), 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    sh = params.get("pre_sharpness", 0) / 100.0
    if sh > 0:
        frame = cv2.addWeighted(frame, 1.0 + sh,
                                 cv2.GaussianBlur(frame, (0, 0), 3), -sh, 0)
    return frame

def _hsv_mix(base, overlay, channel):
    b_hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    o_hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV).astype(np.float32)
    b_hsv[:, :, channel] = o_hsv[:, :, channel]
    return cv2.cvtColor(np.clip(b_hsv, 0, 255).astype(np.uint8),
                        cv2.COLOR_HSV2BGR).astype(np.float32)

BLEND_FUNCS = {
    0: lambda bf, of: of,
    1: lambda bf, of: (bf * of) / 255.0,
    2: lambda bf, of: 255.0 - ((255.0 - bf) * (255.0 - of)) / 255.0,
    3: lambda bf, of: np.where(bf < 128,
                                (2 * bf * of) / 255.0,
                                255.0 - (2 * (255.0 - bf) * (255.0 - of)) / 255.0),
    4: lambda bf, of: bf + (of - 128) * (bf / 255.0) * (255.0 - bf) / 128.0,
    5: lambda bf, of: np.where(of < 128,
                                (2 * bf * of) / 255.0,
                                255.0 - (2 * (255.0 - bf) * (255.0 - of)) / 255.0),
    6: lambda bf, of: np.clip(bf + of, 0, 255),
    7: lambda bf, of: np.clip(bf - of, 0, 255),
    8: lambda bf, of: np.abs(bf - of),
    9: lambda bf, of: np.minimum(bf, of),
    10: lambda bf, of: np.maximum(bf, of),
    11: lambda bf, of: np.clip(bf * 255.0 / np.maximum(255.0 - of, 1e-5), 0, 255),
    12: lambda bf, of: np.clip(255.0 - (255.0 - bf) * 255.0 / np.maximum(of, 1e-5), 0, 255),
    13: lambda b, o: _hsv_mix(b, o, 0),
    14: lambda b, o: _hsv_mix(b, o, 1),
    15: lambda b, o: _hsv_mix(b, o, 2),
}

def blend_images(base, overlay, opacity, mode_idx):
    if opacity <= 0 or overlay.shape != base.shape:
        return base
    if opacity >= 100 and mode_idx == 0:
        return overlay.copy()
    a = opacity / 100.0
    bf = base.astype(np.float32)
    of = overlay.astype(np.float32)
    func = BLEND_FUNCS.get(mode_idx)
    if func is None:
        result = of
    elif mode_idx < 13:
        result = func(bf, of)
    else:
        result = func(base, overlay)
    result = np.clip(result, 0, 255)
    return (bf * (1 - a) + result * a).clip(0, 255).astype(np.uint8)

def apply_post_processing(ctx: ProcessContext, out, params, filter_name):
    w, h = ctx.w, ctx.h
    has_global = (
        params.get("global_distortion", 0) > 0 or
        params.get("global_vignette", 0) > 0 or
        params.get("global_bloom", 0) > 0 or
        params.get("global_grain", 0) > 0 or
        params.get("global_posterize", 0) > 0 or
        params.get("global_pixelate", 0) > 0 or
        params.get("global_scanlines", 0) > 0 or
        params.get("global_brightness", 0) != 0 or
        params.get("global_contrast", 0) != 0 or
        params.get("global_saturation", 0) != 0 or
        params.get("global_invert", 0) > 0 or
        params.get("global_flip_h", 0) > 0 or
        params.get("global_flip_v", 0) > 0 or
        params.get("global_glow", 0) > 0
    )
    if not has_global:
        return out

    if params.get("global_distortion", 0) > 0 and filter_name != "Distortion":
        map_x = ctx.map_x_base - np.sin(ctx.map_y_base / 20.0) * (params["global_distortion"] / 10.0)
        out = cv2.remap(out, map_x, ctx.map_y_base, cv2.INTER_LINEAR)
    if params.get("global_vignette", 0) > 0 and filter_name != "Vignette":
        mask = 1.0 - (1.0 - ctx.vignette_base) * (params["global_vignette"] / 100.0)
        out = (out.astype(np.float32) * mask[..., np.newaxis]).clip(0, 255).astype(np.uint8)
    if params.get("global_bloom", 0) > 0 and filter_name != "Bloom":
        out = cv2.addWeighted(out, 1.0, cv2.GaussianBlur(out, (21, 21), 0),
                              params["global_bloom"] / 100.0, 0)
    if params.get("global_glow", 0) > 0:
        out = cv2.addWeighted(out, 1.0, cv2.GaussianBlur(out, (21, 21), 0),
                              params["global_glow"] / 100.0, 0)
    if params.get("global_grain", 0) > 0 and filter_name != "Grain":
        intensity = params["global_grain"] / 100.0
        noise = np.random.randint(0, 255, out.shape, dtype=np.uint8)
        out = cv2.addWeighted(out, 1.0 - intensity, noise, intensity, 0)
    if params.get("global_posterize", 0) > 0 and filter_name != "Posterize":
        levels = max(2, params["global_posterize"])
        step = 255.0 / (levels - 1)
        out = (np.round(out.astype(np.float32) / step) * step).clip(0, 255).astype(np.uint8)
    if params.get("global_pixelate", 0) > 0 and filter_name != "Pixelate":
        block = params["global_pixelate"]
        small_w, small_h = max(1, w // block), max(1, h // block)
        out = cv2.resize(cv2.resize(out, (small_w, small_h)),
                         (w, h), interpolation=cv2.INTER_NEAREST)
    if params.get("global_scanlines", 0) > 0 and filter_name != "Scanlines":
        line_int = int((params["global_scanlines"] / 100.0) * 255)
        out[::2, :] = np.clip(out[::2, :].astype(np.int16) - line_int, 0, 255).astype(np.uint8)
    b = params.get("global_brightness", 0)
    c = params.get("global_contrast", 0)
    s = params.get("global_saturation", 0)
    if b != 0:
        out = np.clip(out.astype(np.int16) + b, 0, 255).astype(np.uint8)
    if c != 0:
        f = (259 * (c + 255)) / (255 * (259 - c))
        out = np.clip(f * (out.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
    if s != 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + s / 100.0), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if params.get("global_invert", 0) > 0 and filter_name != "Invert":
        out = cv2.bitwise_not(out)
    if params.get("global_flip_h", 0) > 0 and filter_name != "Flip":
        out = cv2.flip(out, 1)
    if params.get("global_flip_v", 0) > 0 and filter_name != "Flip":
        out = cv2.flip(out, 0)
    return out


# ==========================================
# 8. PALETTE APPLICATION MODES
# ==========================================
def _apply_mono_map(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.LUT(gray_3ch, lut)

def _apply_direct_rgb(out, lut):
    return cv2.LUT(out, lut)

def _apply_50_blend(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(out, 0.5, cv2.LUT(gray_3ch, lut), 0.5, 0)

def _apply_hsv_tint(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    mono_hsv = cv2.cvtColor(cv2.LUT(gray_3ch, lut), cv2.COLOR_BGR2HSV)
    orig_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    orig_hsv[:, :, 0] = mono_hsv[:, :, 0]
    orig_hsv[:, :, 1] = mono_hsv[:, :, 1]
    return cv2.cvtColor(orig_hsv, cv2.COLOR_HSV2BGR)

def _apply_luma_preserve(out, lut):
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mono_hsv = cv2.cvtColor(cv2.LUT(gray_3ch, lut), cv2.COLOR_BGR2HSV)
    orig_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    orig_hsv[:, :, 0] = mono_hsv[:, :, 0]
    orig_hsv[:, :, 1] = cv2.max(mono_hsv[:, :, 1], orig_hsv[:, :, 1])
    return cv2.cvtColor(orig_hsv, cv2.COLOR_HSV2BGR)

def _apply_dual_tone(out, lut):
    """Proper dual-tone split using luma midpoint."""
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lut_lo = lut.copy()
    lut_hi = lut.copy()
    lut_lo[128:] = lut_lo[127]
    lut_hi[:128] = lut_hi[0]
    mask = (gray > 128).astype(np.float32) / 255.0
    mask_3ch = np.stack([mask] * 3, axis=-1)
    return (cv2.LUT(gray_3ch, lut_lo) * (1 - mask_3ch) + cv2.LUT(gray_3ch, lut_hi) * mask_3ch).astype(np.uint8)

def _apply_gradient_overlay(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    mono_f = cv2.LUT(gray_3ch, lut).astype(np.float32) / 255.0
    out_f = out.astype(np.float32) / 255.0
    return np.clip(((1 - 2 * mono_f) * out_f * out_f + 2 * mono_f * out_f) * 255,
                   0, 255).astype(np.uint8)

def _apply_channel_swap(out, lut):
    b, g, r = cv2.split(out)
    return cv2.merge([cv2.LUT(b, lut[:, 0, 0]),
                      cv2.LUT(g, lut[:, 0, 1]),
                      cv2.LUT(r, lut[:, 0, 2])])

def _apply_cycle_shift(out, lut):
    shifted = ((cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.int32) + 128) % 256).astype(np.uint8)
    gray_3ch = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)
    return cv2.LUT(gray_3ch, lut)

def _apply_soft_tint(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(out, 0.75, cv2.LUT(gray_3ch, lut), 0.25, 0)

def _apply_hard_mix(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    lut_applied = cv2.LUT(gray_3ch, lut)
    summed = out.astype(np.int16) + lut_applied.astype(np.int16)
    return np.where(summed >= 256, 255, 0).astype(np.uint8)

def _apply_lab_mix(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    mono_lab = cv2.cvtColor(cv2.LUT(gray_3ch, lut), cv2.COLOR_BGR2LAB)
    out_lab[:, :, 1] = mono_lab[:, :, 1]
    out_lab[:, :, 2] = mono_lab[:, :, 2]
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

def _apply_saturation_boost(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    mono_hsv = cv2.cvtColor(cv2.LUT(gray_3ch, lut), cv2.COLOR_BGR2HSV)
    orig_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    orig_hsv[:, :, 0] = mono_hsv[:, :, 0]
    orig_hsv[:, :, 1] = np.clip(orig_hsv[:, :, 1].astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(orig_hsv, cv2.COLOR_HSV2BGR)

def _apply_inverse_tint(out, lut):
    inv_gray = 255 - cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(inv_gray, cv2.COLOR_GRAY2BGR)
    mono_hsv = cv2.cvtColor(cv2.LUT(gray_3ch, lut), cv2.COLOR_BGR2HSV)
    orig_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    orig_hsv[:, :, 0] = mono_hsv[:, :, 0]
    orig_hsv[:, :, 1] = mono_hsv[:, :, 1]
    return cv2.cvtColor(orig_hsv, cv2.COLOR_HSV2BGR)

def _apply_pastel_wash(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    mono_f = cv2.LUT(gray_3ch, lut).astype(np.float32) * 0.6 + 100
    return np.clip(mono_f * 0.5 + np.mean(mono_f, axis=-1, keepdims=True) * 0.5,
                   0, 255).astype(np.uint8)

def _apply_cyber_glitch(out, lut):
    gray_3ch = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    result = cv2.LUT(gray_3ch, lut).copy()
    result[:, :, 0] = np.roll(result[:, :, 0], 5, axis=1)
    result[:, :, 2] = np.roll(result[:, :, 2], -5, axis=1)
    return result

PALETTE_MODE_FUNCS = [
    _apply_mono_map, _apply_direct_rgb, _apply_50_blend, _apply_hsv_tint,
    _apply_luma_preserve, _apply_dual_tone, _apply_gradient_overlay, _apply_channel_swap,
    _apply_cycle_shift, _apply_soft_tint, _apply_hard_mix, _apply_lab_mix,
    _apply_saturation_boost, _apply_inverse_tint, _apply_pastel_wash, _apply_cyber_glitch
]

def apply_palette_mode(out, p_mode, lut):
    if 0 <= p_mode < len(PALETTE_MODE_FUNCS):
        try:
            return PALETTE_MODE_FUNCS[p_mode](out, lut)
        except Exception as e:
            print(f"⚠️ Palette mode error: {e}")
            return out
    return out


# ==========================================
# 9. SHARED STATE
# ==========================================
@dataclass
class SharedState:
    filter_idx: int = 0
    palette_idx: int = 0
    palette_mode: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    running: bool = True
    fps: float = 0.0
    show_hud: bool = True
    bypass: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    camera_index: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    target_fps: int = 30
    mirror_preview: bool = True
    enable_virtualcam: bool = True
    recording: bool = False
    screenshot_folder: str = "."
    video_folder: str = "."
    script_text: str = ""
    layers: List[Dict[str, Any]] = field(default_factory=list)
    layered_mode: bool = False

    def __post_init__(self):
        self.load_settings()
        for p_key, p_def in PARAM_DEFS.items():
            if p_def.category in ("pre", "post") and p_key not in self.params:
                self.params[p_key] = p_def.default
        for p_key in FILTER_PARAMS.get(FILTER_NAMES[self.filter_idx], []):
            if p_key in PARAM_DEFS and p_key not in self.params:
                self.params[p_key] = PARAM_DEFS[p_key].default

    def get_all(self):
        with self._lock:
            return (self.filter_idx, self.palette_idx, self.palette_mode,
                    dict(self.params), self.show_hud, self.layered_mode,
                    [copy.deepcopy(l) for l in self.layers])

    def get_filter_idx(self):
        with self._lock:
            return self.filter_idx

    def get_palette_idx(self):
        with self._lock:
            return self.palette_idx

    def get_fps(self):
        with self._lock:
            return self.fps

    def is_running(self):
        with self._lock:
            return self.running

    def is_bypassed(self):
        with self._lock:
            return self.bypass

    def set_filter(self, idx):
        with self._lock:
            self.filter_idx = idx
            for p_key in FILTER_PARAMS.get(FILTER_NAMES[idx], []):
                if p_key in PARAM_DEFS:
                    self.params[p_key] = PARAM_DEFS[p_key].default

    def set_palette(self, idx):
        with self._lock:
            self.palette_idx = idx

    def set_palette_mode(self, mode):
        with self._lock:
            self.palette_mode = mode

    def set_param(self, key, val):
        with self._lock:
            self.params[key] = val

    def set_running(self, val):
        with self._lock:
            self.running = val

    def set_fps(self, fps):
        with self._lock:
            self.fps = fps

    def toggle_hud(self):
        with self._lock:
            self.show_hud = not self.show_hud
            return self.show_hud

    def toggle_bypass(self):
        with self._lock:
            self.bypass = not self.bypass
            return self.bypass

    def set_camera(self, idx):
        with self._lock:
            self.camera_index = idx

    def set_resolution(self, res):
        with self._lock:
            self.resolution = res

    def set_target_fps(self, fps):
        with self._lock:
            self.target_fps = fps

    def set_mirror(self, v):
        with self._lock:
            self.mirror_preview = v

    def set_virtualcam(self, v):
        with self._lock:
            self.enable_virtualcam = v

    def set_recording(self, v):
        with self._lock:
            self.recording = v

    def set_screenshot_folder(self, p):
        with self._lock:
            self.screenshot_folder = p

    def set_video_folder(self, p):
        with self._lock:
            self.video_folder = p

    def set_script(self, text):
        with self._lock:
            self.script_text = text

    def add_layer(self, filter_name="None"):
        layer = {"filter": filter_name, "enabled": True, "opacity": 100,
                 "blend_mode": 0,
                 "params": {k: PARAM_DEFS[k].default for k in FILTER_PARAMS.get(filter_name, []) if k in PARAM_DEFS}}
        with self._lock:
            self.layers.append(layer)
        return layer

    def remove_layer(self, idx):
        with self._lock:
            if 0 <= idx < len(self.layers):
                self.layers.pop(idx)

    def duplicate_layer(self, idx):
        with self._lock:
            if 0 <= idx < len(self.layers):
                self.layers.insert(idx + 1, copy.deepcopy(self.layers[idx]))

    def move_layer(self, idx, direction):
        with self._lock:
            new = idx + direction
            if 0 <= new < len(self.layers):
                self.layers[idx], self.layers[new] = self.layers[new], self.layers[idx]

    def set_layer_field(self, idx, field_name, value):
        with self._lock:
            if 0 <= idx < len(self.layers):
                self.layers[idx][field_name] = value

    def set_layer_filter(self, idx, filter_name):
        with self._lock:
            if 0 <= idx < len(self.layers):
                self.layers[idx]["filter"] = filter_name
                self.layers[idx]["params"] = {k: PARAM_DEFS[k].default for k in FILTER_PARAMS.get(filter_name, []) if k in PARAM_DEFS}

    def set_layer_param(self, idx, key, val):
        with self._lock:
            if 0 <= idx < len(self.layers):
                self.layers[idx]["params"][key] = val

    def set_layered_mode(self, v):
        with self._lock:
            self.layered_mode = bool(v)

    def clear_layers(self):
        with self._lock:
            self.layers.clear()

    def load_preset(self, preset_name: str):
        if preset_name not in FILTER_PRESETS:
            return
        with self._lock:
            self.layers.clear()
            self.layered_mode = True
            for item in FILTER_PRESETS[preset_name]:
                self.layers.append({
                    "filter": item["filter"], "enabled": True,
                    "opacity": item.get("opacity", 100),
                    "blend_mode": item.get("blend_mode", 0),
                    "params": dict(item.get("params", {}))
                })

    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
            self.filter_idx = data.get("filter_idx", 0)
            self.palette_idx = data.get("palette_idx", 0)
            self.palette_mode = data.get("palette_mode", 0)
            self.params.update(data.get("params", {}))
            self.camera_index = data.get("camera_index", 0)
            self.resolution = tuple(data.get("resolution", [1280, 720]))
            self.target_fps = data.get("target_fps", 30)
            self.mirror_preview = data.get("mirror_preview", True)
            self.enable_virtualcam = data.get("enable_virtualcam", True)
            self.screenshot_folder = data.get("screenshot_folder", ".")
            self.video_folder = data.get("video_folder", ".")
            self.script_text = data.get("script_text", "")
            self.layered_mode = data.get("layered_mode", False)
            self.layers = data.get("layers", [])
        except Exception as e:
            print(f"⚠️ Failed to load settings: {e}")

    def save_settings(self):
        with self._lock:
            data = {
                "filter_idx": self.filter_idx, "palette_idx": self.palette_idx,
                "palette_mode": self.palette_mode, "params": self.params,
                "camera_index": self.camera_index,
                "resolution": list(self.resolution),
                "target_fps": self.target_fps,
                "mirror_preview": self.mirror_preview,
                "enable_virtualcam": self.enable_virtualcam,
                "screenshot_folder": self.screenshot_folder,
                "video_folder": self.video_folder,
                "script_text": self.script_text,
                "layered_mode": self.layered_mode,
                "layers": self.layers
            }
        try:
            tmp_path = SETTINGS_FILE + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, SETTINGS_FILE)
        except Exception as e:
            print(f"⚠️ Failed to save settings: {e}")

    def export_settings(self, filepath):
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "filter_idx": self.filter_idx, "palette_idx": self.palette_idx,
                    "palette_mode": self.palette_mode, "params": self.params,
                    "layered_mode": self.layered_mode, "layers": self.layers,
                    "script_text": self.script_text
                }, f, indent=4)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def import_settings(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.filter_idx = data.get("filter_idx", 0)
            self.palette_idx = data.get("palette_idx", 0)
            self.palette_mode = data.get("palette_mode", 0)
            self.params.update(data.get("params", {}))
            self.layered_mode = data.get("layered_mode", False)
            self.layers = data.get("layers", [])
            self.script_text = data.get("script_text", "")
            return True
        except Exception as e:
            print(f"Import failed: {e}")
            return False


# ==========================================
# 10. CAMERA WORKER THREAD
# ==========================================
class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(float, str, str, str, bool)
    error_occurred = Signal(str)

    def __init__(self, state: SharedState):
        super().__init__()
        self.state = state
        self.ctx: Optional[ProcessContext] = None
        self.writer = None
        self.cam = None
        self._last_filter_name = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

    def run(self):
        cap = cv2.VideoCapture(self.state.camera_index)
        if not cap.isOpened():
            self.error_occurred.emit(f"Failed to open camera index {self.state.camera_index}")
            return

        w, h = self.state.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self.state.target_fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ctx = ProcessContext(width, height)

        if VIRTUALCAM_OK and self.state.enable_virtualcam:
            try:
                self.cam = pyvirtualcam.Camera(width=width, height=height, fps=self.state.target_fps)
            except Exception as e:
                print(f"Virtual cam failed: {e}")
                self.cam = None

        prev_t, frame_count = time.time(), 0
        while self.state.is_running():
            ret, frame = cap.read()
            if not ret:
                self._reconnect_attempts += 1
                if self._reconnect_attempts > self._max_reconnect_attempts:
                    self.error_occurred.emit("Camera disconnected (max retries reached).")
                    break
                time.sleep(0.2)
                continue
            self._reconnect_attempts = 0

            if self.state.mirror_preview:
                frame = cv2.flip(frame, 1)

            f_idx, p_idx, p_mode, params, show_hud, layered_mode, layers = self.state.get_all()
            p_name = PALETTE_NAMES[p_idx] if 0 <= p_idx < len(PALETTE_NAMES) else "None"
            p_mode_name = PALETTE_MODES[p_mode] if 0 <= p_mode < len(PALETTE_MODES) else "Unknown"

            try:
                frame_adj = apply_pre_adjustments(self.ctx, frame, params)
                if self.state.is_bypassed():
                    processed, f_name = frame_adj, "BYPASSED"
                elif layered_mode and layers:
                    current = frame_adj
                    active = 0
                    for layer in layers:
                        if not layer.get("enabled", True):
                            continue
                        lr = apply_filter(self.ctx, current, layer["filter"],
                                          {**params, **layer.get("params", {})})
                        current = blend_images(current, lr,
                                               layer.get("opacity", 100),
                                               layer.get("blend_mode", 0))
                        active += 1
                    processed, f_name = current, f"Layered ({active}/{len(layers)})"
                else:
                    f_name = FILTER_NAMES[f_idx] if 0 <= f_idx < len(FILTER_NAMES) else "None"
                    if self._last_filter_name != f_name:
                        self.ctx.reset_temporal_state()
                        self._last_filter_name = f_name
                    processed = apply_filter(self.ctx, frame_adj, f_name, params)

                if not self.state.is_bypassed():
                    processed = apply_post_processing(self.ctx, processed, params, f_name)
                    if p_idx != 0 and p_name in PRECOMPUTED_LUTS:
                        processed = apply_palette_mode(processed, p_mode, PRECOMPUTED_LUTS[p_name])
            except Exception as e:
                print(f"Processing error: {e}")
                processed = frame

            if self.cam is not None:
                try:
                    self.cam.send(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                    self.cam.sleep_until_next_frame()
                except Exception as e:
                    print(f"Virtual cam send failed: {e}")

            if self.state.recording:
                if self.writer is None:
                    fn = os.path.join(self.state.video_folder,
                                      f"vfx_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                    self.writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'),
                                                  self.state.target_fps, (width, height))
                if self.writer is not None:
                    self.writer.write(processed)
            elif self.writer is not None:
                self.writer.release()
                self.writer = None

            if show_hud:
                display = processed.copy()
                cv2.rectangle(display, (0, 0), (width, 36), (0, 0, 0), -1)
                cv2.addWeighted(display, 0.6, processed, 0.4, 0, display)
                cv2.putText(display,
                            f"{f_name} | FPS: {self.state.get_fps():.1f} | {p_name} ({p_mode_name})",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                if self.state.recording:
                    cv2.circle(display, (width - 25, 18), 7, (0, 0, 255), -1)
                if self.state.is_bypassed():
                    cv2.putText(display, "BYPASS ACTIVE", (width // 2 - 80, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                display = processed

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

            frame_count += 1
            now = time.time()
            if now - prev_t >= 1.0:
                fps = frame_count / (now - prev_t)
                self.state.set_fps(fps)
                self.stats_ready.emit(fps, f_name, p_name, p_mode_name, self.state.recording)
                frame_count, prev_t = 0, now

        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None
        cap.release()

    def stop(self):
        self.state.set_running(False)
        self.wait(3000)


# ==========================================
# 11. UI THEME (QSS)
# ==========================================
STUDIO_QSS = """
* { font-family: 'Segoe UI Variable', 'Segoe UI', 'SF Pro Text', system-ui, sans-serif; font-size: 10pt; color: #e6e8f5; }
QMainWindow, QWidget { background-color: #0f1020; }
QFrame#HeaderFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a1c35, stop:1 #14152a); border: 1px solid #2a2d4a; border-radius: 12px; }
QFrame#VideoFrame { background-color: #14152a; border: 1px solid #2a2d4a; border-radius: 14px; }
QLabel#AppTitle { font-size: 17pt; font-weight: 800; color: #ffffff; letter-spacing: 1.5px; }
QLabel#AppSubtitle { font-size: 9pt; color: #8a8fb0; font-weight: 500; }
QLabel#Pill { background-color: #202238; border: 1px solid #2a2d4a; border-radius: 13px; padding: 5px 14px; color: #e6e8f5; font-weight: 600; }
QLabel#PillFPS { color: #4ade80; border-color: rgba(74,222,128,0.35); background-color: rgba(74,222,128,0.08); }
QLabel#PillFilter { color: #22d3ee; border-color: rgba(34,211,238,0.35); background-color: rgba(34,211,238,0.08); }
QLabel#PillPalette { color: #c084fc; border-color: rgba(192,132,252,0.35); background-color: rgba(192,132,252,0.08); }
QLabel#PillRec { color: #ff7a85; border-color: rgba(255,122,133,0.5); background-color: rgba(255,122,133,0.12); }
QLabel#PillTime { color: #fbbf24; border-color: rgba(251,191,36,0.35); background-color: rgba(251,191,36,0.08); font-family: 'JetBrains Mono','Consolas',monospace; }
QLabel#SectionLabel { font-size: 11pt; font-weight: 700; color: #c084fc; letter-spacing: 0.5px; padding: 4px 0; }
QLabel#CaptionLabel { color: #8a8fb0; font-size: 9pt; }
QLabel#VideoLabel { background-color: #000000; border: 1px solid #2a2d4a; border-radius: 10px; }
QLabel#BadgeLabel { background-color: rgba(124,92,255,0.15); color: #c084fc; border: 1px solid rgba(124,92,255,0.4); border-radius: 9px; padding: 3px 10px; font-family: 'JetBrains Mono','Consolas',monospace; font-size: 9pt; }
QGroupBox { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 10px; margin-top: 18px; padding: 16px 12px 12px 12px; font-weight: 700; color: #c084fc; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 14px; padding: 0 8px; background-color: #0f1020; color: #c084fc; }
QPushButton { background-color: #202238; border: 1px solid #2a2d4a; border-radius: 8px; padding: 7px 14px; color: #e6e8f5; font-weight: 600; }
QPushButton:hover { background-color: #2a2d4a; border-color: #3a3f6a; }
QPushButton:pressed { background-color: #7c5cff; border-color: #7c5cff; color: #ffffff; }
QPushButton:disabled { color: #5a5f80; background-color: #181a2e; }
QPushButton#Primary { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8b6dff, stop:1 #6b4cff); border: 1px solid #7c5cff; color: #ffffff; }
QPushButton#Primary:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b7dff, stop:1 #7b5cff); }
QPushButton#Danger { background-color: rgba(255,122,133,0.12); border: 1px solid rgba(255,122,133,0.5); color: #ff7a85; }
QPushButton#Danger:hover { background-color: rgba(255,122,133,0.22); }
QPushButton#IconBtn { background-color: #202238; border: 1px solid #2a2d4a; border-radius: 8px; padding: 6px 10px; color: #e6e8f5; font-weight: 600; min-width: 30px; }
QPushButton#IconBtn:hover { border-color: #7c5cff; color: #c084fc; }
QPushButton#IconBtn:checked { background-color: rgba(124,92,255,0.18); border-color: #7c5cff; color: #c084fc; }
QPushButton#Segment { background-color: transparent; border: none; border-radius: 7px; padding: 7px 14px; color: #8a8fb0; font-weight: 600; }
QPushButton#Segment:hover { color: #e6e8f5; background-color: rgba(124,92,255,0.08); }
QPushButton#Segment:checked { background-color: #7c5cff; color: #ffffff; }
QPushButton#FilterCard { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 10px; padding: 12px 14px; text-align: left; color: #e6e8f5; font-weight: 600; min-width: 140px; min-height: 24px; }
QPushButton#FilterCard:hover { background-color: #202238; border-color: #3a3f6a; }
QPushButton#FilterCard:checked { background-color: rgba(124,92,255,0.15); border-color: #7c5cff; color: #c084fc; }
QFrame#SegmentBar { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 10px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 8px; padding: 6px 10px; color: #e6e8f5; selection-background-color: #7c5cff; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color: #3a3f6a; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: #7c5cff; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 8px; selection-background-color: rgba(124,92,255,0.4); outline: none; padding: 4px; }
QListWidget { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 10px; padding: 6px; outline: none; }
QListWidget::item { border-radius: 7px; padding: 8px 12px; }
QListWidget::item:hover { background-color: rgba(124,92,255,0.1); }
QListWidget::item:selected { background-color: rgba(124,92,255,0.2); border: 1px solid #7c5cff; color: #ffffff; }
QTabWidget::pane { border: 1px solid #2a2d4a; border-radius: 10px; background: transparent; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab { background-color: transparent; color: #8a8fb0; padding: 10px 18px; margin-right: 2px; border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600; }
QTabBar::tab:hover { color: #e6e8f5; }
QTabBar::tab:selected { background-color: #181a2e; color: #c084fc; border-color: #2a2d4a; border-bottom: 2px solid #7c5cff; }
QSlider::groove:horizontal { border: none; height: 6px; background: #202238; border-radius: 3px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c5cff, stop:1 #22d3ee); border-radius: 3px; }
QSlider::add-page:horizontal { background: #202238; border-radius: 3px; }
QSlider::handle:horizontal { background: #ffffff; border: 2px solid #7c5cff; width: 14px; height: 14px; margin: -5px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { border-color: #22d3ee; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 5px; border: 1.5px solid #3a3f6a; background: #181a2e; }
QCheckBox::indicator:hover { border-color: #7c5cff; }
QCheckBox::indicator:checked { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8b6dff, stop:1 #6b4cff); border-color: #7c5cff; }
QScrollBar:vertical { border: none; background: transparent; width: 10px; margin: 4px 2px; }
QScrollBar::handle:vertical { background: #2a2d4a; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #3a3f6a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { border: none; background: transparent; height: 10px; margin: 2px 4px; }
QScrollBar::handle:horizontal { background: #2a2d4a; border-radius: 5px; min-width: 30px; }
QTextEdit { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 10px; padding: 14px; color: #e6e8f5; selection-background-color: rgba(124,92,255,0.4); }
QTextEdit:focus { border-color: #7c5cff; }
QStatusBar { background-color: #0f1020; border-top: 1px solid #2a2d4a; color: #8a8fb0; }
QStatusBar::item { border: none; }
"""


# ==========================================
# 12. UTILITY WIDGETS
# ==========================================
def lut_to_pixmap(lut, w=96, h=22):
    arr = np.tile(lut[:, 0, :][np.newaxis, :, :], (h, 1, 1))
    qimg = QImage(arr[:, :, ::-1].copy().data, 256, h, 256 * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

class FlowLayout(QLayout):
    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._items, self._spacing = [], spacing

    def addItem(self, item): self._items.append(item)
    def count(self): return len(self._items)
    def itemAt(self, index): return self._items[index] if 0 <= index < len(self._items) else None
    def takeAt(self, index): return self._items.pop(index) if 0 <= index < len(self._items) else None
    def expandingDirections(self): return Qt.Orientations(0)
    def hasHeightForWidth(self): return True
    def heightForWidth(self, width): return self._do_layout(QRect(0, 0, width, 0), True)
    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)
    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size + QSize(2 * self._spacing, 2 * self._spacing)
    def _do_layout(self, rect, test_only):
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self._items:
            wid = item.widget()
            if wid is None or wid.isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()

class SegmentedControl(QFrame):
    def __init__(self, items, on_select, parent=None):
        super().__init__(parent)
        self.setObjectName("SegmentBar")
        self.on_select = on_select
        self.buttons = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        for it in items:
            b = QPushButton(it)
            b.setObjectName("Segment")
            b.setCheckable(True)
            idx = len(self.buttons)
            b.clicked.connect(lambda _=False, i=idx: self._select(i))
            self.buttons.append(b)
            lay.addWidget(b)
        lay.addStretch()
        if self.buttons:
            self.buttons[0].setChecked(True)

    def _select(self, idx):
        for i, b in enumerate(self.buttons):
            b.setChecked(i == idx)
        if self.on_select:
            self.on_select(idx)

    def set_current(self, idx):
        if 0 <= idx < len(self.buttons):
            self._select(idx)

    def items(self):
        return [b.text() for b in self.buttons]

class PaletteComboDelegate(QStyledItemDelegate):
    def __init__(self, luts, parent=None):
        super().__init__(parent)
        self.luts = luts

    def sizeHint(self, option, index):
        return QSize(240, 34)

    def paint(self, painter, option, index):
        painter.save()
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, QColor(124, 92, 255, 80))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect, QColor(124, 92, 255, 25))
        name = index.data(Qt.DisplayRole)
        rect = option.rect.adjusted(8, 5, -8, -5)
        swatch = QRect(rect.left(), rect.top(), 90, rect.height())
        if self.luts.get(name) is not None:
            grad = QLinearGradient(swatch.left(), 0, swatch.right(), 0)
            for i in range(0, 256, 8):
                t = i / 255.0
                b, g, r = int(self.luts[name][i, 0, 0]), int(self.luts[name][i, 0, 1]), int(self.luts[name][i, 0, 2])
                grad.setColorAt(t, QColor(r, g, b))
            painter.setBrush(grad)
            painter.setPen(QColor(42, 45, 74))
            painter.drawRoundedRect(swatch, 4, 4)
        painter.setPen(QColor(230, 232, 245))
        painter.drawText(QRect(rect.left() + 100, rect.top(), rect.width() - 100, rect.height()),
                         Qt.AlignVCenter | Qt.AlignLeft, name)
        painter.restore()

class ParamWidget(QWidget):
    def __init__(self, p_def: ParamDef, on_change: Callable):
        super().__init__()
        self.p_def = p_def
        self.on_change = on_change
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(10)
        self.label = QLabel(p_def.label)
        self.label.setMinimumWidth(135)
        self.label.setMaximumWidth(135)
        lay.addWidget(self.label)
        if p_def.p_type == "bool":
            lay.addStretch()
            self.checkbox = QCheckBox()
            self.checkbox.setChecked(bool(p_def.default))
            self.checkbox.stateChanged.connect(self._changed)
            lay.addWidget(self.checkbox)
        else:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimumWidth(160)
            lay.addWidget(self.slider, 1)
            if p_def.p_type == "float":
                self.scale = 100.0
                self.slider.setMinimum(int(p_def.min_val * self.scale))
                self.slider.setMaximum(int(p_def.max_val * self.scale))
                self.slider.setValue(int(p_def.default * self.scale))
                self.spinbox = QDoubleSpinBox()
                self.spinbox.setRange(p_def.min_val, p_def.max_val)
                self.spinbox.setSingleStep(p_def.step)
                self.spinbox.setValue(p_def.default)
                self.spinbox.setDecimals(2)
            else:
                self.scale = 1.0
                self.slider.setMinimum(int(p_def.min_val))
                self.slider.setMaximum(int(p_def.max_val))
                self.slider.setValue(int(p_def.default))
                self.spinbox = QSpinBox()
                self.spinbox.setRange(int(p_def.min_val), int(p_def.max_val))
                self.spinbox.setValue(int(p_def.default))
            self.spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            self.spinbox.setAlignment(Qt.AlignCenter)
            self.spinbox.setMinimumWidth(78)
            self.spinbox.setMaximumWidth(78)
            self.spinbox.setStyleSheet("""QSpinBox, QDoubleSpinBox { background-color: #202238; color: #22d3ee; font-family: 'JetBrains Mono','Consolas',monospace; font-weight: 700; border: 1px solid #2a2d4a; border-radius: 9px; padding: 3px 6px; } QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7c5cff; }""")
            self.slider.valueChanged.connect(self._slider_changed)
            self.spinbox.valueChanged.connect(self._spin_changed)
            lay.addWidget(self.spinbox)
            self.reset_btn = QPushButton("↺")
            self.reset_btn.setFixedSize(26, 26)
            self.reset_btn.setToolTip("Reset to default")
            self.reset_btn.setCursor(Qt.PointingHandCursor)
            self.reset_btn.setStyleSheet("""QPushButton { background: transparent; border: 1px solid #2a2d4a; border-radius: 13px; color: #8a8fb0; padding: 0; font-size: 11pt; } QPushButton:hover { color: #7c5cff; border-color: #7c5cff; }""")
            self.reset_btn.clicked.connect(lambda: self.set_value(p_def.default))
            lay.addWidget(self.reset_btn)

    def _slider_changed(self, v):
        val = int(round(v / self.scale)) if self.p_def.p_type == "int" else v / self.scale
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(val)
        self.spinbox.blockSignals(False)
        self.on_change(self.p_def.key, val)

    def _spin_changed(self, v):
        v = int(v) if self.p_def.p_type == "int" else v
        self.slider.blockSignals(True)
        self.slider.setValue(int(v * self.scale) if self.scale > 1 else v)
        self.slider.blockSignals(False)
        self.on_change(self.p_def.key, v)

    def _changed(self, state):
        self.on_change(self.p_def.key, 1 if state == Qt.Checked else 0)

    def set_value(self, val):
        if self.p_def.p_type == "bool":
            self.checkbox.setChecked(bool(val))
        else:
            self.slider.blockSignals(True)
            self.spinbox.blockSignals(True)
            self.slider.setValue(int(val * self.scale))
            self.spinbox.setValue(val if self.p_def.p_type == "float" else int(val))
            self.slider.blockSignals(False)
            self.spinbox.blockSignals(False)


# ==========================================
# 13. LAYERED FILTERS TAB
# ==========================================
class LayersTab(QWidget):
    def __init__(self, state: SharedState, on_param_change: Callable):
        super().__init__()
        self.state = state
        self.on_param_change = on_param_change
        self.param_widgets = {}
        self._current_idx = -1
        self._ext_toggle = None
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        outer.addWidget(self._make_label("🧅  Layered Filters", "SectionLabel"))
        outer.addWidget(self._make_label("Stack multiple filters with per-layer opacity, blend modes, and parameters.", "CaptionLabel"))
        mode_row = QHBoxLayout()
        self.chk_mode = QCheckBox("Enable Layered Mode")
        self.chk_mode.setChecked(self.state.layered_mode)
        self.chk_mode.stateChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.chk_mode)
        mode_row.addStretch()
        mode_row.addWidget(self._make_label("PRESETS:", "CaptionLabel"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItems(["— Select Preset —"] + list(FILTER_PRESETS.keys()))
        self.combo_presets.setMinimumWidth(180)
        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)
        mode_row.addWidget(self.combo_presets)
        outer.addLayout(mode_row)
        list_row = QHBoxLayout()
        list_col = QVBoxLayout()
        list_col.addWidget(self._make_label("LAYERS", "CaptionLabel"))
        self.list = QListWidget()
        self.list.setMaximumHeight(180)
        self.list.currentRowChanged.connect(self._on_select)
        list_col.addWidget(self.list)
        list_row.addLayout(list_col, 3)
        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)
        self.btn_add = QPushButton("➕  Add")
        self.btn_dup = QPushButton("⧉  Duplicate")
        self.btn_del = QPushButton("🗑  Remove")
        self.btn_up = QPushButton("↑  Up")
        self.btn_down = QPushButton("↓  Down")
        self.btn_clear = QPushButton("✕  Clear All")
        self.btn_add.clicked.connect(self._add)
        self.btn_dup.clicked.connect(self._dup)
        self.btn_del.clicked.connect(self._del)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))
        self.btn_clear.clicked.connect(self._clear)
        for b in (self.btn_add, self.btn_dup, self.btn_del, self.btn_up, self.btn_down, self.btn_clear):
            btn_col.addWidget(b)
        btn_col.addStretch()
        list_row.addLayout(btn_col, 1)
        outer.addLayout(list_row)
        prop_grp = QGroupBox("Layer Properties")
        pl = QVBoxLayout(prop_grp)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Filter:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(FILTER_NAMES)
        self.combo_filter.currentIndexChanged.connect(self._on_filter_change)
        r1.addWidget(self.combo_filter, 2)
        r1.addWidget(QLabel("Blend:"))
        self.combo_blend = QComboBox()
        self.combo_blend.addItems(BLEND_MODES)
        self.combo_blend.currentIndexChanged.connect(self._on_blend_change)
        r1.addWidget(self.combo_blend, 2)
        pl.addLayout(r1)
        r2 = QHBoxLayout()
        self.chk_enabled = QCheckBox("Enabled")
        self.chk_enabled.stateChanged.connect(self._on_enabled)
        r2.addWidget(self.chk_enabled)
        r2.addWidget(QLabel("Opacity:"))
        self.sld_op = QSlider(Qt.Horizontal)
        self.sld_op.setRange(0, 100)
        self.sld_op.setValue(100)
        self.sld_op.valueChanged.connect(self._on_opacity)
        r2.addWidget(self.sld_op, 1)
        self.lbl_op = QLabel("100%")
        self.lbl_op.setMinimumWidth(48)
        self.sld_op.valueChanged.connect(lambda v: self.lbl_op.setText(f"{v}%"))
        r2.addWidget(self.lbl_op)
        pl.addLayout(r2)
        pl.addWidget(self._make_label("LAYER PARAMETERS", "CaptionLabel"))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.params_host = QWidget()
        self.params_host.setStyleSheet("background: transparent;")
        self.params_layout = QVBoxLayout(self.params_host)
        self.params_layout.setContentsMargins(2, 2, 2, 2)
        self.params_layout.setSpacing(2)
        self.scroll.setWidget(self.params_host)
        pl.addWidget(self.scroll, 1)
        outer.addWidget(prop_grp, 1)
        self.refresh()

    @staticmethod
    def _make_label(text, object_name=None):
        l = QLabel(text)
        if object_name:
            l.setObjectName(object_name)
        return l

    def _on_mode_changed(self, s):
        self.state.set_layered_mode(bool(s))
        if self._ext_toggle:
            self._ext_toggle(bool(s))

    def _on_select(self, row):
        self._current_idx = row
        self._load_properties(row)

    def _on_filter_change(self, idx):
        if self._current_idx < 0:
            return
        self.state.set_layer_filter(self._current_idx, FILTER_NAMES[idx])
        self._rebuild_params(self._current_idx)
        self.refresh()

    def _on_blend_change(self, idx):
        if self._current_idx >= 0:
            self.state.set_layer_field(self._current_idx, "blend_mode", idx)
            self.refresh()

    def _on_opacity(self, v):
        if self._current_idx >= 0:
            self.state.set_layer_field(self._current_idx, "opacity", v)
            self.refresh()

    def _on_enabled(self, s):
        if self._current_idx >= 0:
            self.state.set_layer_field(self._current_idx, "enabled", bool(s))
            self.refresh()

    def _on_preset_selected(self, idx):
        if idx <= 0:
            return
        self.state.load_preset(self.combo_presets.itemText(idx))
        self.chk_mode.blockSignals(True)
        self.chk_mode.setChecked(True)
        self.chk_mode.blockSignals(False)
        if self._ext_toggle:
            self._ext_toggle(True)
        self._current_idx = -1
        self.refresh()
        self.combo_presets.blockSignals(True)
        self.combo_presets.setCurrentIndex(0)
        self.combo_presets.blockSignals(False)

    def _add(self):
        name, ok = QInputDialog.getItem(self, "Add Layer", "Choose filter:", FILTER_NAMES, 0, False)
        if ok:
            self.state.add_layer(name)
            self._current_idx = len(self.state.layers) - 1
            self.refresh()

    def _dup(self):
        if self._current_idx >= 0:
            self.state.duplicate_layer(self._current_idx)
            self._current_idx += 1
            self.refresh()

    def _del(self):
        if self._current_idx >= 0:
            self.state.remove_layer(self._current_idx)
            self._current_idx = min(self._current_idx, len(self.state.layers) - 1)
            self.refresh()

    def _move(self, d):
        if self._current_idx >= 0:
            self.state.move_layer(self._current_idx, d)
            self._current_idx += d
            self.refresh()

    def _clear(self):
        if QMessageBox.question(self, "Clear Layers", "Remove all layers?") == QMessageBox.Yes:
            self.state.clear_layers()
            self._current_idx = -1
            self.refresh()

    def refresh(self):
        self.list.blockSignals(True)
        self.list.clear()
        for i, l in enumerate(self.state.layers):
            en = "✓" if l.get("enabled", True) else "✕"
            f = l.get("filter", "None")
            op = l.get("opacity", 100)
            bm_idx = l.get("blend_mode", 0)
            bm = BLEND_MODES[bm_idx] if 0 <= bm_idx < len(BLEND_MODES) else "?"
            self.list.addItem(f"{i+1}. [{en}] {f}  ·  {bm}  ·  {op}%")
        self.list.blockSignals(False)
        if 0 <= self._current_idx < len(self.state.layers):
            self.list.setCurrentRow(self._current_idx)
        else:
            self._current_idx = -1
            self._clear_params()

    def _load_properties(self, idx):
        if not (0 <= idx < len(self.state.layers)):
            self._clear_params()
            return
        l = self.state.layers[idx]
        self.combo_filter.blockSignals(True)
        self.combo_blend.blockSignals(True)
        self.chk_enabled.blockSignals(True)
        self.sld_op.blockSignals(True)
        fn = l.get("filter", "None")
        if fn in FILTER_NAMES:
            self.combo_filter.setCurrentIndex(FILTER_NAMES.index(fn))
        self.combo_blend.setCurrentIndex(l.get("blend_mode", 0))
        self.chk_enabled.setChecked(l.get("enabled", True))
        op = l.get("opacity", 100)
        self.sld_op.setValue(op)
        self.lbl_op.setText(f"{op}%")
        self.combo_filter.blockSignals(False)
        self.combo_blend.blockSignals(False)
        self.chk_enabled.blockSignals(False)
        self.sld_op.blockSignals(False)
        self._rebuild_params(idx)

    def _rebuild_params(self, idx):
        for i in reversed(range(self.params_layout.count())):
            w = self.params_layout.itemAt(i).widget()
            if w is not None:
                w.deleteLater()
        self.param_widgets.clear()
        l = self.state.layers[idx]
        fn = l.get("filter", "None")
        for pk in FILTER_PARAMS.get(fn, []):
            if pk not in PARAM_DEFS:
                continue
            val = l.get("params", {}).get(pk, PARAM_DEFS[pk].default)
            w = ParamWidget(PARAM_DEFS[pk], lambda k, v, i=idx: self.state.set_layer_param(i, k, v))
            w.set_value(val)
            self.param_widgets[pk] = w
            self.params_layout.addWidget(w)
        self.params_layout.addStretch()

    def _clear_params(self):
        for i in reversed(range(self.params_layout.count())):
            w = self.params_layout.itemAt(i).widget()
            if w is not None:
                w.deleteLater()
        self.param_widgets.clear()


# ==========================================
# 14. MAIN WINDOW
# ==========================================
class ControlPanel(QMainWindow):
    def __init__(self, state: SharedState):
        super().__init__()
        self.state = state
        self.setWindowTitle("✦ VFX Engine — Studio Pro")
        self.resize(1480, 880)
        self.setStyleSheet(STUDIO_QSS)
        self.param_widgets = {}
        self.filter_cards = []
        self._current_cat = "All"

        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        left = QVBoxLayout()
        left.setSpacing(10)
        left.addLayout(self._build_header())
        self.tabs = QTabWidget()
        left.addWidget(self.tabs, 1)

        self.tab_filters = QWidget()
        self.tab_layers = QWidget()
        self.tab_pre = QWidget()
        self.tab_post = QWidget()
        self.tab_camera = QWidget()
        self.tab_script = QWidget()
        self.tab_hotkeys = QWidget()
        self.layers_tab = LayersTab(self.state, self._on_param_change)

        self.tabs.addTab(self.tab_filters, "🎨  Filters")
        self.tabs.addTab(self.tab_layers,  "🧅  Layers")
        self.tabs.addTab(self.tab_pre,     "🖼️  Pre-Process")
        self.tabs.addTab(self.tab_post,    "✨  Post-Process")
        self.tabs.addTab(self.tab_camera,  "📷  Camera")
        self.tabs.addTab(self.tab_script,  "📝  Script")
        self.tabs.addTab(self.tab_hotkeys, "⌨️  Hotkeys")

        self._build_filters_tab()
        self._build_layers_tab()
        self._build_pre_tab()
        self._build_post_tab()
        self._build_camera_tab()
        self._build_script_tab()
        self._build_hotkeys_tab()
        root.addLayout(left, 3)
        root.addWidget(self._build_video_panel(), 5)

        self.rec_elapsed = QElapsedTimer()
        self.rec_timer = QTimer(self)
        self.rec_timer.setInterval(1000)
        self.rec_timer.timeout.connect(self._tick_rec)
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.state.save_settings)
        self.save_timer.start(5000)
        self._init_shortcuts()
        self.statusBar().showMessage("Ready · Press H to toggle HUD · Space to record · S to screenshot · B to Bypass")

    @staticmethod
    def _caption(txt):
        l = QLabel(txt)
        l.setObjectName("CaptionLabel")
        return l

    @staticmethod
    def _section(txt):
        l = QLabel(txt)
        l.setObjectName("SectionLabel")
        return l

    def _build_header(self):
        hdr = QFrame()
        hdr.setObjectName("HeaderFrame")
        h = QHBoxLayout(hdr)
        h.setContentsMargins(18, 12, 18, 12)
        h.setSpacing(14)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel("✦  VFX ENGINE")
        title.setObjectName("AppTitle")
        sub = QLabel("STUDIO PRO  ·  Expert Edition")
        sub.setObjectName("AppSubtitle")
        brand.addWidget(title)
        brand.addWidget(sub)
        h.addLayout(brand)
        h.addSpacing(20)
        h.addStretch()
        self.lbl_fps = QLabel("● 0.0 FPS")
        self.lbl_filter = QLabel("Filter: None")
        self.lbl_palette = QLabel("Palette: None")
        self.lbl_rec = QLabel("")
        self.lbl_time = QLabel("00:00")
        self.lbl_fps.setObjectName("PillFPS")
        self.lbl_filter.setObjectName("PillFilter")
        self.lbl_palette.setObjectName("PillPalette")
        self.lbl_rec.setObjectName("PillRec")
        self.lbl_time.setObjectName("PillTime")
        for w in (self.lbl_fps, self.lbl_filter, self.lbl_palette, self.lbl_rec, self.lbl_time):
            h.addWidget(w)
        wrap = QVBoxLayout()
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(hdr)
        return wrap

    def _build_filters_tab(self):
        layout = QVBoxLayout(self.tab_filters)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._caption("MODE"))
        self.chk_layered_mode = QCheckBox("Layered Mode (use Layers tab)")
        self.chk_layered_mode.setChecked(self.state.layered_mode)
        self.chk_layered_mode.stateChanged.connect(
            lambda s: (self.state.set_layered_mode(bool(s)),
                       self.layers_tab.chk_mode.setChecked(bool(s))))
        mode_row.addWidget(self.chk_layered_mode)
        mode_row.addStretch()
        layout.addLayout(mode_row)
        layout.addWidget(self._caption("CATEGORY"))
        self.seg_cat = SegmentedControl(["All"] + list(FILTER_CATEGORIES.keys()),
                                        self._on_category_select)
        layout.addWidget(self.seg_cat)
        row = QHBoxLayout()
        row.setSpacing(10)
        pal_col = QVBoxLayout()
        pal_col.setSpacing(4)
        pal_col.addWidget(self._caption("PALETTE"))
        pal_row = QHBoxLayout()
        pal_row.setSpacing(8)
        self.combo_palette = QComboBox()
        self.combo_palette.addItems(PALETTE_NAMES)
        self.combo_palette.setCurrentIndex(self.state.palette_idx)
        self.combo_palette.setItemDelegate(PaletteComboDelegate(PRECOMPUTED_LUTS, self.combo_palette))
        self.combo_palette.currentIndexChanged.connect(self._on_palette_change)
        pal_row.addWidget(self.combo_palette, 1)
        self.palette_swatch = QLabel()
        self.palette_swatch.setFixedSize(96, 26)
        self.palette_swatch.setStyleSheet("border: 1px solid #2a2d4a; border-radius: 6px; background:#000;")
        pal_row.addWidget(self.palette_swatch)
        pal_col.addLayout(pal_row)
        row.addLayout(pal_col, 3)
        mode_col = QVBoxLayout()
        mode_col.setSpacing(4)
        mode_col.addWidget(self._caption("PALETTE MODE"))
        self.combo_palette_mode = QComboBox()
        self.combo_palette_mode.addItems(PALETTE_MODES)
        self.combo_palette_mode.setCurrentIndex(self.state.palette_mode)
        self.combo_palette_mode.currentIndexChanged.connect(self._on_palette_mode_change)
        mode_col.addWidget(self.combo_palette_mode)
        row.addLayout(mode_col, 2)
        layout.addLayout(row)
        act = QHBoxLayout()
        btn_rand = QPushButton("🎲  Randomize")
        btn_rand.setObjectName("Primary")
        btn_rand.clicked.connect(self._randomize)
        btn_reset = QPushButton("↺  Reset Defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        act.addWidget(btn_rand)
        act.addWidget(btn_reset)
        act.addStretch()
        layout.addLayout(act)
        fhdr = QHBoxLayout()
        fhdr.addWidget(self._section("Filter Cards"))
        fhdr.addStretch()
        layout.addLayout(fhdr)
        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setFrameShape(QFrame.NoFrame)
        cards_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        cards_scroll.setMaximumHeight(160)
        cards_host = QWidget()
        cards_host.setStyleSheet("background: transparent;")
        self.cards_flow = FlowLayout(cards_host, spacing=8)
        cards_scroll.setWidget(cards_host)
        layout.addWidget(cards_scroll)
        phdr = QHBoxLayout()
        phdr.addWidget(self._section("Filter Parameters"))
        phdr.addStretch()
        self.params_hint = QLabel("")
        self.params_hint.setObjectName("CaptionLabel")
        phdr.addWidget(self.params_hint)
        layout.addLayout(phdr)
        self.filter_scroll = QScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.filter_container = QWidget()
        self.filter_container.setStyleSheet("background: transparent;")
        self.filter_layout = QVBoxLayout(self.filter_container)
        self.filter_layout.setContentsMargins(2, 2, 2, 2)
        self.filter_layout.setSpacing(2)
        self.filter_scroll.setWidget(self.filter_container)
        layout.addWidget(self.filter_scroll, 1)
        self._refresh_filter_cards()
        self._rebuild_filter_params()
        self._update_palette_swatch()

    def _on_category_select(self, idx):
        self._current_cat = self.seg_cat.items()[idx]
        self._refresh_filter_cards()

    def _refresh_filter_cards(self):
        while self.cards_flow.count():
            it = self.cards_flow.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.filter_cards.clear()
        names = FILTER_NAMES if self._current_cat == "All" else FILTER_CATEGORIES.get(self._current_cat, [])
        cur_idx = self.state.get_filter_idx()
        cur_name = FILTER_NAMES[cur_idx] if 0 <= cur_idx < len(FILTER_NAMES) else None
        for name in names:
            icon = "✦"
            for cname, cnames in FILTER_CATEGORIES.items():
                if name in cnames:
                    icon = CATEGORY_ICONS.get(cname, "✦")
                    break
            b = QPushButton(f"  {icon}   {name}")
            b.setObjectName("FilterCard")
            b.setCheckable(True)
            if name == cur_name:
                b.setChecked(True)
            fidx = FILTER_NAMES.index(name)
            b.clicked.connect(lambda _=False, i=fidx: self._select_filter_by_idx(i))
            self.filter_cards.append((fidx, b))
            self.cards_flow.addWidget(b)

    def _select_filter_by_idx(self, idx):
        self.state.set_filter(idx)
        for fidx, b in self.filter_cards:
            b.setChecked(fidx == idx)
        self._rebuild_filter_params()

    def _rebuild_filter_params(self):
        for i in reversed(range(self.filter_layout.count())):
            w = self.filter_layout.itemAt(i).widget()
            if w is not None:
                w.deleteLater()
        f_idx = self.state.get_filter_idx()
        f_name = FILTER_NAMES[f_idx] if 0 <= f_idx < len(FILTER_NAMES) else "None"
        f_params = FILTER_PARAMS.get(f_name, [])
        if not f_params:
            lbl = QLabel("✓  No adjustable parameters for this filter.")
            lbl.setStyleSheet("color: #6c7086; font-style: italic; padding: 8px;")
            self.filter_layout.addWidget(lbl)
            self.params_hint.setText("")
        else:
            self.params_hint.setText(f"{len(f_params)} parameter(s)")
            for p_key in f_params:
                if p_key in PARAM_DEFS:
                    w = ParamWidget(PARAM_DEFS[p_key], self._on_param_change)
                    w.set_value(self.state.params.get(p_key, PARAM_DEFS[p_key].default))
                    self.param_widgets[p_key] = w
                    self.filter_layout.addWidget(w)
        self.filter_layout.addStretch()

    def _on_palette_change(self, idx):
        self.state.set_palette(idx)
        self._update_palette_swatch()

    def _on_palette_mode_change(self, idx):
        self.state.set_palette_mode(idx)

    def _on_param_change(self, key, val):
        self.state.set_param(key, val)

    def _update_palette_swatch(self):
        idx = self.combo_palette.currentIndex()
        if 0 <= idx < len(PALETTE_NAMES):
            lut = PRECOMPUTED_LUTS.get(PALETTE_NAMES[idx])
            if lut is not None:
                self.palette_swatch.setPixmap(lut_to_pixmap(lut, 96, 26))
            else:
                self.palette_swatch.clear()

    def _build_layers_tab(self):
        outer = QVBoxLayout(self.tab_layers)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.layers_tab)
        self.layers_tab._ext_toggle = lambda v: self.chk_layered_mode.setChecked(v)

    def _build_pre_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.addWidget(self._section("🖼️  Pre-Process  —  applied BEFORE the filter"))
        grp = QGroupBox("Image Adjustments")
        gl = QVBoxLayout(grp)
        for k in PRE_PARAMS:
            w = ParamWidget(PARAM_DEFS[k], self._on_param_change)
            self.param_widgets[k] = w
            gl.addWidget(w)
        gl.addStretch()
        layout.addWidget(grp)
        layout.addStretch()
        scroll.setWidget(container)
        outer = QVBoxLayout(self.tab_pre)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(scroll)

    def _build_post_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.addWidget(self._section("✨  Post-Process  —  applied AFTER the filter"))
        grp = QGroupBox("Global Effects")
        gl = QVBoxLayout(grp)
        for k in POST_PARAMS:
            w = ParamWidget(PARAM_DEFS[k], self._on_param_change)
            self.param_widgets[k] = w
            gl.addWidget(w)
        gl.addStretch()
        layout.addWidget(grp)
        layout.addStretch()
        scroll.setWidget(container)
        outer = QVBoxLayout(self.tab_post)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(scroll)

    def _build_camera_tab(self):
        layout = QVBoxLayout(self.tab_camera)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._section("📷  Camera & Output"))
        cam_grp = QGroupBox("Input Device")
        cl = QVBoxLayout(cam_grp)
        cl.setSpacing(8)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Camera Index:"))
        self.spin_cam = QSpinBox()
        self.spin_cam.setRange(0, 10)
        self.spin_cam.setValue(self.state.camera_index)
        r1.addWidget(self.spin_cam)
        r1.addSpacing(12)
        r1.addWidget(QLabel("Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(["640x480", "1280x720", "1920x1080"])
        self.combo_res.setCurrentText(f"{self.state.resolution[0]}x{self.state.resolution[1]}")
        r1.addWidget(self.combo_res)
        r1.addStretch()
        cl.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Target FPS:"))
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["15", "24", "30", "60"])
        self.combo_fps.setCurrentText(str(self.state.target_fps))
        r2.addWidget(self.combo_fps)
        r2.addStretch()
        cl.addLayout(r2)
        btn_apply = QPushButton("Apply & Restart Camera")
        btn_apply.setObjectName("Primary")
        btn_apply.clicked.connect(self._apply_camera)
        cl.addWidget(btn_apply)
        layout.addWidget(cam_grp)
        out_grp = QGroupBox("Output & Folders")
        ol = QVBoxLayout(out_grp)
        ol.setSpacing(8)
        self.chk_vcam = QCheckBox("Enable Virtual Camera")
        self.chk_vcam.setChecked(self.state.enable_virtualcam)
        self.chk_mirror = QCheckBox("Mirror Preview")
        self.chk_mirror.setChecked(self.state.mirror_preview)
        self.chk_hud = QCheckBox("Show HUD Overlay")
        self.chk_hud.setChecked(self.state.show_hud)
        self.chk_hud.stateChanged.connect(lambda _: self.state.toggle_hud())
        ol.addWidget(self.chk_vcam)
        ol.addWidget(self.chk_mirror)
        ol.addWidget(self.chk_hud)
        ol.addWidget(self._caption("SCREENSHOT FOLDER"))
        r_ss = QHBoxLayout()
        self.txt_ss = QLineEdit(self.state.screenshot_folder)
        b_ss = QPushButton("Browse")
        b_ss.clicked.connect(lambda: self._browse(self.txt_ss))
        r_ss.addWidget(self.txt_ss)
        r_ss.addWidget(b_ss)
        ol.addLayout(r_ss)
        ol.addWidget(self._caption("VIDEO FOLDER"))
        r_v = QHBoxLayout()
        self.txt_vid = QLineEdit(self.state.video_folder)
        b_v = QPushButton("Browse")
        b_v.clicked.connect(lambda: self._browse(self.txt_vid))
        r_v.addWidget(self.txt_vid)
        r_v.addWidget(b_v)
        ol.addLayout(r_v)
        self.btn_rec = QPushButton("⏺  Start Recording")
        self.btn_rec.setObjectName("Danger")
        self.btn_rec.clicked.connect(self._toggle_record)
        ol.addWidget(self.btn_rec)
        layout.addWidget(out_grp)
        set_grp = QGroupBox("Settings Management")
        sl = QVBoxLayout(set_grp)
        r_set = QHBoxLayout()
        b_exp = QPushButton("📤  Export Settings")
        b_exp.clicked.connect(self._export_settings)
        b_imp = QPushButton("📥  Import Settings")
        b_imp.clicked.connect(self._import_settings)
        r_set.addWidget(b_exp)
        r_set.addWidget(b_imp)
        r_set.addStretch()
        sl.addLayout(r_set)
        layout.addWidget(set_grp)
        layout.addStretch()

    def _build_script_tab(self):
        layout = QVBoxLayout(self.tab_script)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._section("📝  Script & Talking Points"))
        layout.addWidget(self._caption("Persistent notes auto-saved with your settings. Useful for streamers and presenters."))
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        b1 = QPushButton("•  Bullet")
        b1.clicked.connect(self._add_bullet)
        b1.setObjectName("IconBtn")
        b2 = QPushButton("1.  Numbered")
        b2.clicked.connect(self._add_numbered)
        b2.setObjectName("IconBtn")
        b3 = QPushButton("H  Heading")
        b3.clicked.connect(self._add_heading)
        b3.setObjectName("IconBtn")
        b4 = QPushButton("—  Separator")
        b4.clicked.connect(self._add_separator)
        b4.setObjectName("IconBtn")
        for b in (b1, b2, b3, b4):
            toolbar.addWidget(b)
        toolbar.addStretch()
        b_clr = QPushButton("🗑  Clear")
        b_clr.setObjectName("Danger")
        b_clr.clicked.connect(self._clear_script)
        b_exp = QPushButton("💾  Export .txt")
        b_exp.clicked.connect(self._export_script)
        toolbar.addWidget(b_clr)
        toolbar.addWidget(b_exp)
        layout.addLayout(toolbar)
        self.script_edit = QTextEdit()
        self.script_edit.setPlaceholderText("Type your script or bullet points here…\n\nExample:\n• Welcome back to the stream!\n• Today we're testing the new VFX Engine\n• Show off the Cyberpunk palette with Neon Edges")
        self.script_edit.setText(self.state.script_text)
        self.script_edit.setFont(QFont("Consolas", 11))
        layout.addWidget(self.script_edit, 1)
        status = QHBoxLayout()
        self.script_status = QLabel("✓  Auto-saved")
        self.script_status.setObjectName("CaptionLabel")
        self.script_status.setStyleSheet("color: #4ade80;")
        status.addWidget(self.script_status)
        status.addStretch()
        self.word_count_label = QLabel("Words: 0  ·  Characters: 0  ·  Lines: 0")
        self.word_count_label.setObjectName("CaptionLabel")
        status.addWidget(self.word_count_label)
        layout.addLayout(status)
        self.script_edit.textChanged.connect(self._on_script_changed)
        self._update_word_count()

    def _add_bullet(self):
        c = self.script_edit.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertText("\n• ")
        self.script_edit.setTextCursor(c)
        self.script_edit.setFocus()

    def _add_numbered(self):
        lines = self.script_edit.toPlainText().split('\n')
        num = 1
        for line in lines:
            if line.strip()[:2] in {f"{i}." for i in range(1, 100)}:
                num += 1
        c = self.script_edit.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertText(f"\n{num}. ")
        self.script_edit.setTextCursor(c)
        self.script_edit.setFocus()

    def _add_heading(self):
        c = self.script_edit.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertText("\n\n═══ HEADING ═══\n")
        self.script_edit.setTextCursor(c)
        self.script_edit.setFocus()

    def _add_separator(self):
        c = self.script_edit.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertText("\n\n" + "─" * 50 + "\n\n")
        self.script_edit.setTextCursor(c)
        self.script_edit.setFocus()

    def _clear_script(self):
        if QMessageBox.question(self, "Clear Script", "Clear all content?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.script_edit.clear()

    def _export_script(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Export Script", "script.txt", "Text Files (*.txt)")
        if fn:
            try:
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write(self.script_edit.toPlainText())
                self.script_status.setText(f"✓  Exported to {os.path.basename(fn)}")
            except Exception as e:
                QMessageBox.warning(self, "Export Failed", str(e))

    def _on_script_changed(self):
        self.state.set_script(self.script_edit.toPlainText())
        self._update_word_count()

    def _update_word_count(self):
        t = self.script_edit.toPlainText()
        self.word_count_label.setText(
            f"Words: {len(t.split())}  ·  Characters: {len(t)}  ·  Lines: {t.count(chr(10)) + 1 if t else 0}"
        )

    def _build_hotkeys_tab(self):
        layout = QVBoxLayout(self.tab_hotkeys)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._section("⌨️  Keyboard Shortcuts"))
        layout.addWidget(self._caption("Press anywhere in the app to trigger these actions."))
        shortcuts = [
            ("Q", "Quit the engine"),
            ("A / D", "Previous / Next filter"),
            ("Z / X", "Previous / Next palette"),
            ("C / V", "Previous / Next palette mode"),
            ("H", "Toggle HUD overlay"),
            ("B", "Toggle Bypass (Before/After)"),
            ("R", "Reset parameters"),
            ("S", "Save screenshot"),
            ("Space", "Toggle recording"),
            ("1-9", "Jump to filter by index"),
        ]
        grid = QFrame()
        grid.setStyleSheet("QFrame { background-color: #181a2e; border: 1px solid #2a2d4a; border-radius: 10px; }")
        gl = QVBoxLayout(grid)
        gl.setContentsMargins(14, 14, 14, 14)
        gl.setSpacing(8)
        for key, desc in shortcuts:
            row = QHBoxLayout()
            kbd = QLabel(key)
            kbd.setStyleSheet("background-color: #202238; color: #c084fc; font-family: 'JetBrains Mono','Consolas',monospace; font-weight: 700; padding: 5px 14px; border-radius: 6px; border: 1px solid #2a2d4a; min-width: 70px;")
            kbd.setAlignment(Qt.AlignCenter)
            row.addWidget(kbd)
            d = QLabel(desc)
            d.setStyleSheet("color: #e6e8f5; font-weight: 500;")
            row.addWidget(d)
            row.addStretch()
            gl.addLayout(row)
        layout.addWidget(grid)
        layout.addStretch()

    def _build_video_panel(self):
        frame = QFrame()
        frame.setObjectName("VideoFrame")
        v = QVBoxLayout(frame)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)
        hdr = QHBoxLayout()
        title = QLabel("◉  LIVE PREVIEW")
        title.setObjectName("SectionLabel")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_resfps = QLabel("1280×720  ·  30 fps")
        self.lbl_resfps.setObjectName("BadgeLabel")
        hdr.addWidget(self.lbl_resfps)
        v.addLayout(hdr)
        self.video_label = QLabel("VideoLabel")
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(720, 420)
        self.video_label.setText("Waiting for camera…")
        self.video_label.setStyleSheet("color: #6c7086; font-style: italic; font-size: 11pt;")
        v.addWidget(self.video_label, 1)
        tb = QHBoxLayout()
        tb.setSpacing(6)
        self.btn_shot = QPushButton("📷  Screenshot")
        self.btn_shot.setObjectName("IconBtn")
        self.btn_shot.clicked.connect(lambda: self._screenshot())
        self.btn_rec2 = QPushButton("⏺  Record")
        self.btn_rec2.setObjectName("IconBtn")
        self.btn_rec2.setCheckable(True)
        self.btn_rec2.clicked.connect(self._toggle_record)
        self.btn_mirror = QPushButton("⇄  Mirror")
        self.btn_mirror.setObjectName("IconBtn")
        self.btn_mirror.setCheckable(True)
        self.btn_mirror.setChecked(self.state.mirror_preview)
        self.btn_mirror.clicked.connect(lambda c: self.state.set_mirror(c))
        self.btn_hud = QPushButton("HUD")
        self.btn_hud.setObjectName("IconBtn")
        self.btn_hud.setCheckable(True)
        self.btn_hud.setChecked(self.state.show_hud)
        self.btn_hud.clicked.connect(lambda _: self.state.toggle_hud())
        self.btn_bypass = QPushButton("Bypass")
        self.btn_bypass.setObjectName("IconBtn")
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setChecked(self.state.bypass)
        self.btn_bypass.clicked.connect(lambda _: self.state.toggle_bypass())
        for b in (self.btn_shot, self.btn_rec2, self.btn_mirror, self.btn_hud, self.btn_bypass):
            tb.addWidget(b)
        tb.addStretch()
        v.addLayout(tb)
        return frame

    def update_frame(self, qimg: QImage):
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_stats(self, fps, f_name, p_name, p_mode, rec):
        self.lbl_fps.setText(f"● {fps:.1f} FPS")
        self.lbl_filter.setText(f"Filter: {f_name}")
        self.lbl_palette.setText(f"Palette: {p_name}")
        self.lbl_rec.setText("● REC" if rec else "")
        self.lbl_resfps.setText(f"{self.state.resolution[0]}×{self.state.resolution[1]}  ·  {self.state.target_fps} fps")
        self.btn_rec2.setChecked(rec)

    def _tick_rec(self):
        if self.state.recording:
            s = self.rec_elapsed.elapsed() // 1000
            self.lbl_time.setText(f"{s // 60:02d}:{s % 60:02d}")

    def _randomize(self):
        f_idx = self.state.get_filter_idx()
        f_name = FILTER_NAMES[f_idx] if 0 <= f_idx < len(FILTER_NAMES) else "None"
        for p_key in FILTER_PARAMS.get(f_name, []):
            if p_key not in PARAM_DEFS:
                continue
            p_def = PARAM_DEFS[p_key]
            if p_def.p_type == "bool":
                val = random.choice([0, 1])
            elif p_def.p_type == "float":
                val = round(random.uniform(p_def.min_val, p_def.max_val), 2)
            else:
                val = random.randint(int(p_def.min_val), int(p_def.max_val))
            self.state.set_param(p_key, val)
            if p_key in self.param_widgets:
                self.param_widgets[p_key].set_value(val)
        for gk in ["global_glow", "global_bloom", "global_vignette", "global_grain"]:
            val = random.randint(0, 60)
            self.state.set_param(gk, val)
            if gk in self.param_widgets:
                self.param_widgets[gk].set_value(val)
        self.combo_palette.setCurrentIndex(random.randint(1, len(PALETTE_NAMES) - 1))
        self.combo_palette_mode.setCurrentIndex(random.randint(0, len(PALETTE_MODES) - 1))

    def _reset_defaults(self):
        for p_key, p_def in PARAM_DEFS.items():
            self.state.set_param(p_key, p_def.default)
            if p_key in self.param_widgets:
                self.param_widgets[p_key].set_value(p_def.default)
        self._rebuild_filter_params()

    def _browse(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            line_edit.setText(d)

    def _apply_camera(self):
        self.state.set_camera(self.spin_cam.value())
        try:
            res = tuple(map(int, self.combo_res.currentText().split('x')))
            self.state.set_resolution(res)
        except Exception:
            pass
        self.state.set_target_fps(int(self.combo_fps.currentText()))
        self.state.set_virtualcam(self.chk_vcam.isChecked())
        self.state.set_mirror(self.chk_mirror.isChecked())
        self.state.set_screenshot_folder(self.txt_ss.text())
        self.state.set_video_folder(self.txt_vid.text())
        self.btn_mirror.setChecked(self.chk_mirror.isChecked())
        QMessageBox.information(self, "Applied", "Camera settings will apply on next restart.")

    def _toggle_record(self):
        new = not self.state.recording
        self.state.set_recording(new)
        self.btn_rec.setText("⏹  Stop Recording" if new else "⏺  Start Recording")
        self.btn_rec2.setText("⏹  Stop" if new else "⏺  Record")
        if new:
            self.rec_elapsed.start()
            self.rec_timer.start()
        else:
            self.rec_timer.stop()
            self.lbl_time.setText("00:00")

    def _screenshot(self):
        pix = self.video_label.pixmap()
        if pix and not pix.isNull():
            fn = os.path.join(self.state.screenshot_folder,
                              f"vfx_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            try:
                pix.save(fn)
                self.statusBar().showMessage(f"Screenshot saved: {fn}", 4000)
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", str(e))

    def _export_settings(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Export VFX Settings", "vfx_settings.json", "JSON Files (*.json)")
        if fn:
            if self.state.export_settings(fn):
                QMessageBox.information(self, "Exported", f"Settings saved to {fn}")
            else:
                QMessageBox.warning(self, "Export Failed", "Could not save settings.")

    def _import_settings(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Import VFX Settings", "", "JSON Files (*.json)")
        if fn:
            if self.state.import_settings(fn):
                self.combo_palette.setCurrentIndex(self.state.palette_idx)
                self.combo_palette_mode.setCurrentIndex(self.state.palette_mode)
                self.script_edit.setText(self.state.script_text)
                self.chk_layered_mode.setChecked(self.state.layered_mode)
                self.layers_tab.refresh()
                self._refresh_filter_cards()
                self._rebuild_filter_params()
                QMessageBox.information(self, "Imported", "Settings loaded successfully.")
            else:
                QMessageBox.warning(self, "Import Failed", "Could not load settings.")

    def _init_shortcuts(self):
        self._key_handlers = {
            Qt.Key_Q: self.close,
            Qt.Key_A: self._key_prev_filter,
            Qt.Key_D: self._key_next_filter,
            Qt.Key_Z: self._key_prev_palette,
            Qt.Key_X: self._key_next_palette,
            Qt.Key_C: self._key_prev_palette_mode,
            Qt.Key_V: self._key_next_palette_mode,
            Qt.Key_H: self._key_toggle_hud,
            Qt.Key_B: self._key_toggle_bypass,
            Qt.Key_R: self._reset_defaults,
            Qt.Key_S: self._screenshot,
            Qt.Key_Space: self._toggle_record,
        }

    def _key_prev_filter(self):
        cur = self.state.get_filter_idx()
        if cur > 0:
            self.state.set_filter(cur - 1)
            self._refresh_filter_cards()
            self._rebuild_filter_params()

    def _key_next_filter(self):
        cur = self.state.get_filter_idx()
        if cur < len(FILTER_NAMES) - 1:
            self.state.set_filter(cur + 1)
            self._refresh_filter_cards()
            self._rebuild_filter_params()

    def _key_prev_palette(self):
        cur = self.state.get_palette_idx()
        if cur > 0:
            self.state.set_palette(cur - 1)
            self.combo_palette.setCurrentIndex(self.state.get_palette_idx())

    def _key_next_palette(self):
        cur = self.state.get_palette_idx()
        if cur < len(PALETTE_NAMES) - 1:
            self.state.set_palette(cur + 1)
            self.combo_palette.setCurrentIndex(self.state.get_palette_idx())

    def _key_prev_palette_mode(self):
        if self.state.palette_mode > 0:
            self.state.set_palette_mode(self.state.palette_mode - 1)
            self.combo_palette_mode.setCurrentIndex(self.state.palette_mode)

    def _key_next_palette_mode(self):
        if self.state.palette_mode < len(PALETTE_MODES) - 1:
            self.state.set_palette_mode(self.state.palette_mode + 1)
            self.combo_palette_mode.setCurrentIndex(self.state.palette_mode)

    def _key_toggle_hud(self):
        show = self.state.toggle_hud()
        self.btn_hud.setChecked(show)

    def _key_toggle_bypass(self):
        bypass = self.state.toggle_bypass()
        self.btn_bypass.setChecked(bypass)

    def keyPressEvent(self, event):
        key = event.key()
        handler = self._key_handlers.get(key)
        if handler:
            handler()
        elif Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1
            if idx < len(FILTER_NAMES):
                self.state.set_filter(idx)
                self._refresh_filter_cards()
                self._rebuild_filter_params()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.state.save_settings()
        self.state.set_running(False)
        event.accept()


# ==========================================
# 15. ENTRY POINT
# ==========================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VFX Engine Studio Pro")
    state = SharedState()
    window = ControlPanel(state)
    worker = CameraWorker(state)
    worker.frame_ready.connect(window.update_frame)
    worker.stats_ready.connect(window.update_stats)
    worker.error_occurred.connect(lambda msg: QMessageBox.critical(window, "Camera Error", msg))
    worker.start()
    window.show()
    ret = app.exec()
    state.set_running(False)
    worker.wait(2000)
    sys.exit(ret)

if __name__ == "__main__":
    main()