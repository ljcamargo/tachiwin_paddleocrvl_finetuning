#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import re
import subprocess
from pathlib import Path
import argparse
import time
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image

# Constants for LLM text processing
ABSOLUTE_MIN_SIZE = 10
MIN_CHUNK_SIZE = 20
MAX_CHUNK_CHARS = 600 # Reduced to safe limit for 400px images
MIN_LINES = 1
MAX_LINES = 20
SMALL_SIDE_MIN = 384
SMALL_SIDE_MAX = 1680
ABSOLUTE_MIN_SIZE = 200

EXCLUDE_PATTERNS = ["eng", "esp", "spa", "fra", "deu", "por", "all",]
MIN_PIXELS_PER_CHAR = 170
FONTS = [
    "fonts/Andika-Regular.ttf",
    "fonts/Andika-Bold.ttf",
    "fonts/Andika-BoldItalic.ttf",
    "fonts/Andika-Italic.ttf",
    "fonts/M-PLUS-Rounded-1c-Black.ttf",
    "fonts/arial unicode ms.otf",
    "fonts/arial unicode ms bold.otf",
    "fonts/CharisSIL-Regular.ttf",
    "fonts/CharisSIL-Bold.ttf",
    "fonts/CharisSIL-BoldItalic.ttf",
    "fonts/CharisSIL-Italic.ttf",
    "fonts/DejaVuSans.ttf",
    "fonts/DejaVuSans-Bold.ttf",
    "fonts/DejaVuSans-BoldOblique.ttf",
    "fonts/DejaVuSans-Oblique.ttf",
    "fonts/DejaVuSans-ExtraLight.ttf",
    "fonts/DejaVuSansCondensed.ttf",
    "fonts/DejaVuSansCondensed-Bold.ttf",
    "fonts/DejaVuSansCondensed-BoldOblique.ttf",
    "fonts/DejaVuSansCondensed-Oblique.ttf",
    "fonts/DejaVuSerif.ttf",
    "fonts/DejaVuSerif-Bold.ttf",
    "fonts/DejaVuSerif-BoldItalic.ttf",
    "fonts/DejaVuSerif-Italic.ttf",
    "fonts/DejaVuSerifCondensed.ttf",
    "fonts/DejaVuSerifCondensed-Bold.ttf",
    "fonts/DejaVuSerifCondensed-BoldItalic.ttf",
    "fonts/DejaVuSerifCondensed-Italic.ttf",
    "fonts/GentiumPlus-Regular.ttf",
    "fonts/GentiumPlus-Bold.ttf",
    "fonts/GentiumPlus-BoldItalic.ttf",
    "fonts/GentiumPlus-Italic.ttf",
    "fonts/NotoSans-VariableFont_wdth,wght.ttf",
    "fonts/NotoSans-Italic-VariableFont_wdth,wght.ttf",
    "fonts/NotoSerif-VariableFont_wdth,wght.ttf",
    "fonts/NotoSerif-Italic-VariableFont_wdth,wght.ttf",
    "fonts/MPLUSRounded1c-Regular.ttf",
    "fonts/MPLUSRounded1c-Bold.ttf",
    "fonts/MPLUSRounded1c-Light.ttf",
    "fonts/MPLUSRounded1c-Medium.ttf",
    "fonts/MPLUSRounded1c-Thin.ttf",
    "fonts/MPLUSRounded1c-Black.ttf",
    "fonts/MPLUSRounded1c-ExtraBold.ttf"
]

PARAM_CONFIGS = {
    # Text appearance
    'ptsize':         {'median': 21, 'std': 11.0, 'type': 'int'}, 
    'char_spacing':   {'median': 0.0, 'std': 0.3, 'type': 'float'},
    'line_spacing':   {'median': 0.0, 'std': 3.0, 'type': 'int'}, 
    'uppercase':      {'median': 0, 'std': 0.3, 'type': 'bool'},
    
    # Colors (Grayscale 0-255)
    'background_gray': {'median': 255, 'std': 80, 'type': 'int', 'min': 100, 'max': 255}, # Mostly white/light gray
    'text_gray':       {'median': 0,   'std': 80, 'type': 'int', 'min': 0,   'max': 200}, # Mostly black/dark gray
    'use_color':       {'median': 0,   'std': 0.75, 'type': 'bool'}, # ~25% chance of being True
    'color_variance':  {'median': 50,  'std': 20,   'type': 'int', 'min': 20,  'max': 100}, # Aggressive RGB jitter when used
    
    # Image Geometry
    'short_side':     {'median': 500, 'std': 150, 'type': 'int', 'min': 200, 'max': 800},
    'margin_factor':  {'median': 0.01, 'std': 0.03, 'type': 'float', 'min': 0}, 
    
    # Distortions
    'rotation':       {'median': 0, 'std': 0.5, 'type': 'float'}, 
    'blur':           {'median': 0.2, 'std': 1.0, 'type': 'decimals', 'min': 0.0}, 
    'skew_x':         {'median': 0, 'std': 0.5, 'type': 'float'}, 
    'skew_y':         {'median': 0, 'std': 0.5, 'type': 'float'}, 
    'perspective':    {'median': 0, 'std': 0.1, 'type': 'float'}, 
    
    # New Distortions
    'use_noise':           {'median': 0, 'std': 0.0, 'type': 'bool'}, # ~30% chance? no, bool std is weird in this logic, usually med=0 std=large means flip chance.
                                                                      # Logic: raw > 0.5. If med=0, std=0.4 => 10% chance?
                                                                      # Let's adjust med/std for probability. 
                                                                      # med=0, std=1 -> ~30% > 0.5?
    'noise_amount':        {'median': 0, 'std': 0.5, 'type': 'float', 'min': 0.0},
    
    'use_motion_blur':     {'median': 0, 'std': 0.5, 'type': 'bool'},
    'mb_radius':           {'median': 4, 'std': 2.0, 'type': 'float', 'min': 0.0},
    'mb_sigma':            {'median': 2, 'std': 1.0, 'type': 'float', 'min': 0.1},
    'mb_angle':            {'median': 0, 'std': 90, 'type': 'int', 'min': 0, 'max': 360},
    
    'use_ebc':             {'median': 0, 'std': 0.3, 'type': 'bool'}, # Brightness/Contrast
    'brightness':          {'median': 0, 'std': 10.0, 'type': 'int', 'min': -20, 'max': 20},
    'contrast':            {'median': 0, 'std': 10.0, 'type': 'int', 'min': -20, 'max': 20},
    
    'use_dirty':           {'median': 0, 'std': 0.3, 'type': 'bool'},
    'dirty_attenuate':     {'median': 0.0, 'std': 0.05, 'type': 'float', 'min': 0.0},
    
    'use_banding':         {'median': 0, 'std': 0.0, 'type': 'bool'},
    
    'use_morphology':      {'median': 0, 'std': 0.0, 'type': 'bool'},

    # Old params
    'apply_wave':     {'median': 0, 'std': 0.4, 'type': 'bool'}, 
    'wave_amplitude': {'median': 1.5, 'std': 0.5, 'type': 'float'}, 
    'wave_frequency': {'median': 100, 'std': 50, 'type': 'int'}, 
    'extent_padding': {'median': 25, 'std': 25, 'type': 'int', 'min': 0, 'max': 50},
}

# Probability configurations for choice fields
CHOICE_CONFIGS = {
    # Aspect Ratio Archetypes
    'ar_archetype': {
        'options': ['vertical', 'square', 'horizontal'],
        'weights': [0.2, 0.2, 0.6]
    },
    # Text Alignment
    'alignment': {
        'options': ['West', 'Center', 'East'],
        'weights': [0.60, 0.30, 0.10]
    },
    # New Distortion Choices
    'noise_type': {
        'options': ['Gaussian', 'Impulse', 'Laplacian', 'Poisson', 'Uniform'],
        'weights': [0.3, 0.3, 0.1, 0.1, 0.2]
    },
    'morphology_type': {
        'options': ['Erode', 'Dilate'],
        'weights': [0.5, 0.5]
    },
    'morphology_kernel': {
        'options': ['Diamond', 'Disk', 'Square'],
        'weights': [0.4, 0.4, 0.2]
    }
}

WEIGHTED_RANGES = {
     'vertical':   {'min': 0.7, 'max': 0.8},
     'square':     {'min': 0.8, 'max': 1.2},
     'horizontal': {'min': 1.2, 'max': 2.5}
}

def get_directory_size(path):
    if not os.path.exists(path):
        return 0
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, IOError):
                pass
    return total_size / (1024 * 1024)

def count_csv_rows(csv_path):
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        return sum(1 for _ in reader)

def format_elapsed_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def log_progress(total_rows, current_row, effective_rows, chunk_counter, output_dir, start_time, log_file):
    if not log_file:
        return
    
    if total_rows == 0:
        return
    
    progress_percent = (current_row / total_rows) * 100
    elapsed_time = time.time() - start_time
    formatted_time = format_elapsed_time(elapsed_time)
    output_size = get_directory_size(output_dir)
    
    if progress_percent < 25:
        color = '\033[91m'
    elif progress_percent < 50:
        color = '\033[93m'
    elif progress_percent < 75:
        color = '\033[96m'
    else:
        color = '\033[92m'
    reset = '\033[0m'
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = (
        f"{timestamp} | PROGRESS: {progress_percent:.1f}% "
        f"| Row: {current_row}/{total_rows} "
        f"| Effective: {effective_rows} "
        f"| Chunks: {chunk_counter} "
        f"| Output: {output_size:.1f}MB "
        f"| Elapsed: {formatted_time}\n"
    )
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def log_error(error, log_file):
    if not log_file:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"{timestamp} | ERROR: {error} "
    )
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def select_param_deterministic(seed, index, param_name):
    config = PARAM_CONFIGS[param_name]
    median = config['median']
    std = config['std']
    
    # Hash unique combination for determinism
    h = (seed + index * hash(param_name)) & 0xFFFFFFFF 
    h = h / 0xFFFFFFFF  # Normalize to [0, 1]
    
    # Box-Muller transform for normal distribution
    # We need two independent random numbers. Let's create another hash for the second one.
    h2 = (seed + index * hash(param_name + "_2")) & 0xFFFFFFFF
    h2 = h2 / 0xFFFFFFFF
    
    # Avoid log(0)
    u1 = max(h, 1e-10)
    u2 = h2
    
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    
    raw = median + z * std

    # Clamping Logic
    if 'min' in config:
        raw = max(config['min'], raw)
    if 'max' in config:
        raw = min(config['max'], raw)

    # Type Conversion
    if config['type'] == 'int':
        return int(round(raw))
    elif config['type'] == 'bool':
        return raw > 0.5
    elif config['type'] == 'decimals':
        return max(0.0, round(raw, 2))
    else: # float
        return raw

def select_uniform_deterministic(seed, index, param_name, min_val, max_val):
    # Hash for uniform [0,1]
    h = (seed + index * hash(param_name)) & 0xFFFFFFFF
    u = h / 0xFFFFFFFF
    return min_val + u * (max_val - min_val)

def select_choice_deterministic(seed, index, param_name):
    # Weighted choice
    config = CHOICE_CONFIGS[param_name]
    options = config['options']
    weights = config['weights']
    
    # Normalize weights just in case
    total_weight = sum(weights)
    
    h = (seed + index * hash(param_name)) & 0xFFFFFFFF
    u = (h / 0xFFFFFFFF) * total_weight
    
    current = 0
    for i, w in enumerate(weights):
        current += w
        if u <= current:
            return options[i]
    return options[-1]

def select_font_deterministic(seed, index, fonts):
    if not fonts:
        return None
        
    # Group by family to equalise probability
    # Family is defined as the first part of the name before '-'
    families_map = {}
    for f in fonts:
        fam = f.split('-')[0]
        if fam not in families_map:
            families_map[fam] = []
        families_map[fam].append(f)
    
    sorted_families = sorted(families_map.keys())
    
    if not sorted_families:
        return None
        
    # Select Family
    # Use distinct mixing constants to avoid correlation with other params
    h_family = (seed + index * 0x9E3779B9) & 0xFFFFFFFF # Golden Ratio constant for mixing
    family_idx = h_family % len(sorted_families)
    selected_family = sorted_families[family_idx]
    
    # Select Variant
    variants = families_map[selected_family]
    h_variant = (seed + index * 0x85EBCA6B) & 0xFFFFFFFF # Another mix
    variant_idx = h_variant % len(variants)
    
    return variants[variant_idx]

def clean_markdown(text: str) -> str:
    if not text:
        return text
    
    # Headers (# ## ### etc at line start)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Bold/Italic/Strikethrough
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # Bold+Italic
    text = re.sub(r'___(.+?)___', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
    text = re.sub(r'_([^_\s].*?[^_\s])_', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)  # Strikethrough
    # Inline code
    #text = re.sub(r'`([^`]+)`', r'\1', text)
    # Links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Images ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    # Reference-style links [text][ref] -> text
    text = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', text)
    # Reference definitions [ref]: url
    #text = re.sub(r'^\s*\[[^\]]+\]:\s*.+$', '', text, flags=re.MULTILINE)
    # Code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'~~~[\s\S]*?~~~', '', text)
    # Indented code blocks (4 spaces or tab at line start)
    text = re.sub(r'^(?:    |\t).+$', '', text, flags=re.MULTILINE)
    # Block quotes
    #text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Horizontal rules
    #text = re.sub(r'^[\s]*[-*_]{3,}[\s]*$', '', text, flags=re.MULTILINE)
    # Unordered lists (-, *, + at line start)
    #text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Ordered lists (1. 2. etc at line start)
    #text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Task lists (- [ ] or - [x])
    #ext = re.sub(r'^\s*[-*+]\s*\[[xX\s]\]\s+', '', text, flags=re.MULTILINE)
    # HTML tags
    #text = re.sub(r'<[^>]+>', '', text)
    # Footnotes
    #text = re.sub(r'\[\^[^\]]+\]', '', text)
    # Tables
    #text = re.sub(r'^\s*\|?[\s]*:?-+:?[\s]*\|.*$', '', text, flags=re.MULTILINE)
    #text = re.sub(r'^\s*\|(.+)\|\s*$', r'\1', text, flags=re.MULTILINE)
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
    return text.strip()

def split_text_into_blocks(text, seed):
    """
    Splits text into blocks. 
    Prioritizes natural paragraph structure (newlines).
    Filters out noise (e.g. lines of only dots).
    If a paragraph is too large (likely a full page parsed as one line), falls back to sentence splitting.
    """
    if len(text) < ABSOLUTE_MIN_SIZE:
        return []
        
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by one or more newlines to identify paragraphs
    # Some OCR text might use single \n for line breaks and \n\n for paragraphs, 
    # but often \n is just a break. However, the user wants to "try to split down by paragraphs".
    # We will assume contiguous newlines separate blocks.
    raw_paragraphs = re.split(r'\n+', text)
    
    final_blocks = []
    
    # Helper for randomness
    class SimpleRNG:
        def __init__(self, s): self.state = s
        def randint(self, a, b):
            self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
            return a + (self.state % (b - a + 1))
    rng = SimpleRNG(seed)

    current_block_paragraphs = []
    current_block_len = 0
    target_block_size = rng.randint(200, MAX_CHUNK_CHARS) # Start with a target

    for p in raw_paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # Filter noise: strings of just dots, underscores, whitespaces
        if re.match(r'^[\s\._\-]+$', p):
            continue
            
        # Check for index-like lines: ". . . . ." often > 50% dots
        if len(p) > 10 and (p.count('.') + p.count('_')) > (len(p) * 0.5):
             continue

        p_len = len(p)
        
        # Handling huge single paragraphs
        if p_len > MAX_CHUNK_CHARS:
             # Flush current buffer first
             if current_block_paragraphs:
                 final_blocks.append('\n\n'.join(current_block_paragraphs))
                 current_block_paragraphs = []
                 current_block_len = 0
                 target_block_size = rng.randint(200, MAX_CHUNK_CHARS)

             # Split the huge paragraph by sentences
             segments = re.split(r'(?<=[.!?])\s+', p)
             
             huge_p_chunk = []
             huge_p_len = 0
             # For huge paragraphs, we just fill up to MAX_CHUNK_CHARS greedily or near it
             
             i = 0
             while i < len(segments):
                 seg = segments[i].strip()
                 if not seg: 
                     i+=1 
                     continue
                 
                 if re.match(r'^[\s\._\-]+$', seg):
                     i+=1
                     continue

                 # Very long single segment (unbreakable by sentence split)
                 if len(seg) > MAX_CHUNK_CHARS:
                     words = seg.split()
                     temp_chunk = []
                     for w in words:
                         if len(' '.join(temp_chunk + [w])) > MAX_CHUNK_CHARS:
                             final_blocks.append(' '.join(temp_chunk))
                             temp_chunk = [w]
                         else:
                             temp_chunk.append(w)
                     if temp_chunk:
                          # If this leftover fits into the flow, maybe we could keep it, 
                          # but simplify by flushing it to final_blocks or starting a new huge_p_chunk
                          final_blocks.append(' '.join(temp_chunk))
                     i += 1
                     continue
                 
                 if huge_p_len + len(seg) + 1 > MAX_CHUNK_CHARS:
                      if huge_p_chunk:
                          final_blocks.append('\n'.join(huge_p_chunk))
                      huge_p_chunk = [seg]
                      huge_p_len = len(seg)
                 else:
                      huge_p_chunk.append(seg)
                      huge_p_len += len(seg) + 1
                 i += 1
             
             if huge_p_chunk:
                 final_blocks.append('\n'.join(huge_p_chunk))
             
             continue

        # Standard paragraph accumulation
        # Check if adding this paragraph exceeds LIMIT (MAX) or our random TARGET
        # We enforce MAX hard limit, but TARGET is soft limit to trigger flush
        
        # If adding p exceeds MAX_CHUNK_CHARS, we MUST flush
        if current_block_len + p_len + 2 > MAX_CHUNK_CHARS:
            if current_block_paragraphs:
                final_blocks.append('\n\n'.join(current_block_paragraphs))
            current_block_paragraphs = [p]
            current_block_len = p_len
            target_block_size = rng.randint(200, MAX_CHUNK_CHARS) # New target for new block
        
        # If adding p stays within MAX, check if we reached TARGET
        else:
            # We add it first, THEN check if we should flush logic? 
            # Or we check if CURRENT is already enough?
            # User wants "about 4x more text".
            # Let's add it, then see if we exceeded target.
            
            current_block_paragraphs.append(p)
            current_block_len += p_len + 2
            
            # If we reached our distinct target size for this block, flush it
            # This allows some blocks to be small (if target was small) and some large
            if current_block_len >= target_block_size:
                final_blocks.append('\n\n'.join(current_block_paragraphs))
                current_block_paragraphs = []
                current_block_len = 0
                target_block_size = rng.randint(200, MAX_CHUNK_CHARS)

    # Flush remaining
    if current_block_paragraphs:
        final_blocks.append('\n\n'.join(current_block_paragraphs))

    return final_blocks

def generate_image_with_imagemagick(
    text, text_filepath, font, ptsize, char_spacing, line_spacing, alignment,
    width, height, aspect_ratio, margin_factor,
    rotation, blur, skew_x, skew_y, perspective,
    apply_wave, bg_color_str, text_color_str,
    output_path, log_file=None,
    use_noise=False, noise_type='Gaussian', noise_amount=0,
    use_motion_blur=False, mb_radius=0, mb_sigma=0, mb_angle=0,
    use_ebc=False, brightness=0, contrast=0,
    use_dirty=False, dirty_attenuate=0.1,
    use_banding=False,
    use_morphology=False, morphology_type='Erode', morphology_kernel='Diamond',
    extent_padding=0
):
    
    # Calculate Margins
    margin_x = int(width * margin_factor)
    margin_y = int(height * margin_factor)
    
    # Safe approach: 
    # 1. Define text region size (Canvas Size - 2*Margins).
    # 2. If margins are negative, Text Region > Canvas Size.
    
    text_w = width - 2 * margin_x
    text_h = height - 2 * margin_y
    
    # Ensure text region is positive (sanity check)
    text_w = max(50, text_w)
    text_h = max(50, text_h)
    
    # Convert grays to hex/color string
    bg_color = bg_color_str
    text_color = text_color_str
    
    # Soft Wrap Logic to assist ImageMagick
    if len(text) > 0:
        target_aspect = text_w / text_h
        # Heuristic: C ~ sqrt(K * R * L). Tuning K.
        # 2.0 was approx 0.5 char aspect. 
        # Increasing to 2.5 to make text blocks slightly wider/shorter to reduce vertical white space 
        # or better fit aspect ratios.
        estimated_cols = int(math.sqrt(2.5 * target_aspect * len(text)))
        cols = max(10, estimated_cols)
        
        # Preserve existing paragraphs (newlines)
        wrapped_lines = []
        for paragraph in text.splitlines():
            if paragraph.strip():
                wrapped_lines.append(textwrap.fill(paragraph, width=cols))
            else:
                wrapped_lines.append("") # Preserve empty lines
        
        safe_text = "\n".join(wrapped_lines)
    else:
        safe_text = text

    # Sanitize text for caption:
    # If text starts with @, ImageMagick treats it as a filename.
    if safe_text.startswith("@"):
        safe_text = "\\" + safe_text

    # Basic Command Construction
    cmd = ["convert"]
    
    # 1. Create the text layer using caption (auto-wrap)
    cmd.extend([
        "-background", bg_color,
        "-fill", text_color,
        "-font", font,
        # "-pointsize", str(ptsize), # DISABLED for auto-fit
        "-kerning", str(char_spacing),
        "-interline-spacing", str(line_spacing), 
        "-gravity", alignment, 
        "-size", f"{text_w}x{text_h}",
        f"caption:{safe_text}",
    ])
    
    # 2. Resize/Crop to final canvas size if we played with margins
    # If we had negative margins, text_w > width. 
    # We want to center crop to width x height.
    # If we had positive margins, we have a smaller image that needs padding.
    
    # Determine offset for composition
    # 2. Resize/Crop to final canvas size if we played with margins
    cmd.extend([
        "-gravity", "center", 
        "-extent", f"{width}x{height}" 
    ])
    
    # 3. Geometric Transformations (Rotate, Skew, Perspective)
    # These typically expand the canvas. We will crop back ONCE after all of them.
    
    # Rotate
    if abs(rotation) > 0.1:
        cmd.extend([
            "-rotate", str(rotation),
            "-background", bg_color,
        ])

    # Skew (Shear)
    if abs(skew_x) > 0.1 or abs(skew_y) > 0.1:
        cmd.extend(["-background", bg_color, "-shear", f"{skew_x}x{skew_y}"])

    # Perspective
    if abs(perspective) > 0.01:
        # P = width, height
        d = int(min(width, height) * perspective * 0.2) # Max 20% shift
        if d > 0:
            w, h = width, height
            args = (
                f"0,0 {0+d},{0}  " 
                f"{w},0 {w-d},{0}  "
                f"0,{h} {0},{h}  "
                f"{w},{h} {w},{h}"
            )
            # or randomized? Let's stick to a simple consistent tilt for now based on sign
            # This simulates a photo of a document at an angle
            
            cmd.extend([
                "-virtual-pixel", "background", # Use background color
                "-background", bg_color,
                "-distort", "Perspective", args,
                "-gravity", "center",
                "-extent", f"{width}x{height}"
            ])

    # Noise
    if use_noise:
         if noise_amount > 0:
             cmd.extend(["-attenuate", str(noise_amount)])
         cmd.extend(["+noise", noise_type])
    
    # Dirty Pixels
    if use_dirty and not use_noise:
        cmd.extend(["-attenuate", str(dirty_attenuate), "+noise", "Multiplicative"])

    # Brightness / Contrast
    if use_ebc:
        cmd.extend(["-brightness-contrast", f"{brightness}x{contrast}"])

    # Banding (Scanner streaks)
    if use_banding:
        cmd.extend([
            "(", "-size", f"{width}x{height}", "gradient:", ")",
            "-compose", "multiply",
            "-composite"
        ])

    # Morphology (Erode/Dilate)
    if use_morphology:
        cmd.extend(["-morphology", morphology_type, morphology_kernel])

    # Wave (Legacy/Optional)
    if apply_wave:
         # Need amplitude and frequency. 
         # In original: amp ~ 1.25, freq ~ 300
         # Frequency is length.
         cmd.extend(["-wave", f"2x{width*0.5}"]) # Hardcoded light wave for now

    # Experimental Trim to remove excessive whitespace
    cmd.extend(["-fuzz", "10%", "-trim", "+repage"])

    # ADDED: Padding with -border at the end
    if extent_padding > 0:
        # border adds padding to both sides, so border size is half the total increase
        border_size = extent_padding / 2.0
        cmd.extend([
            "-bordercolor", bg_color,
            "-border", f"{border_size}x{border_size}"
        ])

    # Final Output
    cmd.append(str(output_path))
    str_command = ' '.join(cmd)
    
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
        )
        #print(f"OK Command: {' '.join(cmd)}") # Debug
        try:
            with Image.open(output_path) as img:
                actual_width, actual_height = img.size
                actual_aspect = actual_width / actual_height
                is_too_small = actual_width < ABSOLUTE_MIN_SIZE or actual_height < ABSOLUTE_MIN_SIZE
                is_too_extreme = actual_aspect < 0.25 or actual_aspect > 5
                if is_too_small or is_too_extreme:
                    print(f"!!! POST-RENDER REJECT: {actual_width}x{actual_height} aspect={actual_aspect:.2f}")
                    output_path.unlink()
                else:
                    with open(text_filepath, 'w', encoding='utf-8') as f:
                        f.write(text)
        except Exception as e:
            print(f">>>>Error validating image {output_path}: {e}")
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
        return True
    except Exception as e:
        # Clean and truncate command/error for logging
        clean_cmd = str_command.replace('\n', ' ').replace('\r', ' ')[:500]
        clean_err = str(e).replace('\n', ' ').replace('\r', ' ')[:500]
        err_string = f"Command: {clean_cmd} | Error: {clean_err}"
        
        print(f"NOTOK {err_string}")
        log_error(err_string, log_file)
        
        if hasattr(e, 'stderr') and e.stderr:
             clean_stderr = str(e.stderr).replace('\n', ' ').replace('\r', ' ')[:500]
             print(f"Stderr: {clean_stderr}")
        return False


def process_csv_file(csv_path, output_dir, seed=42, resume_from=0, log_file=None, limit=0, concurrency=8):
    print(f"Processing {csv_path}...")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Concurrency: {concurrency}")
    if limit > 0:
        print(f"Limit: {limit} rows")
    
    if log_file:
        total_rows = count_csv_rows(csv_path)
    else:
        total_rows = 0
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    text_dir = output_path / "text"
    
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_counter = 0
    start_time = time.time()
    last_logged_percent = -1
    
    # Setup for concurrency
    executor = ThreadPoolExecutor(max_workers=concurrency)
    futures = set()
    
    # Function to handle completion update
    completed_chunks = 0
    def check_progress():
        nonlocal completed_chunks, last_logged_percent
        if log_file and total_rows > 0:
            # We estimate progress by rows, but here we are in threaded chunk processing
            # It's harder to map exact chunks to rows for % without tracking.
            # Let's just update based on row loop, but logging actual generation lag might be good.
            # Fallback to existing row-based progress for user feedback, 
            # or update usage of 'chunk_counter' in logs.
            pass

    def valid_size(width, height, text):
        if ABSOLUTE_MIN_SIZE > min(width, height):
            print(">>>>>>>> Side too small, avoiding chunk", min(width, height))
            return False
        area = width * height
        if area < (ABSOLUTE_MIN_SIZE * ABSOLUTE_MIN_SIZE):
            print(">>>>>>>> Area too small, avoiding chunk", area)
            return False
        char_count = len(text)
        if char_count > 0:
            density = area / char_count
            if density < MIN_PIXELS_PER_CHAR:
                print(">>>>>>>> Density too low, avoiding chunk", density)
                return False
        return True

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Check headers (basic validation)
        if not {'id', 'text'}.issubset(set(reader.fieldnames)):
             # Allow looser 'id' or 'text' check if needed, but sticking to generator.py logic
             pass

        row_count = 0
        effective_row_count = 0
        
        for row in reader:
            row_count += 1
            if row_count <= resume_from:
                continue
            
            if limit > 0 and effective_row_count >= limit:
                print(f"Reached limit of {limit} rows. Stopping submission.")
                break

            # Progress Logging (Row Processing Level)
            if log_file and total_rows > 0:
                current_percent = (row_count / total_rows) * 100
                current_percent_rounded = round(current_percent * 10) / 10
                if current_percent_rounded != last_logged_percent and int(current_percent_rounded * 10) % 1 == 0:
                    # Note: This log shows Main Thread progress, not Worker progress.
                    # With a buffer, they should be close.
                    log_progress(total_rows, row_count, effective_row_count, completed_chunks, output_path, start_time, log_file)
                    last_logged_percent = current_percent_rounded

            row_id = row.get('id', f'row_{row_count}')
            lang_code = row.get('language', 'unk')
            text = row.get('text', '')
            
            if any(pattern in row_id for pattern in EXCLUDE_PATTERNS):
                continue
            
            if not text.strip():
                continue
            
            # Fix literal escapes likely present in CSV
            text = text.replace('\\n', '\n').replace('\\r', '\r')
                
            effective_row_count += 1
            # Derive a row-specific seed from main seed + row index
            row_seed = seed + row_count 
            
            # Split into blocks (multi-line chunks)
            text = clean_markdown(text)
            blocks = split_text_into_blocks(text, row_seed)
            
            for i, block_text in enumerate(blocks):
                chunk_counter += 1
                chunk_id = f"{chunk_counter}" # Unique counter for randomization

                # Deterministic Randoms
                selected_font = select_font_deterministic(seed, chunk_counter, FONTS)
                ptsize = select_param_deterministic(seed, chunk_counter, 'ptsize')
                char_spacing = select_param_deterministic(seed, chunk_counter, 'char_spacing')
                uppercase = select_param_deterministic(seed, chunk_counter, 'uppercase')

                # Reflow Text
                reflowed_text = re.sub(r'[ \t]+', ' ', block_text).strip()
                lines = [l for l in reflowed_text.splitlines() if l.strip()]
                if len(lines) == 0:
                    continue
                avg_line_len = sum(len(l) for l in lines) / len(lines)
                line_count = len(lines)
                text_aspect = avg_line_len / line_count if line_count > 0 else 1.0

                """ if line_count > 2:
                    if text_aspect > 0.15:
                        words = reflowed_text.split()
                        if words:
                            total_chars = sum(len(w) for w in words) + len(words)
                            target_width = max(20, int(math.sqrt(total_chars * 1.5)))
                            reflowed_text = textwrap.fill(" ".join(words), width=target_width) """
                
                final_text = reflowed_text.upper() if uppercase else reflowed_text
                margin_factor = select_param_deterministic(seed, chunk_counter, 'margin_factor')
                
                rotation = select_param_deterministic(seed, chunk_counter, 'rotation')
                blur = select_param_deterministic(seed, chunk_counter, 'blur')
                skew_x = select_param_deterministic(seed, chunk_counter, 'skew_x')
                skew_y = select_param_deterministic(seed, chunk_counter, 'skew_y')
                perspective = select_param_deterministic(seed, chunk_counter, 'perspective')
                apply_wave = select_param_deterministic(seed, chunk_counter, 'apply_wave')
                
                line_spacing = select_param_deterministic(seed, chunk_counter, 'line_spacing')
                
                # Color Variations
                bg_gray_base = select_param_deterministic(seed, chunk_counter, 'background_gray')
                text_gray_base = select_param_deterministic(seed, chunk_counter, 'text_gray')
                use_color = select_param_deterministic(seed, chunk_counter, 'use_color')
                color_var = select_param_deterministic(seed, chunk_counter, 'color_variance')
                
                if use_color:
                    # Helper to jitter color
                    def get_jittered_channel(base, var, salt):
                        jitter = select_uniform_deterministic(seed, chunk_counter, salt, -var, var)
                        return int(max(0, min(255, base + jitter)))

                    bg_r = get_jittered_channel(bg_gray_base, color_var, 'bg_r')
                    bg_g = get_jittered_channel(bg_gray_base, color_var, 'bg_g')
                    bg_b = get_jittered_channel(bg_gray_base, color_var, 'bg_b')
                    bg_color_str = f"rgb({bg_r},{bg_g},{bg_b})"
                    
                    txt_r = get_jittered_channel(text_gray_base, color_var, 'txt_r')
                    txt_g = get_jittered_channel(text_gray_base, color_var, 'txt_g')
                    txt_b = get_jittered_channel(text_gray_base, color_var, 'txt_b')
                    text_color_str = f"rgb({txt_r},{txt_g},{txt_b})"
                else:
                    bg_color_str = f"rgb({bg_gray_base},{bg_gray_base},{bg_gray_base})"
                    text_color_str = f"rgb({text_gray_base},{text_gray_base},{text_gray_base})"
                
                # New Distortions Params
                use_noise = select_param_deterministic(seed, chunk_counter, 'use_noise')
                noise_type = select_choice_deterministic(seed, chunk_counter, 'noise_type')
                noise_amount = select_param_deterministic(seed, chunk_counter, 'noise_amount')
                
                use_motion_blur = select_param_deterministic(seed, chunk_counter, 'use_motion_blur')
                mb_radius = select_param_deterministic(seed, chunk_counter, 'mb_radius')
                mb_sigma = select_param_deterministic(seed, chunk_counter, 'mb_sigma')
                mb_angle = select_param_deterministic(seed, chunk_counter, 'mb_angle')
                
                use_ebc = select_param_deterministic(seed, chunk_counter, 'use_ebc')
                brightness = select_param_deterministic(seed, chunk_counter, 'brightness')
                contrast = select_param_deterministic(seed, chunk_counter, 'contrast')
                
                use_dirty = select_param_deterministic(seed, chunk_counter, 'use_dirty')
                dirty_attenuate = select_param_deterministic(seed, chunk_counter, 'dirty_attenuate')
                
                use_banding = select_param_deterministic(seed, chunk_counter, 'use_banding')
                
                use_morphology = select_param_deterministic(seed, chunk_counter, 'use_morphology')
                morphology_type = select_choice_deterministic(seed, chunk_counter, 'morphology_type')
                morphology_kernel = select_choice_deterministic(seed, chunk_counter, 'morphology_kernel')

                extent_padding = int(select_uniform_deterministic(seed, chunk_counter, 'extent_padding', 0, 50))

                def get_aspect_ratio(seed, chunk_counter):
                    archetype = select_choice_deterministic(seed, chunk_counter, 'ar_archetype')
                    ar_range = WEIGHTED_RANGES[archetype]
                    base_ar = select_uniform_deterministic(
                        seed, chunk_counter, f"ar_{archetype}", 
                        ar_range['min'], ar_range['max']
                    )
                    if base_ar >= 1.0:
                        height = max(ABSOLUTE_MIN_SIZE, min(SMALL_SIDE_MAX, short_side))
                        width = int(height * base_ar)
                    else:
                        width = max(ABSOLUTE_MIN_SIZE, min(SMALL_SIDE_MAX, short_side))
                        height = int(width / base_ar)
                    return [base_ar, width, height]

                found_valid = False
                for attempt in range(11): # 0 original + 10 retries
                    current_seed = seed + attempt
                    short_side = select_param_deterministic(current_seed, chunk_counter, 'short_side')
                    [aspect_ratio, width, height] = get_aspect_ratio(current_seed, chunk_counter)
                    if valid_size(width, height, final_text):
                        found_valid = True
                        break
                
                if not found_valid:
                    print(f">>>>>>>> Still Invalid size after 10 retries, skipping chunk {chunk_counter}")
                    continue
                
                # Random Alignment Selection
                alignment_choice = select_choice_deterministic(seed, chunk_counter, 'alignment')
                
                # Filename
                filename_base = f"{lang_code}_{row_id}_{i+1:03d}_{int(aspect_ratio*100)}ar"
                filename_base = re.sub(r'[^\w\-_\.]', '_', filename_base)
                
                # Save Text
                text_filepath = text_dir / f"{filename_base}.txt"
                # Generate Image (Async)
                image_filepath = images_dir / f"{filename_base}.png"
                
                future = executor.submit(
                    generate_image_with_imagemagick,
                    final_text, text_filepath, selected_font, ptsize, char_spacing, line_spacing, alignment_choice,
                    width, height, aspect_ratio, margin_factor,
                    rotation, blur, skew_x, skew_y, perspective,
                    apply_wave, bg_color_str, text_color_str,
                    image_filepath, log_file,
                    use_noise=use_noise, noise_type=noise_type, noise_amount=noise_amount,
                    use_motion_blur=use_motion_blur, mb_radius=mb_radius, mb_sigma=mb_sigma, mb_angle=mb_angle,
                    use_ebc=use_ebc, brightness=brightness, contrast=contrast,
                    use_dirty=use_dirty, dirty_attenuate=dirty_attenuate,
                    use_banding=use_banding,
                    use_morphology=use_morphology, morphology_type=morphology_type, morphology_kernel=morphology_kernel,
                    extent_padding=extent_padding
                )
                futures.add(future)
                
                # Backpressure / Harvest results
                if len(futures) >= concurrency * 3:
                     done, futures = wait(futures, return_when=FIRST_COMPLETED)
                     for f in done:
                         completed_chunks += 1
                         try:
                             f.result() # Check for exceptions
                         except Exception as e:
                             print(f"Task exception: {e}")

            if row_count % 100 == 0:
                print(f"Rows processed: {row_count}. Tasks submitted.")
        
        # Wait for all remaining
        print("Waiting for all tasks to complete...")
        for f in wait(futures)[0]:
             completed_chunks += 1
             try:
                 f.result()
             except Exception as e:
                 print(f"Task exception: {e}")
        
    executor.shutdown()
    print("All done.")

    return chunk_counter

def main():
    parser = argparse.ArgumentParser(description='Generator LLM: Synthetic OCR data for Vision Transformers.')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('output_dir', help='Directory to save text and image files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume-from', type=int, default=0, help='Row to resume from')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--limit', type=int, default=0, help='Max rows to process (0 for unlimited)')
    parser.add_argument('--concurrency', type=int, default=8, help='Number of parallel processes for image generation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        sys.exit(1)
        
    try:
        process_csv_file(args.input_csv, args.output_dir, args.seed, args.resume_from, args.log_file, args.limit, args.concurrency)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
