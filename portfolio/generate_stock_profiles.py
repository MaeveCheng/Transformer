#!/usr/bin/env python3

import subprocess

# Get all stock symbols from the directory
result = subprocess.run(
    'find /workspace/Metadata/stock_train -maxdepth 1 -type d | grep -v "^/workspace/Metadata/stock_train$" | xargs -I {} basename {} | sort -u',
    shell=True,
    capture_output=True,
    text=True
)

ALL_SYMBOLS = result.stdout.strip().split('\n')

# Time intervals configuration for stocks (10 intervals matching the "A" example)
time_intervals = [
    ("5m", 1024, 4, 6),
    ("10m", 1024, 9, 11),
    ("30m", 1024, 29, 31),
    ("1h", 1024, 59, 61),
    ("3h", 1024, 179, 181),
    ("1d", 4096, 389, 391),
    ("2d", 8192, 779, 781),
    ("5d", 16384, 1949, 1951),
    ("10d", 32768, 3899, 3901),
    ("20d", 65536, 7799, 7801)
]

profiles = []

for symbol in ALL_SYMBOLS:
    profiles.append(f"\n    // {symbol}")
    for interval, lookback, start, end in time_intervals:
        profile = f'    {{ name: "{symbol}-{interval}", exponential_lookback: {lookback}, predict_start: {start}, predict_end: {end}, train_folder: "/workspace/Metadata/stock_train/{symbol}"}}'
        profiles.append(profile)

# Generate the complete JSON5 content
json5_content = """{
  _base: {
    weight: 1.0,
    optimizer_weight_decay: 0.01,
    chunk_size_lines: 20000,
    file_pattern: "*.parquet",
    batch_size: 256,
    seq_length: 1024,
    sequential_samples: 0,
    exponential_factor: 1.5,
    min_stride: 100,
    max_stride: 200,
    normalisation: 0,
    adaptive_stats_db: "none",
    
    // Normalization methods
    price_norm_method: "none",
    volume_norm_method: "none",
    count_norm_method: "none",
    spread_norm_method: "none",
    volume_imbalance_norm_method: "none",
    time_norm_method: "cyclical",
    window_norm_method: "context",
    meta_norm_method: "none",
    
    // Scale factors (all 1.0)
    price_scale_factor: 1.0,
    volume_scale_factor: 1.0,
    count_scale_factor: 1.0,
    spread_scale_factor: 1.0,
    volume_imbalance_scale_factor: 1.0,
    time_scale_factor: 1.0,
    window_scale_factor: 1.0,
    meta_scale_factor: 1.0,
    other_scale_factor: 1.0
  },
  profiles: [""" + ',\n'.join(profiles) + """
  ]
}
"""

# Write to file
with open('/workspace/Maeve/Transformer_meta/stock.json5', 'w') as f:
    f.write(json5_content)

print(f"Generated {len(ALL_SYMBOLS)} symbols × {len(time_intervals)} intervals = {len(ALL_SYMBOLS) * len(time_intervals)} profiles")
print("File saved to: /workspace/Maeve/Transformer_meta/stock.json5")