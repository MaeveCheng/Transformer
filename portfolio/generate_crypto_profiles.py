#!/usr/bin/env python3

import os

# All symbols from the directory listing
ALL_SYMBOLS = """1000SATSUSDT
1INCHBTC
1INCHUSDT
AAVEBTC
AAVEETH
AAVEUSDT
ACHBTC
ACHUSDT
ADABTC
ADAETH
ADAUSDT
AGIXBTC
AGIXUSDT
ALGOBTC
ALGOETH
ALGOUSDT
ALPHABTC
ALPHAUSDT
ANKRBTC
ANKRUSDT
APEBTC
APEETH
APEUSDT
API3BTC
API3USDT
APTBTC
APTETH
APTUSDT
ARBBTC
ARBETH
ARBTC
ARBUSDT
ARUSDT
ATOMBTC
ATOMETH
ATOMUSDT
AUDIOBTC
AUDIOUSDT
AVAXBTC
AVAXETH
AVAXUSDT
AXSBTC
AXSETH
AXSUSDT
BAKEBTC
BAKEUSDT
BALBTC
BALUSDT
BATBTC
BATETH
BATUSDT
BCHBTC
BCHUSDT
BEAMBTC
BEAMUSDT
BNBBTC
BNBETH
BNBUSDT
BONKUSDT
BTCSTBTC
BTCSTUSDT
BTCUSDT
BUSDUSDT
C98BTC
C98USDT
CAKEBTC
CAKEUSDT
CELOBTC
CELOUSDT
CELRBTC
CELRETH
CELRUSDT
CFXBTC
CFXUSDT
CHZBTC
CHZUSDT
COMPBTC
COMPUSDT
CRVBTC
CRVETH
CRVUSDT
DASHBTC
DASHETH
DASHUSDT
DOGEBTC
DOGEUSDT
DOTBTC
DOTETH
DOTUSDT
DYDXBTC
DYDXUSDT
EGLDBTC
EGLDETH
EGLDUSDT
ENJBTC
ENJETH
ENJUSDT
ENSBTC
ENSUSDT
ETCBTC
ETCETH
ETCUSDT
ETHBTC
ETHUSDT
FETBTC
FETUSDT
FILBTC
FILETH
FILUSDT
FLOKIUSDT
FLOWBTC
FLOWUSDT
FTMBTC
FTMETH
FTMUSDT
FTTBTC
FTTUSDT
GALABTC
GALAETH
GALAUSDT
GRTBTC
GRTETH
GRTUSDT
HBARBTC
HBARUSDT
HNTBTC
HNTUSDT
ICPBTC
ICPETH
ICPUSDT
IMXBTC
IMXUSDT
INJBTC
INJETH
INJUSDT
KAVABTC
KAVAETH
KAVAUSDT
KLAYBTC
KLAYUSDT
LDOBTC
LDOUSDT
LINKBTC
LINKETH
LINKUSDT
LRCBTC
LRCETH
LRCUSDT
LTCBTC
LTCETH
LTCUSDT
LUNABTC
LUNAUSDT
LUNCUSDT
MANABTC
MANAETH
MANAUSDT
MASKUSDT
MATICBTC
MATICETH
MATICUSDT
MEMEUSDT
METISUSDT
MINABTC
MINAUSDT
MKRBTC
MKRUSDT
NEARBTC
NEARETH
NEARUSDT
NEXOBTC
NEXOUSDT
OCEANBTC
OCEANUSDT
ONDOUSDT
ONEBTC
ONEETH
ONEUSDT
OPBTC
OPETH
OPUSDT
ORDIBTC
ORDIUSDT
PAXGBTC
PAXGUSDT
PENDLEBTC
PENDLEUSDT
PEPEUSDT
PIXELBTC
PIXELUSDT
POLBTC
POLETH
POLUSDT
PYTHBTC
PYTHUSDT
QNTBTC
QNTUSDT
RENDERBTC
RENDERUSDT
RONINBTC
RONINUSDT
RPLBTC
RPLUSDT
RUNEBTC
RUNEETH
RUNEUSDT
SANDBTC
SANDETH
SANDUSDT
SEIBTC
SEIUSDT
SHIBUSDT
SNXBTC
SNXETH
SNXUSDT
SOLBTC
SOLETH
SOLUSDT
STORJBTC
STORJETH
STORJUSDT
STRKBTC
STRKUSDT
STXBTC
STXUSDT
SUIBTC
SUIUSDT
SUSHIBTC
SUSHIUSDT
SXPBTC
SXPUSDT
TAOBTC
TAOUSDT
THETABTC
THETAETH
THETAUSDT
TIABTC
TIAUSDT
TONBTC
TONUSDT
TRXBTC
TRXETH
TRXUSDT
TUSDBTC
TUSDETH
TUSDUSDT
TWTBTC
TWTUSDT
UNIBTC
UNIETH
UNIUSDT
USDCUSDT
USDPUSDT
USTCUSDT
VETBTC
VETETH
VETUSDT
WAVESBTC
WAVESETH
WAVESUSDT
WBTCBTC
WBTCETH
WBTCUSDT
WIFBTC
WIFUSDT
WLDBTC
WLDUSDT
XLMBTC
XLMETH
XLMUSDT
XMRBTC
XMRETH
XMRUSDT
XRPBTC
XRPETH
XRPUSDT
YFIBTC
YFIUSDT
ZECBTC
ZECETH
ZECUSDT
ZILBTC
ZILETH
ZILUSDT
ZKBTC
ZKUSDT""".strip().split('\n')

# Time intervals configuration
time_intervals = [
    ("5m", 1024, 4, 6),
    ("10m", 1024, 9, 11),
    ("30m", 1024, 29, 31),
    ("1h", 1024, 59, 61),
    ("3h", 1024, 179, 181),
    ("6h", 4096, 359, 361),
    ("12h", 4096, 719, 721),
    ("1d", 16384, 1439, 1441),
    ("5d", 65536, 7199, 7201),
    ("10d", 65536, 14399, 14401),
    ("20d", 65536, 28799, 28801)
]

profiles = []

for symbol in ALL_SYMBOLS:
    profiles.append(f"\n    // {symbol}")
    for interval, lookback, start, end in time_intervals:
        profile = f'    {{ name: "{symbol}-{interval}", exponential_lookback: {lookback}, predict_start: {start}, predict_end: {end}, train_folder: "/workspace/Metadata/crypto_train/{symbol}"}}'
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
with open('/workspace/Maeve/Transformer_meta/crypto_profiles_generated.json5', 'w') as f:
    f.write(json5_content)

print(f"Generated {len(ALL_SYMBOLS)} symbols × {len(time_intervals)} intervals = {len(ALL_SYMBOLS) * len(time_intervals)} profiles")
print("File saved to: /workspace/Maeve/Transformer_meta/crypto_profiles_generated.json5")