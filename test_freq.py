import sys
from gluonts.time_feature import get_lags_for_frequency

frequencies = ["Q", "QE", "M", "W", "D", "H", "T", "S"]

print("Testing GluonTS frequencies for lags...")

for freq in frequencies:
    try:
        lags = get_lags_for_frequency(freq_str=freq, num_default_lags=1)
        print(f"Freq '{freq}': OK (Lags: {lags})")
    except Exception as e:
        print(f"Freq '{freq}': FAIL ({e})")
