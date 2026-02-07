
import pandas as pd
from pandas.tseries.frequencies import to_offset
try:
    from gluonts.time_feature import lag
    print(f"GluonTS version: {lag.__file__}")
except:
    print("GluonTS not found")

print(f"Pandas version: {pd.__version__}")
try:
    off = to_offset("Q")
    print(f"to_offset('Q').name: {off.name}")
except Exception as e:
    print(f"to_offset('Q') failed: {e}")

try:
    lags = lag.get_lags_for_frequency("Q")
    print(f"get_lags('Q') success: {lags}")
except Exception as e:
    print(f"get_lags('Q') failed: {e}")
