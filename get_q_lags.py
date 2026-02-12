from gluonts.time_feature import get_lags_for_frequency
try:
    lags = get_lags_for_frequency(freq_str="Q", num_default_lags=1)
    print(f"Q_LAGS={lags}")
except Exception as e:
    print(f"ERROR: {e}")
