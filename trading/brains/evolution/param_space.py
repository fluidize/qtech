from dataclasses import dataclass

@dataclass
class ParamSpec:
    """Specifies the space of values that a parameter can take."""
    search_space: tuple[int, int] | tuple[float, float]

#full TA search space for every function and its parameters
registered_param_specs = {
    "sma": {
        "timeperiod": (2, 200)
    },
    "ema": {
        "timeperiod": (2, 200)
    },
    "rsi": {
        "timeperiod": (2, 200)
    },
    "macd": {
        "fastperiod": (2, 100),
        "slowperiod": (2, 200),
        "signalperiod": (2, 50)
    },
    "macd_dema": {
        "fastperiod": (2, 100),
        "slowperiod": (2, 200),
        "signalperiod": (2, 50)
    },
    "bbands": {
        "timeperiod": (2, 200),
        "devup": (1.0, 5.0),
        "devdn": (1.0, 5.0)
    },
    "stoch": {
        "fastk_period": (2, 50),
        "slowk_period": (2, 20),
        "slowd_period": (2, 20)
    },
    "atr": {
        "timeperiod": (2, 200)
    },
    "cci": {
        "timeperiod": (2, 200)
    },
    "adx": {
        "timeperiod": (2, 200)
    },
    "dpo": {
        "timeperiod": (2, 200)
    },
    "dema": {
        "timeperiod": (2, 200)
    },
    "tema": {
        "timeperiod": (2, 200)
    },
    "fisher_transform": {
        "timeperiod": (2, 200)
    },
    "aroon": {
        "timeperiod": (2, 200)
    },
    "awesome_oscillator": {
        "fast_period": (2, 50),
        "slow_period": (2, 200)
    },
    "keltner_channels": {
        "timeperiod": (2, 200),
        "atr_multiplier": (0.1, 5.0)
    },
    "vwap_bands": {
        "timeperiod": (2, 200),
        "stdev_multiplier": (1.0, 5.0)
    },
    "elder_ray": {
        "timeperiod": (2, 200)
    },
    "rvi": {
        "timeperiod": (2, 200)
    },
    "choppiness_index": {
        "timeperiod": (2, 200)
    },
    "mass_index": {
        "timeperiod": (2, 200),
        "ema_period": (2, 50)
    },
    "volume_zone_oscillator": {
        "short_period": (2, 50),
        "long_period": (2, 200)
    },
    "volatility_ratio": {
        "roc_period": (2, 200),
        "atr_period": (2, 200)
    },
    "hurst_exponent": {
        "max_lag": (5, 50)
    },
    "zscore": {
        "timeperiod": (2, 200)
    },
    "volatility": {
        "timeperiod": (2, 200)
    },
    "percent_rank": {
        "timeperiod": (2, 200)
    },
    "historical_volatility": {
        "output_period": (2.0, 365.0)
    },
    "fractal_indicator": {
        "n": (2, 10)
    },
    "donchian_channel": {
        "timeperiod": (2, 200)
    },
    "price_cycle": {
        "cycle_period": (5, 100)
    },
    "stddev": {
        "timeperiod": (2, 200)
    },
    "roc": {
        "timeperiod": (2, 200)
    },
    "mom": {
        "timeperiod": (2, 200)
    },
    "willr": {
        "timeperiod": (2, 200)
    },
    "mfi": {
        "timeperiod": (2, 200)
    },
    "kama": {
        "er_period": (2, 50),
        "fast_period": (2, 10),
        "slow_period": (10, 100)
    },
    "supertrend": {
        "period": (2, 200),
        "multiplier": (1.0, 5.0)
    },
    "tsi": {
        "long_period": (5, 100),
        "short_period": (2, 50),
        "signal_period": (2, 50)
    },
    "cmf": {
        "timeperiod": (2, 200)
    },
    "hma": {
        "timeperiod": (2, 200)
    },
    "wma": {
        "timeperiod": (2, 200)
    },
    "ichimoku": {
        "tenkan_period": (2, 50),
        "kijun_period": (2, 100),
        "senkou_period": (10, 200),
        "chikou_period": (2, 100)
    },
    "ppo": {
        "fast_period": (2, 100),
        "slow_period": (2, 200),
        "signal_period": (2, 50)
    },
    "aobv": {
        "fast_period": (2, 50),
        "slow_period": (2, 200)
    },
    "psar": {
        "acceleration_start": (0.01, 0.1),
        "acceleration_step": (0.01, 0.1),
        "max_acceleration": (0.1, 0.5)
    },
    "kalman_filter": {
        "process_noise": (0.001, 0.1),
        "measurement_noise": (0.001, 0.5)
    }
}
