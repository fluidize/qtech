### SINGLE RETURN INDICATORS ARE EXCLUDED !!!

from dataclasses import dataclass

@dataclass
class ParamSpec:
    """Specifies the space of values that a parameter can take."""
    parameter_name: str
    search_space: tuple[int, int] | tuple[float, float]

@dataclass
class FunctionSpec:
    """Specifies the function and its parameters."""
    function_name: str
    parameters: list[ParamSpec]

registered_param_specs = {
    "sma": FunctionSpec(
        function_name="sma",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "ema": FunctionSpec(
        function_name="ema",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "vwma": FunctionSpec(
        function_name="vwma",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
        ]
    ),
    "rsi": FunctionSpec(
        function_name="rsi",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    # MACD single-return functions
    "macd_macd": FunctionSpec(
        function_name="macd_macd",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 200))
        ]
    ),
    "macd_signal": FunctionSpec(
        function_name="macd_signal",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 200))
        ]
    ),
    "macd_hist": FunctionSpec(
        function_name="macd_hist",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 200))
        ]
    ),
    # MACD DEMA single-return functions
    "macd_dema_macd": FunctionSpec(
        function_name="macd_dema_macd",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 200))
        ]
    ),
    "macd_dema_signal": FunctionSpec(
        function_name="macd_dema_signal",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 200))
        ]
    ),
    "macd_dema_hist": FunctionSpec(
        function_name="macd_dema_hist",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 200))
        ]
    ),
    # Bollinger Bands single-return functions
    "bbands_upper": FunctionSpec(
        function_name="bbands_upper",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="devup", search_space=(1.0, 5.0)),
            ParamSpec(parameter_name="devdn", search_space=(1.0, 5.0))
        ]
    ),
    "bbands_middle": FunctionSpec(
        function_name="bbands_middle",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="devup", search_space=(1.0, 5.0)),
            ParamSpec(parameter_name="devdn", search_space=(1.0, 5.0))
        ]
    ),
    "bbands_lower": FunctionSpec(
        function_name="bbands_lower",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="devup", search_space=(1.0, 5.0)),
            ParamSpec(parameter_name="devdn", search_space=(1.0, 5.0))
        ]
    ),
    # Stochastic single-return functions
    "stoch_k": FunctionSpec(
        function_name="stoch_k",
        parameters=[
            ParamSpec(parameter_name="fastk_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slowk_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slowd_period", search_space=(2, 200))
        ]
    ),
    "stoch_d": FunctionSpec(
        function_name="stoch_d",
        parameters=[
            ParamSpec(parameter_name="fastk_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slowk_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slowd_period", search_space=(2, 200))
        ]
    ),
    "atr": FunctionSpec(
        function_name="atr",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "cci": FunctionSpec(
        function_name="cci",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    # ADX single-return functions
    "adx_adx": FunctionSpec(
        function_name="adx_adx",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "adx_plus_di": FunctionSpec(
        function_name="adx_plus_di",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "adx_minus_di": FunctionSpec(
        function_name="adx_minus_di",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "dpo": FunctionSpec(
        function_name="dpo",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "dema": FunctionSpec(
        function_name="dema",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "tema": FunctionSpec(
        function_name="tema",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "fisher_transform": FunctionSpec(
        function_name="fisher_transform",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    # Aroon single-return functions
    "aroon_up": FunctionSpec(
        function_name="aroon_up",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "aroon_down": FunctionSpec(
        function_name="aroon_down",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "awesome_oscillator": FunctionSpec(
        function_name="awesome_oscillator",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200))
        ]
    ),
    # Keltner Channels single-return functions
    "keltner_upper": FunctionSpec(
        function_name="keltner_upper",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="atr_multiplier", search_space=(0.1, 5.0))
        ]
    ),
    "keltner_middle": FunctionSpec(
        function_name="keltner_middle",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="atr_multiplier", search_space=(0.1, 5.0))
        ]
    ),
    "keltner_lower": FunctionSpec(
        function_name="keltner_lower",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="atr_multiplier", search_space=(0.1, 5.0))
        ]
    ),
    # VWAP Bands single-return functions
    "vwap_bands_upper": FunctionSpec(
        function_name="vwap_bands_upper",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="stdev_multiplier", search_space=(1.0, 5.0))
        ]
    ),
    "vwap_bands_middle": FunctionSpec(
        function_name="vwap_bands_middle",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="stdev_multiplier", search_space=(1.0, 5.0))
        ]
    ),
    "vwap_bands_lower": FunctionSpec(
        function_name="vwap_bands_lower",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="stdev_multiplier", search_space=(1.0, 5.0))
        ]
    ),
    # Elder Ray single-return functions
    "elder_ray_bull": FunctionSpec(
        function_name="elder_ray_bull",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "elder_ray_bear": FunctionSpec(
        function_name="elder_ray_bear",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "rvi": FunctionSpec(
        function_name="rvi",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "choppiness_index": FunctionSpec(
        function_name="choppiness_index",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "mass_index": FunctionSpec(
        function_name="mass_index",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="ema_period", search_space=(2, 200))
        ]
    ),
    "volume_zone_oscillator": FunctionSpec(
        function_name="volume_zone_oscillator",
        parameters=[
            ParamSpec(parameter_name="short_period", search_space=(2, 200)),
            ParamSpec(parameter_name="long_period", search_space=(2, 200))
        ]
    ),
    "volatility_ratio": FunctionSpec(
        function_name="volatility_ratio",
        parameters=[
            ParamSpec(parameter_name="roc_period", search_space=(2, 200)),
            ParamSpec(parameter_name="atr_period", search_space=(2, 200))
        ]
    ),
    "zscore": FunctionSpec(
        function_name="zscore",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "volatility": FunctionSpec(
        function_name="volatility",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "percent_rank": FunctionSpec(
        function_name="percent_rank",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    # Donchian Channel single-return functions
    "donchian_upper": FunctionSpec(
        function_name="donchian_upper",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "donchian_middle": FunctionSpec(
        function_name="donchian_middle",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "donchian_lower": FunctionSpec(
        function_name="donchian_lower",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "price_cycle": FunctionSpec(
        function_name="price_cycle",
        parameters=[ParamSpec(parameter_name="cycle_period", search_space=(2, 200))]
    ),
    "stddev": FunctionSpec(
        function_name="stddev",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "roc": FunctionSpec(
        function_name="roc",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "mom": FunctionSpec(
        function_name="mom",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "willr": FunctionSpec(
        function_name="willr",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "mfi": FunctionSpec(
        function_name="mfi",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "kama": FunctionSpec(
        function_name="kama",
        parameters=[
            ParamSpec(parameter_name="er_period", search_space=(2, 200)),
            ParamSpec(parameter_name="fast_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200))
        ]
    ),
    # SuperTrend single-return functions
    "supertrend_direction": FunctionSpec(
        function_name="supertrend_direction",
        parameters=[
            ParamSpec(parameter_name="period", search_space=(2, 200)),
            ParamSpec(parameter_name="multiplier", search_space=(1.0, 5.0))
        ]
    ),
    "supertrend_line": FunctionSpec(
        function_name="supertrend_line",
        parameters=[
            ParamSpec(parameter_name="period", search_space=(2, 200)),
            ParamSpec(parameter_name="multiplier", search_space=(1.0, 5.0))
        ]
    ),
    # TSI single-return functions
    "tsi_tsi": FunctionSpec(
        function_name="tsi_tsi",
        parameters=[
            ParamSpec(parameter_name="long_period", search_space=(2, 200)),
            ParamSpec(parameter_name="short_period", search_space=(2, 200)),
            ParamSpec(parameter_name="signal_period", search_space=(2, 200))
        ]
    ),
    "tsi_signal": FunctionSpec(
        function_name="tsi_signal",
        parameters=[
            ParamSpec(parameter_name="long_period", search_space=(2, 200)),
            ParamSpec(parameter_name="short_period", search_space=(2, 200)),
            ParamSpec(parameter_name="signal_period", search_space=(2, 200))
        ]
    ),
    "cmf": FunctionSpec(
        function_name="cmf",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "hma": FunctionSpec(
        function_name="hma",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    "wma": FunctionSpec(
        function_name="wma",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))]
    ),
    # Ichimoku single-return functions
    "ichimoku_tenkan": FunctionSpec(
        function_name="ichimoku_tenkan",
        parameters=[
            ParamSpec(parameter_name="tenkan_period", search_space=(2, 200)),
            ParamSpec(parameter_name="kijun_period", search_space=(2, 200)),
            ParamSpec(parameter_name="senkou_period", search_space=(2, 200)),
            ParamSpec(parameter_name="chikou_period", search_space=(2, 200))
        ]
    ),
    "ichimoku_kijun": FunctionSpec(
        function_name="ichimoku_kijun",
        parameters=[
            ParamSpec(parameter_name="tenkan_period", search_space=(2, 200)),
            ParamSpec(parameter_name="kijun_period", search_space=(2, 200)),
            ParamSpec(parameter_name="senkou_period", search_space=(2, 200)),
            ParamSpec(parameter_name="chikou_period", search_space=(2, 200))
        ]
    ),
    "ichimoku_senkou_a": FunctionSpec(
        function_name="ichimoku_senkou_a",
        parameters=[
            ParamSpec(parameter_name="tenkan_period", search_space=(2, 200)),
            ParamSpec(parameter_name="kijun_period", search_space=(2, 200)),
            ParamSpec(parameter_name="senkou_period", search_space=(2, 200)),
            ParamSpec(parameter_name="chikou_period", search_space=(2, 200))
        ]
    ),
    "ichimoku_senkou_b": FunctionSpec(
        function_name="ichimoku_senkou_b",
        parameters=[
            ParamSpec(parameter_name="tenkan_period", search_space=(2, 200)),
            ParamSpec(parameter_name="kijun_period", search_space=(2, 200)),
            ParamSpec(parameter_name="senkou_period", search_space=(2, 200)),
            ParamSpec(parameter_name="chikou_period", search_space=(2, 200))
        ]
    ),
    "ichimoku_chikou": FunctionSpec(
        function_name="ichimoku_chikou",
        parameters=[
            ParamSpec(parameter_name="tenkan_period", search_space=(2, 200)),
            ParamSpec(parameter_name="kijun_period", search_space=(2, 200)),
            ParamSpec(parameter_name="senkou_period", search_space=(2, 200)),
            ParamSpec(parameter_name="chikou_period", search_space=(2, 200))
        ]
    ),
    # PPO single-return functions
    "ppo_ppo": FunctionSpec(
        function_name="ppo_ppo",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200)),
            ParamSpec(parameter_name="signal_period", search_space=(2, 200))
        ]
    ),
    "ppo_signal": FunctionSpec(
        function_name="ppo_signal",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200)),
            ParamSpec(parameter_name="signal_period", search_space=(2, 200))
        ]
    ),
    "ppo_hist": FunctionSpec(
        function_name="ppo_hist",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200)),
            ParamSpec(parameter_name="signal_period", search_space=(2, 200))
        ]
    ),
    # AOBV single-return functions
    "aobv_obv": FunctionSpec(
        function_name="aobv_obv",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200))
        ]
    ),
    "aobv_signal": FunctionSpec(
        function_name="aobv_signal",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 200)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200))
        ]
    )
}
