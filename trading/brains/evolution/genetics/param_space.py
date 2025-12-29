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
    return_count: int

registered_param_specs = {
    "sma": FunctionSpec(
        function_name="sma",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "ema": FunctionSpec(
        function_name="ema",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "rsi": FunctionSpec(
        function_name="rsi",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "macd": FunctionSpec(
        function_name="macd",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 100)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 50))
        ],
        return_count=3
    ),
    "macd_dema": FunctionSpec(
        function_name="macd_dema",
        parameters=[
            ParamSpec(parameter_name="fastperiod", search_space=(2, 100)),
            ParamSpec(parameter_name="slowperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="signalperiod", search_space=(2, 50))
        ],
        return_count=3
    ),
    "bbands": FunctionSpec(
        function_name="bbands",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="devup", search_space=(1.0, 5.0)),
            ParamSpec(parameter_name="devdn", search_space=(1.0, 5.0))
        ],
        return_count=3
    ),
    "stoch": FunctionSpec(
        function_name="stoch",
        parameters=[
            ParamSpec(parameter_name="fastk_period", search_space=(2, 50)),
            ParamSpec(parameter_name="slowk_period", search_space=(2, 20)),
            ParamSpec(parameter_name="slowd_period", search_space=(2, 20))
        ],
        return_count=2
    ),
    "atr": FunctionSpec(
        function_name="atr",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "cci": FunctionSpec(
        function_name="cci",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "adx": FunctionSpec(
        function_name="adx",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=3
    ),
    "dpo": FunctionSpec(
        function_name="dpo",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "dema": FunctionSpec(
        function_name="dema",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "tema": FunctionSpec(
        function_name="tema",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "fisher_transform": FunctionSpec(
        function_name="fisher_transform",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "aroon": FunctionSpec(
        function_name="aroon",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=2
    ),
    "awesome_oscillator": FunctionSpec(
        function_name="awesome_oscillator",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 50)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200))
        ],
        return_count=1
    ),
    "keltner_channels": FunctionSpec(
        function_name="keltner_channels",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="atr_multiplier", search_space=(0.1, 5.0))
        ],
        return_count=3
    ),
    "vwap_bands": FunctionSpec(
        function_name="vwap_bands",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="stdev_multiplier", search_space=(1.0, 5.0))
        ],
        return_count=3
    ),
    "elder_ray": FunctionSpec(
        function_name="elder_ray",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=2
    ),
    "rvi": FunctionSpec(
        function_name="rvi",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "choppiness_index": FunctionSpec(
        function_name="choppiness_index",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "mass_index": FunctionSpec(
        function_name="mass_index",
        parameters=[
            ParamSpec(parameter_name="timeperiod", search_space=(2, 200)),
            ParamSpec(parameter_name="ema_period", search_space=(2, 50))
        ],
        return_count=1
    ),
    "volume_zone_oscillator": FunctionSpec(
        function_name="volume_zone_oscillator",
        parameters=[
            ParamSpec(parameter_name="short_period", search_space=(2, 50)),
            ParamSpec(parameter_name="long_period", search_space=(2, 200))
        ],
        return_count=1
    ),
    "volatility_ratio": FunctionSpec(
        function_name="volatility_ratio",
        parameters=[
            ParamSpec(parameter_name="roc_period", search_space=(2, 200)),
            ParamSpec(parameter_name="atr_period", search_space=(2, 200))
        ],
        return_count=1
    ),
    "zscore": FunctionSpec(
        function_name="zscore",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "volatility": FunctionSpec(
        function_name="volatility",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "percent_rank": FunctionSpec(
        function_name="percent_rank",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "donchian_channel": FunctionSpec(
        function_name="donchian_channel",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=3
    ),
    "price_cycle": FunctionSpec(
        function_name="price_cycle",
        parameters=[ParamSpec(parameter_name="cycle_period", search_space=(5, 100))],
        return_count=1
    ),
    "stddev": FunctionSpec(
        function_name="stddev",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "roc": FunctionSpec(
        function_name="roc",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "mom": FunctionSpec(
        function_name="mom",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "willr": FunctionSpec(
        function_name="willr",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "mfi": FunctionSpec(
        function_name="mfi",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "kama": FunctionSpec(
        function_name="kama",
        parameters=[
            ParamSpec(parameter_name="er_period", search_space=(2, 50)),
            ParamSpec(parameter_name="fast_period", search_space=(2, 10)),
            ParamSpec(parameter_name="slow_period", search_space=(10, 100))
        ],
        return_count=1
    ),
    "supertrend": FunctionSpec(
        function_name="supertrend",
        parameters=[
            ParamSpec(parameter_name="period", search_space=(2, 200)),
            ParamSpec(parameter_name="multiplier", search_space=(1.0, 5.0))
        ],
        return_count=2
    ),
    "tsi": FunctionSpec(
        function_name="tsi",
        parameters=[
            ParamSpec(parameter_name="long_period", search_space=(5, 100)),
            ParamSpec(parameter_name="short_period", search_space=(2, 50)),
            ParamSpec(parameter_name="signal_period", search_space=(2, 50))
        ],
        return_count=2
    ),
    "cmf": FunctionSpec(
        function_name="cmf",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "hma": FunctionSpec(
        function_name="hma",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "wma": FunctionSpec(
        function_name="wma",
        parameters=[ParamSpec(parameter_name="timeperiod", search_space=(2, 200))],
        return_count=1
    ),
    "ichimoku": FunctionSpec(
        function_name="ichimoku",
        parameters=[
            ParamSpec(parameter_name="tenkan_period", search_space=(2, 50)),
            ParamSpec(parameter_name="kijun_period", search_space=(2, 100)),
            ParamSpec(parameter_name="senkou_period", search_space=(10, 200)),
            ParamSpec(parameter_name="chikou_period", search_space=(2, 100))
        ],
        return_count=5
    ),
    "ppo": FunctionSpec(
        function_name="ppo",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 100)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200)),
            ParamSpec(parameter_name="signal_period", search_space=(2, 50))
        ],
        return_count=3
    ),
    "aobv": FunctionSpec(
        function_name="aobv",
        parameters=[
            ParamSpec(parameter_name="fast_period", search_space=(2, 50)),
            ParamSpec(parameter_name="slow_period", search_space=(2, 200))
        ],
        return_count=2
    )
}
