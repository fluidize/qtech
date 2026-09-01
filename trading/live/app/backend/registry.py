import importlib
import inspect

# Module that holds the pure strategy definitions (see strategies.py).
STRATEGIES_MODULE = "trading.live.app.backend.strategies"


def _strategy_params(func) -> dict:
    """Read default param values straight off the function signature."""
    return {
        p.name: p.default
        for p in inspect.signature(func).parameters.values()
        if p.default is not inspect.Parameter.empty and p.name != "data"
    }


def _is_strategy(obj) -> bool:
    """A strategy is any public function whose first parameter is ``data``."""
    if not inspect.isfunction(obj) or obj.__name__.startswith("_"):
        return False
    params = list(inspect.signature(obj).parameters.values())
    return bool(params) and params[0].name == "data"


def get_strategies() -> dict:
    """Discover code strategies plus any *.pth model strategies in /models."""
    module = importlib.import_module(STRATEGIES_MODULE)
    strategies = {
        name: {"func": obj, "params": _strategy_params(obj)}
        for name, obj in vars(module).items()
        if _is_strategy(obj)
    }

    from .model_wrapper import get_model_strategies

    for name, func in get_model_strategies().items():
        strategies[name] = {"func": func, "params": _strategy_params(func)}
    return strategies


def get_strategy(name: str) -> dict | None:
    """Look up a single strategy by name, or None if it isn't defined."""
    return get_strategies().get(name)
