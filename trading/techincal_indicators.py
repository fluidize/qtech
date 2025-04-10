import pandas as pd

def calculate_adx(context, period=14):
    high = context["High"]
    low = context["Low"]
    close = context["Close"]

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.Series([max(a, b, c) for a, b, c in zip(tr1, tr2, tr3)])

    # Calculate +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=context.index)
    minus_dm = pd.Series(0.0, index=context.index)

    plus_dm[up_move > down_move] = up_move[up_move > down_move]
    minus_dm[down_move > up_move] = down_move[down_move > up_move]

    # Calculate smoothed TR, +DM, and -DM
    smoothed_tr = tr.rolling(window=period).mean()
    smoothed_plus_dm = plus_dm.rolling(window=period).mean()
    smoothed_minus_dm = minus_dm.rolling(window=period).mean()

    # Calculate +DI and -DI
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calculate ADX (smoothed DX)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


def calculate_rsi(context, period=14):
    delta_p = context["Close"].diff()
    gain = delta_p.where(delta_p > 0, 0)
    loss = -delta_p.where(delta_p < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calculate_macd(context, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = context["Close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = context["Close"].ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(context, period=14):
    high = context["High"]
    low = context["Low"]
    close = context["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.Series([max(a, b, c) for a, b, c in zip(tr1, tr2, tr3)])

    atr = tr.rolling(window=period).mean()
    return atr


def calculate_supertrend(context, period=14):
    atr = calculate_atr(context, period=period)
    multiplier = 3.0

    hl2 = (context["High"] + context["Low"]) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)

    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()
    supertrend = pd.Series(index=context.index, dtype=float)

    supertrend.iloc[0] = final_upperband.iloc[0]

    close = context["Close"]
    prev_upperband = final_upperband.shift(1)
    prev_lowerband = final_lowerband.shift(1)
    prev_supertrend = supertrend.shift(1)

    mask_upper = (basic_upperband < prev_upperband) | (close.shift(1) > prev_upperband)
    final_upperband.loc[mask_upper] = basic_upperband.loc[mask_upper]
    final_upperband.loc[~mask_upper] = prev_upperband.loc[~mask_upper]

    mask_lower = (basic_lowerband > prev_lowerband) | (close.shift(1) < prev_lowerband)
    final_lowerband.loc[mask_lower] = basic_lowerband.loc[mask_lower]
    final_lowerband.loc[~mask_lower] = prev_lowerband.loc[~mask_lower]

    mask_supertrend = close > final_upperband
    supertrend.loc[mask_supertrend] = final_lowerband.loc[mask_supertrend]
    supertrend.loc[~mask_supertrend] = final_upperband.loc[~mask_supertrend]
    return supertrend


def calculate_std(context, period=20):
    raw_std = context["Close"].rolling(window=period).std()
    std = (raw_std - raw_std.min()) / (
        raw_std.max() - raw_std.min()
    )  # normalize std or else different symbols will have different thresholds
    return std