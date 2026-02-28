import pandas as pd
import numpy as np
from EDA.utils.bitcoin_parser import BitCoinParser


class IndicatorCalculator:
    @staticmethod
    def sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
        return df[column].rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
        return df[column].ewm(span=period, min_periods=period, adjust=False).mean()

    @staticmethod
    def wma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
        weights = np.arange(1, period + 1)
        return df[column].rolling(window=period, min_periods=period).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == period else np.nan, raw=True
        )

    @staticmethod
    def hma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))
        wma_half = IndicatorCalculator.wma(df, half_length, column)
        wma_full = IndicatorCalculator.wma(df, period, column)
        wma_diff = 2 * wma_half - wma_full
        return wma_diff.rolling(window=sqrt_length, min_periods=sqrt_length).mean()

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
        delta = df[column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def stoch_rsi(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3,
                  column: str = "close") -> pd.DataFrame:
        rsi = IndicatorCalculator.rsi(df, period, column)

        min_rsi = rsi.rolling(window=period, min_periods=period).min()
        max_rsi = rsi.rolling(window=period, min_periods=period).max()

        stoch = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
        k = stoch.rolling(window=smooth_k, min_periods=smooth_k).mean()
        d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()

        return pd.DataFrame({"stoch_k": k, "stoch_d": d})

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close") -> pd.DataFrame:
        exp1 = df[column].ewm(span=fast, min_periods=fast, adjust=False).mean()
        exp2 = df[column].ewm(span=slow, min_periods=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        })

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2, column: str = "close") -> pd.DataFrame:
        sma = df[column].rolling(window=period, min_periods=period).mean()
        std = df[column].rolling(window=period, min_periods=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        bandwidth = (upper - lower) / sma
        percent_b = (df[column] - lower) / (upper - lower)
        return pd.DataFrame({
            "middle": sma,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b
        })

    @staticmethod
    def keltner_channels(df: pd.DataFrame, period: int = 20, atr_period: int = 10,
                         multiplier: float = 2.0) -> pd.DataFrame:
        ema = IndicatorCalculator.ema(df, period, "close")
        atr = IndicatorCalculator.atr(df, atr_period)
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        return pd.DataFrame({
            "middle": ema,
            "upper": upper,
            "lower": lower
        })

    @staticmethod
    def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        upper = df["high"].rolling(window=period, min_periods=period).max()
        lower = df["low"].rolling(window=period, min_periods=period).min()
        middle = (upper + lower) / 2
        return pd.DataFrame({
            "middle": middle,
            "upper": upper,
            "lower": lower
        })

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period, min_periods=period).mean()

        return pd.DataFrame({
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": adx
        })

    @staticmethod
    def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
        tenkan_sen = (df["high"].rolling(window=tenkan).max() + df["low"].rolling(window=tenkan).min()) / 2
        kijun_sen = (df["high"].rolling(window=kijun).max() + df["low"].rolling(window=kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((df["high"].rolling(window=senkou).max() + df["low"].rolling(window=senkou).min()) / 2).shift(
            kijun)
        chikou_span = df["close"].shift(-kijun)

        return pd.DataFrame({
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span
        })

    @staticmethod
    def parabolic_sar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        length = len(df)
        psar = np.zeros(length)
        psar[0] = close[0]

        bull_trend = True
        af = acceleration
        ep = low[0]
        hp = high[0]

        for i in range(1, length):
            if bull_trend:
                psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
                psar[i] = min(psar[i], low[i - 1], low[i] if i > 1 else low[i - 1])

                if low[i] < psar[i]:
                    bull_trend = False
                    psar[i] = hp
                    af = acceleration
                    ep = low[i]
            else:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                psar[i] = max(psar[i], high[i - 1], high[i] if i > 1 else high[i - 1])

                if high[i] > psar[i]:
                    bull_trend = True
                    psar[i] = ep
                    af = acceleration
                    hp = high[i]

            if bull_trend:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + acceleration, maximum)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)

        return pd.Series(psar, index=df.index)

    @staticmethod
    def volume_profile(df: pd.DataFrame, bins: int = 10) -> dict:
        price_range = df["close"].max() - df["close"].min()
        bin_size = price_range / bins
        profile = {}

        for i in range(bins):
            lower = df["close"].min() + i * bin_size
            upper = lower + bin_size
            volume = df[(df["close"] >= lower) & (df["close"] < upper)]["volume"].sum()
            profile[f"{lower:.2f}-{upper:.2f}"] = volume

        return profile

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        return (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
        negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]

        positive_mf = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv[i] = obv[i - 1] - df["volume"].iloc[i]
            else:
                obv[i] = obv[i - 1]
        return pd.Series(obv, index=df.index)

    @staticmethod
    def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mfv = mfm * df["volume"]
        cmf = mfv.rolling(window=period, min_periods=period).sum() / df["volume"].rolling(window=period,
                                                                                          min_periods=period).sum()
        return cmf

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        highest_high = df["high"].rolling(window=period, min_periods=period).max()
        lowest_low = df["low"].rolling(window=period, min_periods=period).min()
        wr = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)
        return wr

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(window=period, min_periods=period).mean()
        mad = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        cci = (tp - sma) / (0.015 * mad)
        return cci

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12, column: str = "close") -> pd.Series:
        return ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100

    @staticmethod
    def momentum(df: pd.DataFrame, period: int = 10, column: str = "close") -> pd.Series:
        return df[column] - df[column].shift(period)

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        result["sma_20"] = IndicatorCalculator.sma(result, 20)
        result["sma_50"] = IndicatorCalculator.sma(result, 50)
        result["sma_200"] = IndicatorCalculator.sma(result, 200)
        result["ema_12"] = IndicatorCalculator.ema(result, 12)
        result["ema_26"] = IndicatorCalculator.ema(result, 26)
        result["wma_20"] = IndicatorCalculator.wma(result, 20)
        result["hma_20"] = IndicatorCalculator.hma(result, 20)

        result["rsi_14"] = IndicatorCalculator.rsi(result, 14)
        stoch_rsi = IndicatorCalculator.stoch_rsi(result)
        result["stoch_k"] = stoch_rsi["stoch_k"]
        result["stoch_d"] = stoch_rsi["stoch_d"]
        result["williams_r"] = IndicatorCalculator.williams_r(result)
        result["cci_20"] = IndicatorCalculator.cci(result, 20)
        result["mfi_14"] = IndicatorCalculator.mfi(result, 14)

        macd_data = IndicatorCalculator.macd(result)
        result["macd"] = macd_data["macd"]
        result["macd_signal"] = macd_data["signal"]
        result["macd_histogram"] = macd_data["histogram"]

        bb_data = IndicatorCalculator.bollinger_bands(result)
        result["bb_middle"] = bb_data["middle"]
        result["bb_upper"] = bb_data["upper"]
        result["bb_lower"] = bb_data["lower"]
        result["bb_bandwidth"] = bb_data["bandwidth"]
        result["bb_percent_b"] = bb_data["percent_b"]

        kc_data = IndicatorCalculator.keltner_channels(result)
        result["kc_middle"] = kc_data["middle"]
        result["kc_upper"] = kc_data["upper"]
        result["kc_lower"] = kc_data["lower"]

        dc_data = IndicatorCalculator.donchian_channels(result)
        result["dc_middle"] = dc_data["middle"]
        result["dc_upper"] = dc_data["upper"]
        result["dc_lower"] = dc_data["lower"]

        result["atr_14"] = IndicatorCalculator.atr(result, 14)

        adx_data = IndicatorCalculator.adx(result)
        result["plus_di"] = adx_data["plus_di"]
        result["minus_di"] = adx_data["minus_di"]
        result["adx"] = adx_data["adx"]

        ichimoku_data = IndicatorCalculator.ichimoku(result)
        result["tenkan_sen"] = ichimoku_data["tenkan_sen"]
        result["kijun_sen"] = ichimoku_data["kijun_sen"]
        result["senkou_span_a"] = ichimoku_data["senkou_span_a"]
        result["senkou_span_b"] = ichimoku_data["senkou_span_b"]
        result["chikou_span"] = ichimoku_data["chikou_span"]

        result["parabolic_sar"] = IndicatorCalculator.parabolic_sar(result)

        result["vwap"] = IndicatorCalculator.vwap(result)
        result["obv"] = IndicatorCalculator.obv(result)
        result["cmf_20"] = IndicatorCalculator.chaikin_money_flow(result, 20)

        result["roc_12"] = IndicatorCalculator.roc(result, 12)
        result["momentum_10"] = IndicatorCalculator.momentum(result, 10)

        return result

    @staticmethod
    def add_multi_timeframe_indicators(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> dict:
        return {
            "1h": IndicatorCalculator.add_all_indicators(df_1h),
            "4h": IndicatorCalculator.add_all_indicators(df_4h),
            "1d": IndicatorCalculator.add_all_indicators(df_1d)
        }