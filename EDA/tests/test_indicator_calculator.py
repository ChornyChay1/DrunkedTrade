import pytest
import pandas as pd
import numpy as np
from EDA.utils.indicator_сalculator import IndicatorCalculator


class TestIndicatorCalculator:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")

        close = 40000
        closes = []

        for i in range(200):
            change = np.random.normal(0, 100)
            close = close + change
            closes.append(close)

        df = pd.DataFrame({
            "open": closes,
            "high": [c + abs(np.random.normal(0, 50)) for c in closes],
            "low": [c - abs(np.random.normal(0, 50)) for c in closes],
            "close": closes,
            "volume": np.random.uniform(100, 1000, 200)
        }, index=dates)

        return df

    def test_sma(self, sample_df):
        sma_20 = IndicatorCalculator.sma(sample_df, 20)
        assert len(sma_20) == 200
        assert sma_20.isna().sum() == 19
        assert not sma_20.iloc[20:].isna().any()

    def test_ema(self, sample_df):
        ema_20 = IndicatorCalculator.ema(sample_df, 20)
        assert len(ema_20) == 200
        assert ema_20.isna().sum() <= 20

    def test_rsi(self, sample_df):
        rsi = IndicatorCalculator.rsi(sample_df, 14)
        assert len(rsi) == 200
        assert rsi.isna().sum() == 14
        assert rsi.iloc[100] >= 0 and rsi.iloc[100] <= 100

    def test_stoch_rsi(self, sample_df):
        stoch = IndicatorCalculator.stoch_rsi(sample_df, 14, 3, 3)
        assert isinstance(stoch, pd.DataFrame)
        assert list(stoch.columns) == ["stoch_k", "stoch_d"]
        assert stoch["stoch_k"].iloc[100] >= 0 and stoch["stoch_k"].iloc[100] <= 100
        assert stoch["stoch_d"].iloc[100] >= 0 and stoch["stoch_d"].iloc[100] <= 100

    def test_macd(self, sample_df):
        macd = IndicatorCalculator.macd(sample_df)
        assert isinstance(macd, pd.DataFrame)
        assert list(macd.columns) == ["macd", "signal", "histogram"]

    def test_bollinger_bands(self, sample_df):
        bb = IndicatorCalculator.bollinger_bands(sample_df)
        assert isinstance(bb, pd.DataFrame)
        assert list(bb.columns) == ["middle", "upper", "lower", "bandwidth", "percent_b"]

        valid_mask = bb["middle"].notna()
        if valid_mask.any():
            assert (bb.loc[valid_mask, "upper"] >= bb.loc[valid_mask, "middle"]).all()
            assert (bb.loc[valid_mask, "lower"] <= bb.loc[valid_mask, "middle"]).all()

    def test_keltner_channels(self, sample_df):
        kc = IndicatorCalculator.keltner_channels(sample_df)
        assert isinstance(kc, pd.DataFrame)
        assert list(kc.columns) == ["middle", "upper", "lower"]

        valid_mask = kc["middle"].notna()
        if valid_mask.any():
            assert (kc.loc[valid_mask, "upper"] >= kc.loc[valid_mask, "middle"]).all()
            assert (kc.loc[valid_mask, "lower"] <= kc.loc[valid_mask, "middle"]).all()

    def test_donchian_channels(self, sample_df):
        dc = IndicatorCalculator.donchian_channels(sample_df, 20)
        assert isinstance(dc, pd.DataFrame)
        assert list(dc.columns) == ["middle", "upper", "lower"]

        valid_mask = dc["middle"].notna()
        if valid_mask.any():
            assert (dc.loc[valid_mask, "upper"] >= dc.loc[valid_mask, "middle"]).all()
            assert (dc.loc[valid_mask, "lower"] <= dc.loc[valid_mask, "middle"]).all()

    def test_atr(self, sample_df):
        atr = IndicatorCalculator.atr(sample_df, 14)
        assert len(atr) == 200
        assert atr.isna().sum() in [13, 14]
        assert (atr.iloc[20:] >= 0).all()

    def test_adx(self, sample_df):
        adx = IndicatorCalculator.adx(sample_df, 14)
        assert isinstance(adx, pd.DataFrame)
        assert list(adx.columns) == ["plus_di", "minus_di", "adx"]
        assert not adx.isna().all().all()

    def test_ichimoku(self, sample_df):
        ichimoku = IndicatorCalculator.ichimoku(sample_df)
        assert isinstance(ichimoku, pd.DataFrame)
        assert list(ichimoku.columns) == ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span"]

    def test_parabolic_sar(self, sample_df):
        psar = IndicatorCalculator.parabolic_sar(sample_df)
        assert len(psar) == 200
        assert not psar.isna().any()

    def test_vwap(self, sample_df):
        vwap = IndicatorCalculator.vwap(sample_df)
        assert len(vwap) == 200
        assert not vwap.isna().any()

    def test_mfi(self, sample_df):
        mfi = IndicatorCalculator.mfi(sample_df, 14)
        assert len(mfi) == 200
        assert mfi.isna().sum() in [13, 14]
        assert mfi.iloc[100] >= 0 and mfi.iloc[100] <= 100

    def test_obv(self, sample_df):
        obv = IndicatorCalculator.obv(sample_df)
        assert len(obv) == 200
        assert not obv.isna().any()

    def test_chaikin_money_flow(self, sample_df):
        cmf = IndicatorCalculator.chaikin_money_flow(sample_df, 20)
        assert len(cmf) == 200
        assert cmf.isna().sum() in [19, 20]
        assert cmf.iloc[100] >= -1 and cmf.iloc[100] <= 1

    def test_williams_r(self, sample_df):
        wr = IndicatorCalculator.williams_r(sample_df, 14)
        assert len(wr) == 200
        assert wr.isna().sum() in [13, 14]
        assert wr.iloc[100] >= -100 and wr.iloc[100] <= 0

    def test_cci(self, sample_df):
        cci = IndicatorCalculator.cci(sample_df, 20)
        assert len(cci) == 200
        assert cci.isna().sum() == 19

    def test_roc(self, sample_df):
        roc = IndicatorCalculator.roc(sample_df, 12)
        assert len(roc) == 200
        assert roc.isna().sum() == 12

    def test_momentum(self, sample_df):
        momentum = IndicatorCalculator.momentum(sample_df, 10)
        assert len(momentum) == 200
        assert momentum.isna().sum() == 10

    def test_wma(self, sample_df):
        wma_20 = IndicatorCalculator.wma(sample_df, 20)
        assert len(wma_20) == 200
        assert wma_20.isna().sum() == 19

    def test_hma(self, sample_df):
        hma_20 = IndicatorCalculator.hma(sample_df, 20)
        assert len(hma_20) == 200
        assert hma_20.isna().sum() > 0

    def test_add_all_indicators(self, sample_df):
        result = IndicatorCalculator.add_all_indicators(sample_df)

        expected_columns = [
            "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "wma_20", "hma_20",
            "rsi_14", "stoch_k", "stoch_d", "williams_r", "cci_20", "mfi_14",
            "macd", "macd_signal", "macd_histogram",
            "bb_middle", "bb_upper", "bb_lower", "bb_bandwidth", "bb_percent_b",
            "kc_middle", "kc_upper", "kc_lower",
            "dc_middle", "dc_upper", "dc_lower",
            "atr_14",
            "plus_di", "minus_di", "adx",
            "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span",
            "parabolic_sar",
            "vwap", "obv", "cmf_20",
            "roc_12", "momentum_10"
        ]

        for col in expected_columns:
            assert col in result.columns

        assert len(result) == 200

    def test_add_multi_timeframe_indicators(self, sample_df):
        result = IndicatorCalculator.add_multi_timeframe_indicators(sample_df, sample_df, sample_df)

        assert isinstance(result, dict)
        assert list(result.keys()) == ["1h", "4h", "1d"]
        assert all(isinstance(df, pd.DataFrame) for df in result.values())

    def test_volume_profile(self, sample_df):
        profile = IndicatorCalculator.volume_profile(sample_df, bins=10)
        assert isinstance(profile, dict)
        assert len(profile) == 10
        assert all(isinstance(v, (int, float)) for v in profile.values())
