import os
import pytest
import pandas as pd
import shutil
from datetime import datetime, timedelta

from EDA.utils.bitcoin_parser import BitCoinParser, BitcoinConstants


@pytest.fixture(autouse=True)
def cleanup_data_folder():
    yield
    if os.path.exists(BitcoinConstants.DATA_FOLDER.value):
        shutil.rmtree(BitcoinConstants.DATA_FOLDER.value)


class TestBitCoinParser:
    def test_get_candles_returns_data(self):
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)

        df = BitCoinParser.get_candles(start_time, end_time)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"

    def test_get_and_save_candles_creates_file(self):
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)

        df, path = BitCoinParser.get_and_save_candles(start_time, end_time)

        assert path is not None
        assert os.path.exists(path)
        saved_df = pd.read_csv(path, index_col=0)
        assert len(saved_df) == len(df)
        assert list(saved_df.columns) == ["open", "high", "low", "close", "volume"]

    def test_get_full_history_downloads_multiple_pages(self):
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=10)).timestamp() * 1000)

        df, path = BitCoinParser.get_and_save_full_history(start_time, end_time)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 200
        assert os.path.exists(path)
        assert df.index.is_monotonic_increasing

    def test_invalid_dates_return_empty(self):
        start_time = int(datetime.now().timestamp() * 1000)
        end_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)

        df = BitCoinParser.get_candles(start_time, end_time)
        assert df.empty

    def test_candles_data_types(self):
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)

        df = BitCoinParser.get_candles(start_time, end_time)

        assert df["open"].dtype == float
        assert df["high"].dtype == float
        assert df["low"].dtype == float
        assert df["close"].dtype == float
        assert df["volume"].dtype == float