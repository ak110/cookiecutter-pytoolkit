"""データの読み書きなど"""
import pathlib

import numpy as np
import pandas as pd

import pytoolkit as tk

data_dir = pathlib.Path("data")
cache_dir = pathlib.Path("cache")
logger = tk.log.get(__name__)


# @tk.cache.memoize(cache_dir)
def load_check_data():
    """チェック用データの読み込み"""
    return load_train_data().slice(list(range(32)))


@tk.cache.memoize(cache_dir)
def load_train_data():
    """訓練データの読み込み"""
    X_train = None  # TODO
    y_train = None  # TODO
    return tk.data.Dataset(X_train, y_train)


@tk.cache.memoize(cache_dir)
def load_test_data():
    """テストデータの読み込み"""
    X_test = None  # TODO
    return tk.data.Dataset(X_test)


def save_prediction(models_dir, test_set, pred):
    """テストデータの予測結果の保存"""
    df = pd.DataFrame()
    df["id"] = np.arange(1, len(test_set) + 1)
    df["y"] = pred.argmax(axis=-1)
    df.to_csv(models_dir / "submission.csv", index=False)
