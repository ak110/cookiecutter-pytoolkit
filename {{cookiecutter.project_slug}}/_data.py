"""データの読み書きなど"""
import pathlib

import numpy as np
import pandas as pd

import pytoolkit as tk

data_dir = pathlib.Path(f"data")
cache_dir = pathlib.Path(f"cache")
logger = tk.log.get(__name__)


def load_data():
    """訓練データ・テストデータの読み込み"""
    return load_train_data(), load_test_data()


@tk.cache.memorize(cache_dir)
def load_train_data():
    """訓練データの読み込み"""
    X_train = None  # TODO
    y_train = None  # TODO
    return tk.data.Dataset(X_train, y_train)


@tk.cache.memorize(cache_dir)
def load_test_data():
    """テストデータの読み込み"""
    X_test = None  # TODO
    return tk.data.Dataset(X_test)


def save_oofp(models_dir, train_set, pred):
    """訓練データのout-of-fold predictionsの保存と評価"""
    if tk.hvd.is_master():
        tk.utils.dump(pred, models_dir / "pred_train.pkl")

        evals = tk.evaluations.evaluate_classification(train_set.labels, pred)
        tk.notifications.post_evals(evals)


def save_prediction(models_dir, test_set, pred):
    """テストデータの予測結果の保存"""
    if tk.hvd.is_master():
        tk.utils.dump(pred, models_dir / "pred_test.pkl")

        df = pd.DataFrame()
        df["id"] = np.arange(1, len(test_set) + 1)
        df["y"] = pred.argmax(axis=-1)
        df.to_csv(models_dir / "submission.csv", index=False)
