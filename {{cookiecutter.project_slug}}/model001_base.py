#!/usr/bin/env python3
# region imports

# pylint: disable=unused-import

import functools  # noqa: F401
import pathlib  # noqa: F401
import random  # noqa: F401

import albumentations as A  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import sklearn.datasets  # noqa: F401
import sklearn.ensemble  # noqa: F401
import sklearn.linear_model  # noqa: F401
import sklearn.metrics  # noqa: F401
import sklearn.model_selection  # noqa: F401
import sklearn.neighbors  # noqa: F401
import tensorflow as tf  # noqa: F401
import tensorflow_addons as tfa  # noqa: F401

import _data
import pytoolkit as tk

# endregion

num_classes = 10
train_shape = (320, 320, 3)
predict_shape = (480, 480, 3)
batch_size = 16
nfold = 5
split_seed = 1
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


def create_model():
    return tk.pipeline.KerasModel(
        create_network_fn=create_network,
        score_fn=score,
        nfold=nfold,
        models_dir=models_dir,
        train_data_loader=MyDataLoader(mode="train"),
        refine_data_loader=MyDataLoader(mode="refine"),
        val_data_loader=MyDataLoader(mode="test"),
        epochs=1800,
        refine_epochs=10,
        base_models_dir=None,
        callbacks=[tk.callbacks.CosineAnnealing()],
        on_batch_fn=tta,
    )


# region data/score


def load_check_data():
    dataset = _data.load_check_data()
    return dataset


def load_train_data():
    dataset = _data.load_train_data()
    return dataset


def load_test_data():
    dataset = _data.load_test_data()
    return dataset


def score(
    y_true: tk.data.LabelsType, y_pred: tk.models.ModelIOType
) -> tk.evaluations.EvalsType:
    return tk.evaluations.evaluate_classification(y_true, y_pred)


# endregion

# region commands


@app.command(logfile=False)
def check():  # utility
    create_model().check(load_check_data())


@app.command(logfile=False)
def check_data():  # utility
    check_set = load_check_data()
    it = create_model().train_data_loader.iter(check_set, shuffle=True)
    i = 0
    for X_batch, y_batch in it.ds.take(256 // batch_size):
        for X_i, y_i in zip(X_batch.numpy(), y_batch):
            save_path = models_dir / "check-data" / f"{i:03d}.jpg"
            i += 1
            img = np.uint8((X_i + 1) * 127.5)  # tk.ndimage.preprocess_tf()の逆変換
            tk.ndimage.save(save_path, img)
            logger.info(f"{save_path}: y={y_i}")


@app.command(logfile=False)
def migrate():  # utility
    create_model().load().save()


@app.command(distribute_strategy_fn=tf.distribute.MirroredStrategy)
def train_one():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model()
    model.skip_folds = list(range(1, nfold))
    evals = model.cv(train_set, folds)
    tk.notifications.post_evals(evals)


@app.command(distribute_strategy_fn=tf.distribute.MirroredStrategy, then="validate")
def train():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model()
    evals = model.cv(train_set, folds)
    tk.notifications.post_evals(evals)


@app.command(distribute_strategy_fn=tf.distribute.MirroredStrategy, then="predict")
def validate():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model().load()
    pred = model.predict_oof(train_set, folds)
    if tk.hvd.is_master():
        tk.utils.dump(pred, models_dir / "pred_train.pkl")
        tk.notifications.post_evals(score(train_set.labels, pred))


@app.command(distribute_strategy_fn=tf.distribute.MirroredStrategy)
def predict():
    test_set = load_test_data()
    model = create_model().load()
    pred_list = model.predict_all(test_set)
    pred = np.mean(pred_list, axis=0)
    if tk.hvd.is_master():
        tk.utils.dump(pred_list, models_dir / "pred_test.pkl")
        _data.save_prediction(models_dir, test_set, pred)


# endregion


def create_network():
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tf.keras.layers.BatchNormalization,
        gamma_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    act = functools.partial(tf.keras.layers.Activation, "relu")

    def blocks(filters, count, down=True):
        def layers(x):
            if down:
                in_filters = x.shape[-1]
                g = conv2d(in_filters // 8)(x)
                g = bn()(g)
                g = act()(g)
                g = conv2d(in_filters, use_bias=True, activation="sigmoid")(g)
                x = tf.keras.layers.multiply([x, g])
                x = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
                x = tk.layers.BlurPooling2D(taps=4)(x)
                x = conv2d(filters)(x)
                x = bn()(x)
            for _ in range(count):
                sc = x
                x = conv2d(filters)(x)
                x = bn()(x)
                x = act()(x)
                x = conv2d(filters)(x)
                # resblockのadd前だけgammaの初期値を0にする。 <https://arxiv.org/abs/1812.01187>
                x = bn(gamma_initializer="zeros")(x)
                x = tf.keras.layers.add([sc, x])
            x = bn()(x)
            x = act()(x)
            return x

        return layers

    inputs = x = tf.keras.layers.Input((None, None, 3))
    x = tf.keras.layers.concatenate(
        [
            conv2d(16, kernel_size=2, strides=2)(x),
            conv2d(16, kernel_size=4, strides=2)(x),
            conv2d(16, kernel_size=6, strides=2)(x),
            conv2d(16, kernel_size=8, strides=2)(x),
        ]
    )  # 1/2
    x = bn()(x)
    x = act()(x)
    x = blocks(128, 4)(x)  # 1/4
    x = blocks(256, 4)(x)  # 1/8
    x = blocks(512, 4)(x)  # 1/16
    x = blocks(512, 4)(x)  # 1/32
    x = tk.layers.GeMPooling2D()(x)
    x = tf.keras.layers.Dense(
        num_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    learning_rate = 1e-3 * batch_size * tk.hvd.size() * app.num_replicas_in_sync
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True
    )

    def loss(y_true, logits):
        return tk.losses.categorical_crossentropy(
            y_true, logits, from_logits=True, label_smoothing=0.2
        )

    tk.models.compile(model, optimizer, loss, ["acc"])

    x = tf.keras.layers.Activation("softmax")(x)
    prediction_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model, prediction_model


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, mode):
        super().__init__(
            batch_size=batch_size, data_per_sample=2 if mode == "train" else 1,
        )
        self.mode = mode
        if self.mode == "train":
            self.aug1 = A.Compose(
                [
                    tk.image.RandomTransform(
                        size=train_shape[:2],
                        base_scale=predict_shape[0] / train_shape[0],
                    ),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
        elif self.mode == "refine":
            self.aug1 = tk.image.RandomTransform.create_refine(size=predict_shape[:2])
            self.aug2 = None
        else:
            self.aug1 = tk.image.Resize(width=predict_shape[1], height=predict_shape[0])
            self.aug2 = None

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        X = tk.ndimage.load(X)
        X = self.aug1(image=X)["image"]
        y = tf.keras.utils.to_categorical(y, num_classes) if y is not None else None
        return X, y

    def get_sample(self, data: list) -> tuple:
        if self.mode == "train":
            sample1, sample2 = data
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta")
            X = self.aug2(image=X)["image"]
        else:
            X, y = super().get_sample(data)
        X = tk.ndimage.preprocess_tf(X)
        return X, y


def tta(model: tf.keras.models.Model, X_batch: np.ndarray):
    pred_list = tk.models.predict_on_batch_augmented(
        model,
        X_batch,
        flip=(False, True),
        crop_size=(5, 5),
        padding_size=(16, 16),
        padding_mode="edge",
    )
    return np.mean(pred_list, axis=0)


if __name__ == "__main__":
    # app.run(default="train_one")
    app.run(default="train")
