#!/usr/bin/env python3
import functools
import pathlib
import typing

import albumentations as A
import numpy as np
import tensorflow as tf

import _data
import pytoolkit as tk

num_classes = 10
train_shape = (320, 320, 3)
predict_shape = (480, 480, 3)
batch_size = 16
nfold = 5
split_seed = 2
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    _ = _data.load_data()  # 動作確認用に呼ぶだけ呼んでおく
    create_model().check()


@app.command(then="validate", use_horovod=True)
def train():
    train_set = _data.load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model()
    evals = model.cv(train_set, folds, models_dir)
    tk.notifications.post_evals(evals)


@app.command(then="predict", use_horovod=True)
def validate():
    train_set = _data.load_train_data()
    folds = tk.validation.split(train_set, nfold, stratify=True, split_seed=split_seed)
    model = create_model().load(models_dir)
    pred = model.predict_oof(train_set, folds)
    _data.save_oofp(models_dir, train_set, pred)


@app.command(use_horovod=True)
def predict():
    test_set = _data.load_test_data()
    model = create_model().load(models_dir)
    pred_list = model.predict(test_set)
    pred = np.mean(pred_list, axis=0)
    _data.save_prediction(models_dir, test_set, pred)


def create_model():
    return MyModel(
        train_data_loader=MyDataLoader(mode="train"),
        refine_data_loader=MyDataLoader(mode="refine"),
        val_data_loader=MyDataLoader(mode="test"),
        fit_params={"epochs": 1800, "callbacks": [tk.callbacks.CosineAnnealing()]},
        models_dir=models_dir,
        on_batch_fn=_tta,
        use_horovod=True,
    )


class MyModel(tk.pipeline.KerasModel):
    """KerasModel"""

    def create_network(self) -> tf.keras.models.Model:
        K = tf.keras.backend

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

        def down(filters):
            def layers(x):
                in_filters = K.int_shape(x)[-1]
                g = conv2d(in_filters // 8)(x)
                g = bn()(g)
                g = act()(g)
                g = conv2d(in_filters, use_bias=True, activation="sigmoid")(g)
                x = tf.keras.layers.multiply([x, g])
                x = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
                x = tk.layers.BlurPooling2D(taps=4)(x)
                x = conv2d(filters)(x)
                x = bn()(x)
                return x

            return layers

        def blocks(filters, count):
            def layers(x):
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
        x = tf.keras.layers.concatenate(
            [
                conv2d(64, kernel_size=2, strides=2)(x),
                conv2d(64, kernel_size=4, strides=2)(x),
            ]
        )  # 1/4
        x = bn()(x)
        x = blocks(128, 2)(x)
        x = down(256)(x)  # 1/8
        x = blocks(256, 4)(x)
        x = down(512)(x)  # 1/16
        x = blocks(512, 4)(x)
        x = down(512)(x)  # 1/32
        x = blocks(512, 4)(x)
        x = tk.layers.GeM2D()(x)
        x = tf.keras.layers.Dense(
            num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="logits",
        )(x)
        x = tf.keras.layers.Activation(activation="softmax")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        return model

    def create_optimizer(self, mode: str) -> tk.models.OptimizerType:
        base_lr = 1e-3 if mode != "refine" else 1e-5
        lr = base_lr * batch_size * tk.hvd.size()
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
        return optimizer

    def create_loss(self, model: tf.keras.models.Model) -> tuple:
        def loss(y_true, y_pred):
            del y_pred
            logits = model.get_layer("logits").output
            return tk.losses.categorical_crossentropy(
                y_true, logits, from_logits=True, label_smoothing=0.2
            )

        metrics = ["acc"]
        return loss, metrics


def _tta(model, X_batch):
    pred_list = tk.models.predict_on_batch_augmented(
        model,
        X_batch,
        flip=(False, True),
        crop_size=(3, 3),
        padding_size=(32, 32),
        padding_mode="edge",
    )
    return np.mean(pred_list, axis=0)


class MyDataLoader(tk.data.DataLoader):
    """DataLoader"""

    def __init__(self, mode):
        super().__init__(
            batch_size=batch_size,
            data_per_sample=2 if mode == "train" else 1,
            parallel=True,
        )
        self.mode = mode
        if self.mode == "train":
            self.aug1 = A.Compose(
                [
                    tk.image.RandomTransform(
                        width=train_shape[1],
                        height=train_shape[0],
                        base_scale=predict_shape[0] / train_shape[0],
                    ),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
        elif self.mode == "refine":
            self.aug1 = tk.image.RandomTransform.create_refine(
                width=predict_shape[1], height=predict_shape[0]
            )
            self.aug2 = None
        else:
            self.aug1 = tk.image.Resize(width=predict_shape[1], height=predict_shape[0])
            self.aug2 = None

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
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


if __name__ == "__main__":
    app.run(default="train")
