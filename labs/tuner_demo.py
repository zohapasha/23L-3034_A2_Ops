import keras
import keras_tuner as kt
import pandas as pd

keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# small subset so it runs quickly
x_train, y_train = x_train[:10000] / 255.0, y_train[:10000]
x_test, y_test = x_test[:2000] / 255.0, y_test[:2000]


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model.compile(keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=6, factor=3, seed=42,
                     overwrite=True, directory="tuner_runs", project_name="fashion")
tuner.search(x_train, y_train, validation_split=0.2)

trials = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
          for t in tuner.oracle.trials.values()]
df = pd.DataFrame(trials)[["trial", "units", "learning_rate", "val_accuracy"]]
print(df.sort_values("val_accuracy", ascending=False).to_string(index=False))

best = tuner.get_best_hyperparameters(1)[0]
print("\nbest units:", best.get("units"), " best learning rate:", best.get("learning_rate"))
best_model = tuner.get_best_models(1)[0]
print("test accuracy:", round(best_model.evaluate(x_test, y_test, verbose=0)[1], 4))