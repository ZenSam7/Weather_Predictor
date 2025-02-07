print(">>> Importing libraries", end="\t\t")

import numpy as np
from keras.layers import (
    Flatten,
    Dense,
    SimpleRNN,
    LSTM,
    BatchNormalization,
    Conv1D,
    Dropout,
    Input,
)
import keras
from tqdm import tqdm
import tensorflow as tf
from logging import ERROR
import Weather_Data as wd

# tf.data.experimental.enable_debug_mode()

# Убираем предупреждения
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel(ERROR)

print("Done")


def what_device_use(device="cpu"):
    # Работаем с CPU
    tf.config.set_visible_devices([], "GPU")

    if device.lower() == "gpu":
        # Работаем с GPU
        tf.config.set_visible_devices(tf.config.list_physical_devices("GPU"), "GPU")


def load_data(name_db="moscow", how_many_context_days=20):
    """Загружаем данные"""
    global train_data, train_data_answer
    print("\n>>> Loading and processing Dataset", end="\t\t")

    # В wd.get_moscow_data     167_217 записей      (все данные идут с шагом в 3 часа)
    # В wd.get_fresh_data        1_455 записей      (данные за последние 60 дней, идут с шагом в 3 часа)

    DATA = wd.get_moscow_data()
    if name_db == "fresh":
        DATA = wd.get_fresh_data(how_many_context_days)

    """DATA_in == Данные погоды, начиная с 1й записи (принимает)
       DATA_out == Данные погоды, начиная с 2й записи (должен предсказать)"""

    # Создаём смещени назад во времени
    DATA_in = DATA[:-1]
    DATA_out = DATA[1:]

    # Преобразуем данные
    DATA_in = np.array(DATA_in).reshape((len(DATA_out), 1, 8))
    DATA_out = np.array(DATA_out).reshape((len(DATA_out), 1, 8))

    # Остаточное обучение
    DATA_out = DATA_in - DATA_out
    # ИИшке не надо предсказывать время
    DATA_out = DATA_out[:, :, 3:]
    # Нормализуем (чтобы ИИшка могла как можно шире )
    DATA_out = wd.normalize(DATA_out)

    train_data = DATA_in
    train_data_answer = DATA_out

    print("Done\n")


def ai_name(name):
    """Всякие функции"""

    global get_save_path, get_save_name, get_start_with, save_ai, load_ai

    def get_save_path(ai_name):
        return f"./Saves_Weather_Prophet/{ai_name}"

    def get_save_name(num):
        return f"{name}~{num}"

    def save_ai(num):
        print(f"\n>>> Saving the {get_save_name(num)}  (Ignore the WARNING)", end="\t\t")
        ai.save(get_save_path(get_save_name(num)))
        print("Done\n")

    def get_start_with(start_with=-1):
        # Вычисляем номер последнего сохранения с текущем именем
        if start_with == -1:
            try:
                saves = []
                for save_name in os.listdir("Saves_Weather_Prophet"):
                    if get_save_name(0)[:-2] in save_name:
                        saves.append(save_name)

                assert saves != [], f"Нет ни одного сохранения с именем {get_save_name(0)[:-2]}"

                start_with = int(sorted(saves)[-1].split("~")[-1])
            except BaseException:
                return 0

        return start_with

    def load_ai(load_with=-1, print_summary=False):
        """ЗАГРУЖАЕМСЯ"""
        global ai

        # Вычисляем номер последнего сохранения с текущем именем
        loading_with = get_start_with(load_with)

        print(f">>> Loading the {get_save_name(loading_with)}", end="\t\t")
        ai = tf.keras.models.load_model(get_save_path(get_save_name(loading_with)))
        print("Done\n")
        if print_summary:
            ai.summary()
            print()


def show_architecture_ai(ai):
    from keras.utils.vis_utils import plot_model

    name = str(get_save_name(0))[:-2]
    plot_model(ai, to_file=f"{name}.png", show_shapes=True, show_layer_names=True)


def create_ai(
    num_layers_conv=3,
    num_main_layers=5,
    num_neurons=32,
    batch_size=100,
    print_summary=True,
):
    """Создаём ИИшки"""
    global ai

    # Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
    # Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы
    general_input = Input(batch_input_shape=(batch_size, 1, 8))

    class Create_AI:
        def get_model(self):
            num_conv_neurons = 4
            model = keras.Sequential()
            model.add(general_input)

            # Добавляем Conv1D
            for _ in range(num_layers_conv):
                num_conv_neurons *= 2
                list_layers.append(Conv1D(num_conv_neurons, 8, padding="same"))

            # Добавляем основные слои (чередуем Dense и LSTM)
            for i in range(num_main_layers):
                list_layers.append(
                    LSTM(
                        num_neurons,
                        activation="tanh",
                        return_sequences=True,
                        unroll=False,
                        stateful=True,
                    )
                )

            return keras.Sequential(list_layers)(general_input)

    # Создаём 5 полностью независимые нейронки
    temperature = Dense(1, activation="tanh", name="temp")(Architecture().get_ai())
    pressure = Dense(1, activation="tanh", name="press")(Architecture().get_ai())
    humidity = Dense(1, activation="tanh", name="humid")(Architecture().get_ai())
    cloud = Dense(1, activation="tanh", name="cloud")(Architecture().get_ai())
    rain = Dense(1, activation="tanh", name="rain")(Architecture().get_ai())

    ai = keras.Model(
        general_input,
        [temperature, pressure, humidity, cloud, rain],
        # temperature,
        name="Weather_Predictor",
    )

    # mean_absolute_percentage_error
    ai.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mean_absolute_error",
        loss_weights={
            "temp": 1_000,
            "press": 100,
            "humid": 100,
            "cloud": 100,
            "rain": 100,
        },
    )
    # Отдаём приоритет температуре, и увеличиваем всем ошибки (иначе они будут ≈0)

    if print_summary:
        ai.summary()
        print()


@tf.function
def train_step(all_times, all_data_batch, len_predict, increased_error_factor):
    """Делаем свою функцию fit()
    (Нужна она чтобы обучать ИИшку составлять прогноз, на основе своих данных)"""
    global loss_func, optimizer

    times = all_times
    data_batch = all_data_batch

    with tf.GradientTape() as tape:
        # Составляем прогноз длиной len_predict
        ai_predicts = tf.expand_dims(tf.expand_dims(data_batch[0], axis=0), axis=0)

        for _ in range(len_predict):
            joind_vector = tf.concat([times[-1], ai_predicts[-1][0]], 0)
            joind_vector = tf.expand_dims(tf.expand_dims(joind_vector, axis=0), axis=0)

            # Записываем предсказание ИИшки на следующий час
            ai_ans = [ai(joind_vector, training=True)[i][0] for i in range(5)]
            ai_ans = tf.expand_dims(
                tf.cast(tf.reshape(ai_ans, [1, -1]), tf.float64), axis=0
            )
            ai_predicts = tf.concat([ai_predicts, ai_ans], axis=0)

            # Обновляем время
            time = times[-1]

            to_add = tf.constant([1 / 12, 1 / 15.5, 1 / 6], dtype=tf.float64)
            where_to_add = tf.cast([True, time[0] > 1, time[1] > 1], tf.float64)
            time += to_add * where_to_add

            # Следим, чтобы зачения не выходили за границы
            overflow = tf.cast(time > 1, tf.float64)
            not_overflow = tf.cast(time <= 1, tf.float64)
            time = time * not_overflow + overflow * tf.constant([-1], dtype=tf.float64)

            times = tf.concat(
                [times[1:], tf.reshape(time, [1, -1])], 0
            )  # Смещаем прогноз

        # ИИшка должна предсказать только будущую погоду
        real_ans = data_batch[1: len_predict + 1] * increased_error_factor
        ai_pred = ai_predicts[1: len_predict + 1] * increased_error_factor
        loss = tf.keras.losses.mean_squared_error(real_ans, ai_pred)

        # Состовляем градиенты
        gradients = tape.gradient(loss, ai.trainable_variables)

        # Изменяем веса
        (keras.optimizers.Adam(5e-4)).apply_gradients(
            zip(gradients, ai.trainable_variables)
        )

        return loss


def train_make_predict(
    batch_size=100,
    amount_batches=10,
    len_predict=24,
    start=-1,
    finish_on=99,
    increased_error_factor=100,
):
    """Эта функция нужна чтобы обучить ИИшку состовлять прогноз"""

    assert len_predict < batch_size, "len_predict sould be < batch_size"

    tf.config.run_functions_eagerly(True)
    ai.reset_states()  # Очищаем данные, оставшиеся после обучения

    # Продолжаем с последнего сохранения если start_on == -1 (или создаём новое)
    start_with = get_start_with(start) +1

    # Циклы обучения
    for learning_cycle in range(start_with, finish_on):
        print(f">>> Learning the {get_save_name(learning_cycle)}")
        losses = []

        # Разделяем train_data на батчи (В посленем батче — ненужные данные)
        batchs_data = [
            train_data[i: i + batch_size]
            for i in range(0, len(train_data), batch_size)
        ][:-1]
        # Берём рандомный промежуток батчей
        rand = np.random.randint(len(batchs_data) - amount_batches)
        batchs_data = batchs_data[:-1][rand: rand + amount_batches]

        for b in tqdm(
            range(len(batchs_data)), desc=f"Epoch {learning_cycle}/{finish_on}"
        ):
            times = tf.Variable(batchs_data[b][:, 0, :3], tf.float64)
            data_batch = tf.Variable(batchs_data[b][:, 0, 3:], tf.float64)

            losses.append(
                train_step(times, data_batch, len_predict, increased_error_factor)
            )

        print(
            f"Loss: {round(np.mean(losses), 5)} (mean); {round(np.min(losses), 5)} min\n"
        )

        # Сохраняем
        sane_ai(learning_cycle)

        wd.print_weather_predict(ai, 1)


def start_train(
    start_on=-1,
    finish_on=99,  # Начинаем с номера последнего сохранения до finish_on
    epochs=3,
    batch_size=100,
    verbose=1,
    print_ai_answers=True,
    len_prints_ai_answers=100,
    print_weather_predict=True,
    len_predict_days=3,
    use_callbacks=False,
    callbacks_min_delta=10,
    callbacks_patience=3,
    save_model=True,
):
    """Это просто большая обёртка вокруг функции обучения"""
    global train_data, train_data_answer, ai

    callbacks = (
        [
            keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=callbacks_min_delta,
                patience=callbacks_patience,
                verbose=False,
            )
        ]
        if use_callbacks
        else None
    )

    # Продолжаем с последнего сохранения если start_on == -1 (или создаём новое)
    start_with = get_start_with(start_on) +1

    # Убираем немного записей, чтобы train_data можно было ровно разделить на batch_size
    train_data = train_data[: len(train_data) // batch_size * batch_size]
    train_data_answer = train_data_answer[: len(train_data) // batch_size * batch_size]

    # Циклы обучения
    for learning_cycle in range(start_with, finish_on):
        print(f">>> Learning the {get_save_name(learning_cycle)}")

        ai.reset_states()  # Очищаем данные, оставшиеся от обучения

        ai.fit(
            train_data,
            train_data_answer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=False,
            callbacks=callbacks,
        )

        # Сохраняем
        if save_model:
            save_ai(learning_cycle)

        # Выводим данные и сравниваем
        if print_ai_answers:
            wd.print_ai_answers(ai, train_data, batch_size, len_prints_ai_answers)
        if print_weather_predict:
            wd.print_weather_predict(ai, len_predict_days, batch_size)


if __name__ == "__main__":
    what_device_use("cpu")
    ai_name("AI_v1.0")
    load_data("moscow")

    batch_size = 128

    # create_ai(0, 7, 128, batch_size)
    load_ai(print_summary=True)

    start_train(-1, 2, epochs=4,
        batch_size=batch_size,
        print_weather_predict=False,
        print_ai_answers=True,
    )
