import logging
from tqdm import tqdm
from time import time
import Weather_Data as WD
import numpy as np
from keras import Sequential
import keras
import os
from keras.layers import (
    Flatten,
    Dense,
    SimpleRNN,
    LSTM,
    BatchNormalization,
    Conv1D,
    Dropout,
)
import tensorflow as tf

tf.data.experimental.enable_debug_mode()

# Убираем предупреждения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel(logging.ERROR)


def what_device_use(device="gpu"):
    if device.lower() == "gpu":
        # Работаем с GPU
        tf.config.set_visible_devices(tf.config.list_physical_devices("GPU"), "GPU")

    if device.lower() == "cpu":
        # Работаем с CPU
        tf.config.set_visible_devices([], "GPU")


def load_data(name_db="moscow", how_many_context_days=10):
    """Загружаем данные"""
    global train_data, train_data_answer

    # В WD.get_moscow_data     477_603 записей      (все данные идут с шагом в 1 часа)
    # В WD.get_fresh_data        1_455 записей      (данные за последние 60
    # дней, идут с шагом в 1 час)

    if name_db == "moscow":
        DATA_out = WD.get_moscow_data()
    elif name_db == "fresh":
        DATA_out = WD.get_fresh_data(how_many_context_days)
    print(">>> Dataset loaded\n")

    """DATA_in == Данные погоды, начиная с 1ого дня (принимает)
       DATA_out == Данные погоды, начиная с 2ого дня (должен предсказать)"""

    # Создаём смещени назад во времени
    DATA_in = DATA_out[:-1]
    DATA_out = DATA_out[1:]

    # Преобразуем данные
    DATA_out = np.array(DATA_out).reshape((len(DATA_out), 1, 8))
    DATA_in = np.array(DATA_in).reshape((len(DATA_out), 1, 8))

    # Остаточное обучение + нормализуем от -1 до 1 (а не от -0.01 до 0.01)
    DATA_out = DATA_out - DATA_in
    DATA_out = WD.normalize(DATA_out)
    DATA_out = DATA_out[:, :, 3:]  # ИИшке не надо предсказывать время

    train_data = DATA_in
    train_data_answer = DATA_out


def ai_name(name):
    """Сохранения / Загрузки"""
    global save_path, SAVE_NAME

    def save_path(ai_name):
        return f"Saves Weather Prophet/{ai_name}"

    def SAVE_NAME(num):
        return f"{name}~{num}"


def load_ai(loading_with_learning_cycle=-1, print_summary=False):
    """ЗАГРУЖАЕМСЯ"""
    global ai

    # Вычисляем номер последнего сохранения с текущем именем
    if loading_with_learning_cycle == -1:
        loading_with_learning_cycle = int(
            sorted(
                [
                    save_name if SAVE_NAME(0)[:-2] in save_name else None
                    for save_name in os.listdir("Saves Weather Prophet")
                ]
            )[-1].split("~")[-1]
        )

    print(f">>> Loading the {SAVE_NAME(loading_with_learning_cycle)}", end="\t\t")
    ai = tf.keras.models.load_model(save_path(SAVE_NAME(loading_with_learning_cycle)))
    print("Done\n")

    if print_summary:
        ai.summary()
        print()


def create_ai(num_layers_conv=3, num_ai_layers=5, num_neurons=32, print_summary=True):
    """Создаём ИИшки"""
    global ai
    # Суть в том, чтобы расперелить задачи по предсказыванию между разными нейронками
    # Т.к. одна нейросеть очень плохо предскаывает одновременно все факторы

    # У всех нейронок одна архитектура и один вход
    input_layer = keras.Input((1, 8))

    class Architecture:
        def get_ai(self):
            num_conv_neurons = 8
            list_layers = []

            # Добавляем Conv1D
            for _ in range(num_layers_conv):
                list_layers.append(Conv1D(num_conv_neurons, 8, padding="same"))
                num_conv_neurons *= 2

            # Добавляем остальные слои
            for i in range(num_ai_layers):
                if i % 2 == 0:
                    list_layers.append(Dense(num_neurons, activation="relu"))
                else:
                    list_layers.append(
                        LSTM(num_neurons, return_sequences=True, unroll=True)
                    )

            return Sequential(list_layers)(input_layer)

    # Создаём 5 полностью независимые нейронки
    temperature = Dense(1, activation="tanh", name="temp")(Architecture().get_ai())
    pressure = Dense(1, activation="tanh", name="press")(Architecture().get_ai())
    humidity = Dense(1, activation="tanh", name="humid")(Architecture().get_ai())
    cloud = Dense(1, activation="tanh", name="cloud")(Architecture().get_ai())
    rain = Dense(1, activation="tanh", name="rain")(Architecture().get_ai())

    ai = keras.Model(
        input_layer,
        [temperature, pressure, humidity, cloud, rain],
        name="Weather_Predictor",
    )

    ai.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mean_squared_error",
               loss_weights={"temp": 100_000, "press": 10_000, "humid": 10_000, "cloud": 10_000, "rain": 100_000},)
               # Отдаём приоритет температуре и осадкам, и увеличиваем ошибки (иначе они будут ≈0)

    if print_summary:
        ai.summary()
        print()


def start_train(  # ЭТА ФУНКЦИЯ НУЖНА ЧТОБЫ ОБУЧТЬ ИИШКУ ПРЕДСКАЗВАТЬ СЛЕДУЮЩИЙ ЧАС
        start_on=-1,
        finish_on=99,  # Начинаем с номера последнего сохранения до finish_on
        epochs=3,
        batch_size=100,
        verbose=2,  # Параметры fit()
        print_ai_answers=True,
        len_prints_ai_answers=100,  # Выводить данные, или нет
        print_weather_predict=True,  # Выводить ли прогноз погоды
        len_predict_days=1,
        use_callbacks=False,
        callbacks_min_delta=10,
        callbacks_patience=3,  # Параметры callbacks
        shift_dataset_every_cycle=True,
        start_with_dataset_offset=0,  # Смещаем данные на 1 час каждый цикл
):
    """Это просто большая обёртка вокруг функции обучения"""
    global train_data, train_data_answer
    num_dataset_offset = 1

    # Сдвигаемм наборы данных
    if start_with_dataset_offset > 0:
        num_dataset_offset += start_with_dataset_offset
        train_data = train_data[:-start_with_dataset_offset]
        train_data_answer = train_data_answer[start_with_dataset_offset:]

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
    if start_on == -1:
        try:
            start_on = int(
                sorted(
                    [
                        save_name if SAVE_NAME(0)[:-2] in save_name else None
                        for save_name in os.listdir("Saves Weather Prophet")
                    ]
                )[-1].split("~")[-1]
            ) +1
        except BaseException:
            start_on = 0

    # Циклы обучения
    for learning_cycle in range(start_on, finish_on):
        print(
            f">>> Learning the {SAVE_NAME(learning_cycle)}\t\t\t"
            f"Предсказывает на: {num_dataset_offset} ч вперёд"
        )
        ai.fit(
            train_data,
            train_data_answer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=False,
            callbacks=callbacks,
        )
        print()

        # Сохраняем
        print(
            f">>> Saving the {SAVE_NAME(learning_cycle)}  (Ignore the WARNING)",
            end="\t\t",
        )
        ai.save(save_path(SAVE_NAME(learning_cycle)))
        print("Done\n")

        # Выводим данные и сравниваем
        if print_ai_answers:
            WD.print_ai_answers(ai, train_data, len_prints_ai_answers)
        if print_weather_predict:
            WD.print_weather_predict(ai, len_predict_days)

        # Создаём смещение данных на 1 час
        if shift_dataset_every_cycle:
            num_dataset_offset += 1
            train_data = train_data[:-1]
            train_data_answer = train_data_answer[1:]


@tf.function
def train_step(Times, Data_batch, len_predict):
    """Делаем свою функцию fit()
    (Нужна она чтобы обучать ИИшку составлять прогноз, на основе своих данных)"""
    global loss_func, optimizer

    times = Times
    data_batch = Data_batch

    with tf.GradientTape() as tape:
        # Составляем прогноз длиной len_predict
        ai_predicts = tf.expand_dims(tf.expand_dims(data_batch[0], axis=0), axis=0)

        for _ in range(len_predict):
            joind_vector = tf.concat([times[-1], ai_predicts[-1][0]], 0)
            joind_vector = tf.expand_dims(
                tf.expand_dims(joind_vector, axis=0), axis=0
            )

            # Записываем предсказание ИИшки на следующий час
            ai_ans = [ai(joind_vector, training=True)[i][0] for i in range(5)]
            ai_ans = tf.expand_dims(tf.cast(
                tf.reshape(ai_ans, [1, -1]),
                tf.float64), axis=0)
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
            # time = [-1 if i > 1 else i for i in time]

            times = tf.concat(
                [times[1:], tf.reshape(time, [1, -1])], 0
            )  # Смещаем прогноз

        # ИИшка должна предсказать будущую погоду
        real_ans = data_batch[1: len_predict + 1]
        ai_pred = ai_predicts[1: len_predict + 1]
        loss = tf.keras.losses.mean_squared_error(real_ans, ai_pred)

        # Состовляем градиенты
        gradients = tape.gradient(loss, ai.trainable_variables)

        # Изменяем веса
        (keras.optimizers.Adam(2e-4)).apply_gradients(
            zip(gradients, ai.trainable_variables)
        )

        return loss


def train_make_predict(
        batch_size=100, amount_batches=10, len_predict=24, start_on=-1, finish_on=99
):
    """Эта функция нужна чтобы обучить ИИшку состовлять прогноз"""
    tf.config.run_functions_eagerly(True)

    if batch_size <= len_predict:
        raise "len_predict sould be < batch_size"

    # Продолжаем с последнего сохранения если start_on == -1 (или создаём новое)
    if start_on == -1:
        try:
            start_on = int(
                sorted(
                    [
                        save_name if SAVE_NAME(0)[:-2] in save_name else None
                        for save_name in os.listdir("Saves Weather Prophet")
                    ]
                )[-1].split("~")[-1]
            ) + 1
        except BaseException:
            start_on = 0

    # Циклы обучения
    for learning_cycle in range(start_on, finish_on):
        print(f">>> Learning the {SAVE_NAME(learning_cycle)}")
        losses = []

        # Разделяем train_data на батчи (В посленем батче — оставшиеся данные)
        batchs_data = [
                          train_data[i: i + batch_size]
                          for i in range(0, len(train_data), batch_size)
                      ][:-1]
        # Берём рандомный промежуток батчей
        rand = np.random.randint(len(batchs_data) - amount_batches)
        batchs_data = batchs_data[:-1][rand: rand + amount_batches]

        for b in tqdm(range(len(batchs_data)), desc=f"Epoch {learning_cycle}/{finish_on}"):
            times = tf.Variable(batchs_data[b][:, 0, :3], tf.float64)
            data_batch = tf.Variable(batchs_data[b][:, 0, 3:], tf.float64)

            losses.append(train_step(times, data_batch, len_predict))

        print(
            f"Loss: {round(np.mean(losses), 5)} (mean); {round(np.min(losses), 5)} min\n"
        )

        # Сохраняем
        print(f">>> Saving the {SAVE_NAME(learning_cycle)}  (Ignore the WARNING)", end="\t\t")
        ai.save(save_path(SAVE_NAME(learning_cycle)))
        print("Done\n")

        WD.print_weather_predict(ai, 1)


if __name__ == "__main__":
    what_device_use("cpu")
    ai_name("AI_v4.2")
    load_data("moscow")

    # create_ai(5, 5, 64, print_summary=True)
    load_ai(-1, print_summary=False)

    start_train(-1, 7, epochs=10, batch_size=100, verbose=1)
    train_make_predict(50, 20, len_predict=24)
