import datetime as dt
import gzip
import os
from urllib.request import urlretrieve

import numpy as np

np.set_printoptions(suppress=True)  # Убраем экспонинцеальную запись


class NormalizeData():
    """ Данные:
    0) Время: "11.07.2025  9:00:00"
    1) Температура: ℃, от -33 до +38.5
    2) Давление на уровне станции: мм.рт.ст., от 709.1 до 781.4
    3) Влажность: %
    4) Скорость ветра: м/с, от 0 до 15
    5) Облачность: %, с шагом в 10%
    6) Состояние погоды: буковы
    7) Количество осадков: мм, от 0 до 65
    8) Время за которое были осадки: ч, 1/2/6/18/24
    """

    @staticmethod
    def temperature(x: np.ndarray, convert_back=False):
        """Нормализуем температуру от -1 до 1"""
        if convert_back:
            return ((x + 1) / 2) * (38.5 - -33) + -33

        return ((x - -33) / (38.5 - -33)) * 2 - 1

    @staticmethod
    def pressure(x: np.ndarray, convert_back=False):
        """Нормализуем давление от -1 до 1"""
        if convert_back:
            return ((x + 1) / 2) * (781.4 - 709.1) - 709.1

        return ((x - 709.1) / (781.4 - 709.1)) * 2 - 1

    @staticmethod
    def humidity(x: np.ndarray, convert_back=False):
        """Нормализуем влажность от -1 до 1"""
        if convert_back:
            return int(x * 50 + 50)

        return (x - 50) / 50

    @staticmethod
    def cloud(x: np.ndarray, convert_back=False):
        """Нормализуем облачность от -1 до 1"""
        if convert_back:
            return int(x * 50 + 50)

        return (x - 50) / 50

    @staticmethod
    def rain(x: np.ndarray, convert_back=False):
        """Нормализуем осадки от 0 до ∞"""
        if convert_back:
            return np.exp(x) - 1

        return np.log(x + 1)

    @staticmethod
    def time(x: np.ndarray):
        """Нормализуем время от -1 до 1, в зависимости от того что это
        (день/месяц/год), период разный:
        час: 24, день: 31, месяц: 12, год не подаём"""

        radians = 2 * np.pi * x / period
        return np.stack([np.sin(radians), np.cos(radians)], axis=-1)

    @staticmethod
    def clamp(num: float, minimum: float, maximum: float):
        """Ограничиваем сверху и снизу"""
        return min(max(minimum, num), maximum)

    @staticmethod
    def conv_ai_ans_for_human(ai_answer: list):
        """
        ai_answer = [
            температура,
            давление,
            влажность,
            облачность,
            осадки должны быть,
            количество осадков
        ]
        """
        return [
            NormalizateData.temperature(ai_answer[0], True),
            NormalizateData.pressure(ai_answer[1], True),
            NormalizateData.humidity(ai_answer[2], True),
            NormalizateData.cloud(ai_answer[3], True),
            NormalizateData.rain(ai_answer[4], True),
            ai_answer[5] == True,
        ]


def get_moscow_data():
    """
    -> out
    0) Время (часы)
    1) Время (день)
    2) Время (месяц)
    3) Температура
    4) Давление (мм.рт.ст.)
    5) Влажность
    6) Облачность
    7) Количество осадков
    """

    DATA = []
    with open(f"Moscow_Weather.txt") as dataset:
        # Без первой строки с начальным символом;
        # Без первой (идём от староого к новому) записи, т.к. она используется для смещения ответа
        # (т.е. чтобы мы на основе предыдущей записи создавали следующую)

        for string in dataset.readlines()[1:][::-1]:
            data = string.split(";")[:-1]

            # Если попался брак, то пропускаем шаг
            if "" in data or len(data) != 6:
                continue

            processed_data = [0 for _ in range(8)]

            # Преобразуем строку
            # data[0] -> часы (в течении дня)
            # datap[1] -> день
            # data[2] -> месяц
            processed_data[0] = norm_hours(int(data[0][11:13]))
            processed_data[1] = norm_day(int(data[0][:2]))
            processed_data[2] = norm_month(int(data[0][3:5]))
            processed_data[3] = norm_temperature(
                clamp(float(data[1].replace(",", ".")), -40, 40)
            )
            processed_data[4] = norm_pressure(
                clamp(float(data[2].replace(",", ".")), 700, 800)
            )
            processed_data[5] = norm_humidity(int(data[3]))

            # Облачность...
            clouds = data[4].replace(".", "").replace("%", "").split()

            if "–" in clouds[0]:
                clouds[0] = clouds[0].split("–")

                processed_data[6] = norm_cloud(int(clouds[0][1]))
            else:
                # Заменяем слова числами
                for ind, h in enumerate(clouds):
                    if h.isnumeric():
                        clouds[ind] = int(h)
                    else:
                        clouds[ind] = 0
                # В качестве облачности выбираем максимальное значение
                processed_data[6] = norm_cloud(max(clouds))

            # Добавляем осадки
            # (-1 -> сильный снег, -0.5 -> снег, 0 -> Осадков нет, 0.5 -> дождь, 1 -> сильный дождь)
            # (кстати, "ливень"/"ливневый" не означает "сильный")
            processed_data[7] = 0
            data[5] = data[5].lower()
            if "снег" in data[5] or "дождь и снег" in data[5]:
                processed_data[7] = -0.5
            if "дожд" in data[5]:
                processed_data[7] = 0.5

            if "слаб" in data[5]:
                processed_data[7] /= 2
            if "силь" in data[5]:
                processed_data[7] *= 2

            # На всякий случай ограничиваем все значения от -1 до 1 (обрезаем лишнее)
            for i in range(len(processed_data)):
                processed_data[i] = clamp(processed_data[i], -1, 1)

            DATA.append(processed_data)

    return DATA


def get_fresh_data(how_many_context_days: int):
    now_date = dt.datetime.today()
    last_date = now_date - dt.timedelta(days=how_many_context_days // 1 + 1)

    now_date = now_date.strftime("%d.%m.%Y")
    last_date = last_date.strftime("%d.%m.%Y")

    # Скачиваем архивчик
    for i in range(1, 10):
        try:
            urlretrieve(
                f"https://ru{i}.rp5.ru/download/files.synop/27/27612."
                f"{last_date}.{now_date}.1.0.0.ru.utf8.00000000.csv.gz",
                f"Datasets/FRESH_ARCHIVE.csv.gz",
            )
        except BaseException:
            continue
        else:
            break
    else:  # Если не удалось скачать
        raise RuntimeError(
            f"Пожалуйста, перейдите по ссылке: https://rp5.ru/%D0%90%D1%80%D1%85%D0%B8%D0%B2_"
            f"%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D1%8B_%D0%B2_%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B5_%28%D0%92%D0%94%D0%9D%D0%A5%29\n"
            f"Выберете дату с {last_date} по {now_date}, в формате csv и кодировкой utf8, "
            f"потом нажмите 'Выбрать в файл GZ (архив)' (архив можно не скачивать)\n"
            f"После этого перезапустите программу"
        )

    # Загружаем данные
    with open("Datasets/FRESH_ARCHIVE.csv.gz", "rb") as byte_file:
        data = str(gzip.decompress(byte_file.read()), "utf-8")

        data = [string.split(";") for string in data.split("\n")[7:]]

        # Убираем всякие "", ' ', '\r', '', '""'
        processed_data = []
        for record in data:
            to_append = []
            for i in record:
                val = i.replace('"', "").replace("\r", "")
                to_append.append(val)
            processed_data.append(to_append)
        data = processed_data[:-1]

        # Оставляем только необходимые данные
        required_data = [[d[0], d[1], d[2], d[5], d[10], d[11]] for d in data]

    # Удаляем архив
    os.remove("Datasets/FRESH_ARCHIVE.csv.gz")

    # Переводим строки в числа для ИИшки
    DATA = []
    for data in required_data:
        # Если попался брак, то пропускаем шаг
        if "" in data or len(data) != 6:
            continue

        processed_data = [0 for _ in range(8)]

        # Преобразуем строку
        # data[0] -> часы (в течении дня)
        # datap[1] -> день
        # data[2] -> месяц
        processed_data[0] = norm_hours(int(data[0][11:13]))
        processed_data[1] = norm_day(int(data[0][:2]))
        processed_data[2] = norm_month(int(data[0][3:5]))
        processed_data[3] = norm_temperature(
            clamp(float(data[1].replace(",", ".")), -40, 40)
        )
        processed_data[4] = norm_pressure(
            clamp(float(data[2].replace(",", ".")), 700, 800)
        )
        processed_data[5] = norm_humidity(int(data[3]))

        # Облачность...
        clouds = data[4].replace(".", "").replace("%", "").split()

        if "–" in clouds[0]:
            clouds[0] = clouds[0].split("–")

            processed_data[6] = norm_cloud(int(clouds[0][1]))
        else:
            # Заменяем слова числами
            for ind, h in enumerate(clouds):
                if h.isnumeric():
                    clouds[ind] = int(h)
                else:
                    clouds[ind] = 0
            # В качестве облачности выбираем максимальное значение
            processed_data[6] = norm_cloud(max(clouds))

        # Добавляем осадки
        # (-1 -> сильный снег, -0.5 -> снег, 0 -> Осадков нет, 0.5 -> дождь, 1 -> сильный дождь)
        # (кстати, "ливень"/"ливневый" не означает "сильный")
        processed_data[7] = 0
        data[5] = data[5].lower()
        if "снег" in data[5] or "дождь и снег" in data[5]:
            processed_data[7] = -0.5
        if "дожд" in data[5]:
            processed_data[7] = 0.5

        if "слаб" in data[5]:
            processed_data[7] /= 2
        if "силь" in data[5]:
            processed_data[7] *= 2

        # На всякий случай ограничиваем все значения от -1 до 1 (обрезаем лишнее)
        for i in range(len(processed_data)):
            processed_data[i] = clamp(processed_data[i], -1, 1)

        DATA.append(processed_data)

    return DATA[: int(how_many_context_days * 24 // 3)]


def print_ai_answers(ai, real_data, batch_size=100, num_answers=50):
    print("\n")
    print("Time\t\t\tReal Data\t\t\t\t\t\t\tAI answer\t\t\t\t\t\t\tErrors ∆")

    ai.reset_states()  # Очищаем данные, оставшиеся от обучения

    total_errors = []

    # Случайный батч            (-num_answers -batch_size чтобы не вышли за границу)
    rand = np.random.randint(len(real_data) - num_answers - batch_size)
    real_batch = real_data[rand : rand + num_answers + batch_size]

    for b in range(num_answers):
        real_data_list = real_batch[b : batch_size + b]

        ai.reset_states()  # Очищаем данные, оставшиеся от прошлого батча

        # Предсказание ИИшки на основе батча данных
        pred = ai.predict(real_data_list, verbose=False, batch_size=batch_size)
        ai_ans = np.reshape(np.array(pred)[:, -1], (5))

        # Не забываем про остаточное обучение и нормализацию
        ai_ans = np.array([[ai_ans + real_data_list[-1, 0, 3:]]])
        ai_ans_list = normalize(ai_ans, convert_back=True)[0].tolist()
        # Для версии без остаточного обучения
        # ai_ans_list = normalize(np.array([[ai_ans]]), convert_back=True)[0].tolist()

        # Конвертируем данные из промежутка [-1; 1] в нормальную физическую  величину
        real_data_vect = real_data_list[-1, 0].tolist()
        real_data_for_human = [
            norm_hours(real_data_vect[0], True),
            norm_day(real_data_vect[1], True),
            norm_month(real_data_vect[2], True),
        ] + conv_ai_ans_for_human(real_data_vect[3:])

        ai_ans_for_human = conv_ai_ans_for_human(ai_ans_list)

        # В качестве ошибки просто добавляем разность между ответом ИИ и реальностью
        errors = np.array(
            np.abs(np.array(real_data_for_human[3:]) - np.array(ai_ans_for_human))
        )
        total_errors.append(errors)

        # Выводим всё
        print(
            np.round(np.array(real_data_for_human[:3]), 1),
            "\t",
            np.round(np.array(real_data_for_human[3:]), 1),
            "\t",
            np.round(np.array(ai_ans_for_human), 1),
            "\t",
            np.round(np.array(errors), 1),
        )

    total_errors = np.array(total_errors)

    print(
        "\nMean errors:",
        "\n\t Temperature:  ",
        np.round(np.mean(total_errors[:, 0]), 1),
        "\n\t Pressure:     ",
        np.round(np.mean(total_errors[:, 1]), 1),
        "\n\t Humidity:     ",
        np.round(np.mean(total_errors[:, 2]), 1),
        "\n\t Cloud:        ",
        np.round(np.mean(total_errors[:, 3]), 1),
        "\n\t Rain:         ",
        np.round(np.mean(total_errors[:, 4]), 1),
        "\n\n\t TOTAL:        ",
        np.round(np.mean(total_errors), 1),
    )

    print("\n")


def print_weather_predict(ai, len_predict_days=3, batch_size=100):
    print(
        f"Prediction for the next {len_predict_days} days:\t\t\t",
        f"(Temperature, Pressure, Humidity, Cloud, Raininess)",
    )
    ai.reset_states()  # Очищаем данные, оставшиеся от обучения

    fresh_data = np.array(get_fresh_data(batch_size / 8))[-batch_size:]
    # Самое последнее - самое свежее
    fresh_data = np.reshape(fresh_data, (fresh_data.shape[0], 8))[::-1]
    times = fresh_data[:, :3].tolist()
    predicts_history = fresh_data[:, 3:].tolist()

    # Делаем прогноз по всей истории, а потом отбираем один прогноз, относящееся к последней записи
    for _ in range(int(len_predict_days * 24 // 3)):
        preds_on_preds = ai.predict(
            [[t + p] for t, p in zip(times, predicts_history)],
            verbose=False,
            batch_size=batch_size,
        )
        ai_ans = np.reshape(np.array(preds_on_preds)[:, -1], (1, 1, 5))
        ai_ans = normalize(ai_ans, convert_back=True)

        # Не забываем про остаточное обучение
        ai_ans = ai_ans + np.array(predicts_history[-1])

        # Добавляем
        predicts_history.append(ai_ans.tolist()[0])

        # Сдвигаем последовательность
        predicts_history = predicts_history[1:]

        # Обновляем время
        time = times[-1]
        time[0] += 3 / 12  # Увеличиваем часы
        time[1] += 1 / 15.5 if time[0] >= 1 else 0  # Увеличиваем день
        time[2] += 1 / 6 if time[1] >= 1 else 0  # Увеличиваем месяц

        # Следим, чтобы зачения не выходили за границы
        time = [-1 if i >= 1 else i for i in time]

        times.append(time)

        # Выводим прогноз
        time_for_human = [
            norm_hours(time[0], True),
            norm_day(time[1], True),
            norm_month(time[2], True),
        ]

        pred_weather_for_human = conv_ai_ans_for_human(predicts_history[-1])
        print(
            f"{'{:02}'.format(time_for_human[1])}.{'{:02}'.format(time_for_human[2])}",
            f"{'{:02}'.format(time_for_human[0])}:00:\t",
            pred_weather_for_human[0],
            "℃\t",
            pred_weather_for_human[1],
            "mmHg\t",
            pred_weather_for_human[2],
            "%\t",
            pred_weather_for_human[3],
            "%\t",
            conv_rain_to_words(pred_weather_for_human[4]),
        )
    print("\n")
