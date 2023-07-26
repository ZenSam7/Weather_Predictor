from numpy import tanh, arctanh
import numpy as np

import time
from time import time as Time
import datetime as dt

from urllib.request import urlretrieve
import requests
import json
import gzip

import os

np.set_printoptions(suppress=True) # Убраем экспонинцеальную запись

def norm_temperature(x, convert_back=False):
    """Нормализуем температуру от -1 до 1"""
    if convert_back:
        return round(arctanh(x)*20, 1)

    return tanh(x/20)

def norm_pressure(x, convert_back=False):
    """Нормализуем давление от -1 до 1"""
    if convert_back:
        return round(arctanh(x) *20 +750, 1)

    return tanh( (x -750)/20 )

def norm_humidity(x, convert_back=False):
    """Нормализуем влажность от -1 до 1"""
    if convert_back:
        return int(x *50 +50)

    return (x -50)/50

def norm_cloud(x, convert_back=False):
    """Нормализуем облачность от -1 до 1"""
    if convert_back:
        return int(x *50 +50)

    return (x -50)/50

def norm_hours(x, convert_back=False):
    """Нормализуем время суток от -1 до 1"""
    if convert_back:
        return int(x*12 +12)

    return (x -12)/12

def norm_day(x, convert_back=False):
    """Нормализуем номер дня от -1 до 1"""
    if convert_back:
        return int(x *15.5 +15.5)

    return (x -15.5) /15.5

def norm_month(x, convert_back=False):
    """Нормализуем номер месяца от -1 до 1"""
    if convert_back:
        return int(x *6 +6)

    return (x -6)/6



def clamp(num, Min, Max):
    return min(max(Min, num), Max)

def normalize(x, convert_back=False):
    """Нормализуем данные от -1 до 1"""

    if convert_back:
        with open("Datasets/Info_About_Last_Dataset.txt", "r") as save:
            save = save.read().split("\n")
            MIN_DATA = float(save[1][4:])
            MAX_DATA = float(save[2][4:])

        result = (x +1) / 2 # от 0 до 1
        result = result * (MAX_DATA - MIN_DATA) + MIN_DATA
        return result


    # Сохраняем информацию о том, как потом нормализовать данные обратно
    os.remove("Datasets/Info_About_Last_Dataset.txt")
    with open("Datasets/Info_About_Last_Dataset.txt", "w+") as save:
        save.write(f"Data set data for last saved AI\n"
                   f"MIN={np.min(x)}\n"
                   f"MAX={np.max(x)}")

    # Сначала нормализуем от 0 до 1
    result = x - np.min(x)
    if np.max(x) != 0.0:
        result = result / np.max(result)

    # Потом от -1 до 1
    result = result *2 -1

    return result


def conv_ai_ans_for_human(List):
    return [
        norm_temperature(List[0], True),
        norm_pressure(List[1], True),
        norm_humidity(List[2], True),
        norm_cloud(List[3], True),
        List[4],
    ]

def conv_rain_to_words(x):
    word = ""

    if abs(x) <= 0.1 :
        return "Clear"

    if x > 0:
        word = "rain"
    elif x < 0:
        word = "snow"

    if abs(x) <= 0.4:
        word = "light " + word
    elif 0.4 <= abs(x) <= 0.7:
        word = word #"moderate " + word
    elif abs(x) >= 0.7:
        word = "heavy " + word

    return word.capitalize()



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
    with open(f"Datasets/Moscow_Weather.txt") as dataset:
        # Без первой строки с начальным символом;
        # Без первой (идём от староого к новому) записи, т.к. она используется для смещения ответа
        # (т.е. чтобы мы на основе предыдущей записи создавали следующую)

        for string in dataset.readlines()[1:][::-1]:
            data = string.split(";")[:-1]

            # Если попался брак, то пропускаем шаг
            if '' in data or len(data) != 6: continue


            processed_data = [0 for _ in range(8)]

            # Преобразуем строку
            # data[0] -> часы (в течении дня)
            # datap[1] -> день
            # data[2] -> месяц
            processed_data[0] = norm_hours(int(data[0][11:13]))
            processed_data[1] = norm_day(int(data[0][:2]))
            processed_data[2] = norm_month(int(data[0][3:5]))
            processed_data[3] = norm_temperature(clamp(float(data[1].replace(",", ".")), -40, 40))
            processed_data[4] = norm_pressure(clamp(float(data[2].replace(",", ".")), 700, 800))
            processed_data[5] = norm_humidity(int(data[3]))

            # Облачность...
            clouds = data[4].replace('.', '').replace('%', '').split()

            if '–' in clouds[0]:
                clouds[0] = clouds[0].split('–')

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

            DATA.append(processed_data)

    # Заполняем промежуточными значеними (т.к. у нас данные идут с шагом в 3 часа)
    DATA = np.array(DATA)
    conv_DATA = []
    for i in range(len(DATA) - 1):
        for conved in np.linspace(DATA[i], DATA[i + 1], num=4).tolist()[1:]:
            conv_DATA.append(conved)

    # Часы и числа дня заполняем отдельно
    time_h, time_d, time_m = DATA[0, 0], DATA[0, 1], DATA[0, 2]    # Начинаем с начала
    ind = 0     # Надо, чтобы одновременно заполнять DATA
    while ind != len(DATA) -1:
        time_h += 1/12
        if time_h >1:
            time_h = -1

        time_d += 1/15.5 if time_h == -1 else 0
        if time_d > 1:
            time_d = -1

        time_m += 1/6 if time_d == -1 else 0
        if time_m > 1:
            time_m = -1

        DATA[ind, 0], DATA[ind, 1], DATA[ind, 2] = time_h, time_d, time_m
        ind += 1

    return conv_DATA


def get_fresh_data(how_many_context):
    now_date = dt.datetime.today()
    last_date = now_date - dt.timedelta(days=how_many_context // 24 + 1)

    now_date = now_date.strftime("%d.%m.%Y")
    last_date = last_date.strftime("%d.%m.%Y")

    # Скачиваем архивчик
    for i in range(1, 10):
        try:
            urlretrieve(f"https://ru{i}.rp5.ru/download/files.synop/27/27612."\
                        f"{last_date}.{now_date}.1.0.0.ru.utf8.00000000.csv.gz",
                        f"Datasets/FRESH_ARCHIVE.csv.gz")
        except:
            continue
        else:
            break
    else:     # Если не удалось скачать
        raise RuntimeError(
              f"Пожалуйста, перейдите по ссылке: https://rp5.ru/%D0%90%D1%80%D1%85%D0%B8%D0%B2_"
              f"%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D1%8B_%D0%B2_%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B5_%28%D0%92%D0%94%D0%9D%D0%A5%29\n"
              f"Выберете дату с {last_date} по {now_date}, в формате csv и кодировкой utf8, "
              f"потом нажмите 'Выбрать в файл GZ (архив)' (архив можно не скачивать)\n"
              f"После этого перезапустите программу")


    # Загружаем данные
    with open("Datasets/FRESH_ARCHIVE.csv.gz", "rb") as byte_file:
        data = str(gzip.decompress(byte_file.read()), "utf-8")

        data = [string.split(";") for string in data.split("\n")[7:]]

        # Убираем всякие "", ' ', '\r', '', '""'
        processed_data = []
        for record in data:
            to_append = []
            for i in record:
                val = i.replace('"', '').replace('\r', '')
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
        if '' in data or len(data) != 6: continue

        processed_data = [0 for _ in range(8)]

        # Преобразуем строку
        # data[0] -> часы (в течении дня)
        # datap[1] -> день
        # data[2] -> месяц
        processed_data[0] = norm_hours(int(data[0][11:13]))
        processed_data[1] = norm_day(int(data[0][:2]))
        processed_data[2] = norm_month(int(data[0][3:5]))
        processed_data[3] = norm_temperature(clamp(float(data[1].replace(",", ".")), -40, 40))
        processed_data[4] = norm_pressure(clamp(float(data[2].replace(",", ".")), 700, 800))
        processed_data[5] = norm_humidity(int(data[3]))

        # Облачность...
        clouds = data[4].replace('.', '').replace('%', '').split()

        if '–' in clouds[0]:
            clouds[0] = clouds[0].split('–')

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

        DATA.append(processed_data)

    # Заполняем промежуточными значеними (т.к. у нас данные идут с шагом в 3 часа)
    DATA = np.array(DATA)
    conv_DATA = []
    for i in range(len(DATA) - 1):
        for conved in np.linspace(DATA[i], DATA[i + 1], num=4).tolist()[1:]:
            conv_DATA.append(conved)

    # Часы и числа дня заполняем отдельно
    time_h, time_d, time_m = DATA[0, 0], DATA[0, 1], DATA[0, 2]    # Начинаем с начала
    ind = 0     # Надо, чтобы одновременно заполнять DATA
    while ind != len(DATA) -1:
        time_h += 1/12
        if time_h >1:
            time_h = -1

        time_d += 1/15.5 if time_h == -1 else 0
        if time_d > 1:
            time_d = -1

        time_m += 1/6 if time_d == -1 else 0
        if time_m > 1:
            time_m = -1

        DATA[ind, 0], DATA[ind, 1], DATA[ind, 2] = time_h, time_d, time_m
        ind += 1


    return conv_DATA[:how_many_context]




def print_ai_answers(ai, real_data, batch_size):
    print("\n")
    print("Time\t\t\tReal Data\t\t\t\t\t\t\tAI answer\t\t\t\t\t\t\tErrors ∆")

    total_errors = []

    # Случайный батч
    rand = np.random.randint(len(real_data) -batch_size)
    real_matrix = np.reshape(np.array([real_data[rand: rand +batch_size]]), (batch_size, 8))

    for b in range(1, batch_size):
        real_data_list = np.resize(real_matrix[b], (8)).tolist()

        ai_ans_list = np.reshape(np.array(ai.predict([[real_data_list]], verbose=False)), (5))
        # Не забываем про остаточное обучение
        ai_ans_list = normalize(ai_ans_list, True)
        ai_ans_list = (ai_ans_list + real_matrix[b][3:]).tolist()

        # Конвертируем данные из промежутка [-1; 1] в нормальную физическую величину
        real_data_list = [
                norm_hours(real_data_list[0], True),
                norm_day(real_data_list[1], True),
                norm_month(real_data_list[2], True),
        ] + conv_ai_ans_for_human(real_data_list[3:])

        ai_ans_list = conv_ai_ans_for_human(ai_ans_list)

        # В качестве ошибки просто добавляем разность между ответом ИИ и реальностью
        errors = np.array( np.abs(
            np.array(real_data_list[3:]) - np.array(ai_ans_list)
        ))
        total_errors.append(errors)


        # Выводим всё
        print(np.round(np.array(real_data_list[:3]),      1), "\t",
              np.round(np.array(real_data_list[3:]),      1), "\t",
              np.round(np.array(ai_ans_list),             1), "\t",
              np.round(np.array(errors),                       1))


    total_errors = np.array(total_errors)

    print("\nMean errors:",
          "\n\t Temperature:  ", np.round(np.mean(total_errors[:, 0]), 1),
          "\n\t Pressure:     ", np.round(np.mean(total_errors[:, 1]), 1),
          "\n\t Humidity:     ", np.round(np.mean(total_errors[:, 2]), 1),
          "\n\t Cloud:        ", np.round(np.mean(total_errors[:, 3]), 1),
          "\n\t Rain:         ", np.round(np.mean(total_errors[:, 4]), 1),
          "\n\n\t TOTAL:        ",np.round(np.mean(total_errors     ), 1))

    print("\n")



def print_weather_predict(ai, len_predict_days=3):
    print(f"Prediction for the next {len_predict_days} days:\t\t\t",
          f"(Temperature, Pressure, Humidity, Cloud, Raininess)")

    # Самое последнее - самое свежее
    # Также сразу подаём только amount_available_context данных для прогноза
    fresh_data = np.reshape(np.array(get_fresh_data(len_predict_days * 24)),
                            (len_predict_days * 24, 8))[::-1]
    times = fresh_data[:, :3].tolist()
    predicts_history = fresh_data[:, 3:].tolist()


    for _ in range(0, len_predict_days*24):
        # Делаем прогноз по всей истории, а потом отбираем один прогноз, относящееся к последней записи
        # preds_on_preds = ai.predict(np.array( [[t + p] for t, p in zip(times, predicts_history)] ), verbose=False)
        # ai_ans = np.reshape(np.array(preds_on_preds)[:, -1], (5))
        preds_on_preds = ai.predict(np.array([[times[-1] + predicts_history[-1]]]), verbose=False)
        ai_ans = np.reshape(np.array(preds_on_preds), (5))
        ai_ans = normalize(ai_ans, True)

        # predicts_history.append( (ai_ans + np.array(predicts_history[-1])).tolist() )
        predicts_history.append( (ai_ans + np.array(predicts_history)).tolist() )
        # predicts_history = predicts_history[1:]
        print(predicts_history)
        # Обновляем время
        time = times[-1]
        time[0] += 1/12                          # Увеличиваем часы
        time[1] += 1/15.5 if time[0] >1 else 0   # Увеличиваем день
        time[2] += 1/6    if time[1] >1 else 0   # Увеличиваем месяц

        # Следим, чтобы зачения не выходили за границы
        time = [-1 if i>1 else i for i in time]

        times.append(time)


        # Выводим прогноз
        time_for_human = [norm_hours(time[0], True),
                          norm_day(time[1], True),
                          norm_month(time[2], True)]

        pred_weather_for_human = conv_ai_ans_for_human(predicts_history[-1])
        print(f"{'{:02}'.format(time_for_human[1])}.{'{:02}'.format(time_for_human[2])}",
              f"{'{:02}'.format(time_for_human[0])}:00:\t",
              pred_weather_for_human[0], "℃\t",
              pred_weather_for_human[1], "mmHg\t",
              pred_weather_for_human[2], "%\t",
              pred_weather_for_human[3], "%\t",
              conv_rain_to_words(pred_weather_for_human[4]),
              )
    print("\n")
