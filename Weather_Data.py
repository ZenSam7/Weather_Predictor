


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
    """

    DATA = []
    for NAME_DATASET in ["Москва (ВДНХ)", "Москва (Центр)", "Москва (Аэропорт)"]:
        with open(f"Datasets/{NAME_DATASET}.csv") as dataset:
            # Без первой строки с начальным символом;
            # Без первой (идём от староого к новому) записи, т.к. она используется для смещения ответа
            # (т.е. чтобы мы на основе предыдущей записи создавали следующую)
            records = dataset.readlines()[1:][::-1]
            len_file = len(records)
            num_data = 0

            for string in records:
                data = string.split(";")[:-2]

                # Если попался брак, то пропускаем шаг
                if '' in data or len(data) != 7:
                    continue

                processed_data = [0 for _ in range(7)]

                # Преобразуем строку
                # data[0] -> часы (в течении дня)
                # datap[1] -> день
                # data[2] -> месяц
                processed_data[0] = int(data[0][11:13])
                processed_data[1] = int(data[0][:2])
                processed_data[2] = int(data[0][3:5])
                processed_data[3] = float(data[1].replace(",", "."))
                processed_data[4] = float(data[2].replace(",", "."))
                processed_data[5] = int(data[3])
                processed_data[6] = int(data[4])

                DATA.append(processed_data)
    return DATA


def get_plank_history():
    """
    <- inp
    0) Время
    1) Давление (паскаль)
    2) ℃
    3) Температура (Кельвин)
    4) Точка росы
    5) Относительная влажность
    6) Давление пара насыщения
    7) Давление газа
    8) Дефицит давления пара
    9) Удельная влажность
    10) Концентрация водяного пара
    11) Герметичный (?)
    12) Скорость ветра (м/с)
    13) Максимальная скорость ветра
    14) Направление ветра в градусах


    -> out
    0) Время (часы) (float)
    1) Время (день)
    2) Время (месяц)
    3) ℃
    4) Давление (мм.рт.ст.)
    5) Влажность (%)
    6) Скорость ветра (м/с)
    """

    DATA = []
    with open("Datasets/max_planck_weather_ts.txt") as file:
        for string in file.readlines():
            data = string.split(",")

            # Если попался брак, то пропускаем шаг
            if '' in data or len(data) != 15: continue

            processed_data = [0 for _ in range(7)]

            processed_data[0] = int(data[0][11:13]) + int(data[0][14:16]) /60
            processed_data[1] = int(data[0][:2])
            processed_data[2] = int(data[0][3:5])
            processed_data[3] = float(data[2])
            processed_data[4] = float(data[1]) *100 / 133.322
            processed_data[5] = float(data[5])
            processed_data[6] = float(data[12]) *10

            DATA.append(processed_data)
    return DATA


def get_weather_history():
    """
    <- inp
    0) Время
    1) Описание
    2) Тип осадков
    3) Температура ℃
    4) Кажущаяся температура ℃
    5) Влажность
    6) Скорость ветра (km/h)
    7) Направление ветра °
    8) Видимость (km)
    9) Loud Cover
    10) Давление (миллибар)
    11) Ежедневная сводка


    -> out
    0) Время (часы)
    1) Время (день)
    2) Время (месяц)
    3) Температура
    4) Давление (мм.рт.ст.)
    5) Влажность (%)
    6) Скорость ветра (м/c)
    """

    DATA = []
    with open("Datasets/weatherHistory.txt") as file:
        for string in file.readlines():
            data = string.split(",")

            # Если попался брак, то пропускаем шаг
            if '' in data or len(data) != 12: continue

            processed_data = [0 for _ in range(7)]

            processed_data[0] = int(data[0][11:13])
            processed_data[1] = int(data[0][8:10])
            processed_data[2] = int(data[0][5:7])
            processed_data[3] = float(data[3])
            processed_data[4] = float(data[10]) * 0.750063755419211
            processed_data[5] = float(data[5]) * 100
            processed_data[6] = float(data[6]) * 1_000 / 3_600

            DATA.append(processed_data)
    return DATA
