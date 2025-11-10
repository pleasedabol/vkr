import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from io import StringIO
    import lets_plot as lp
    from lets_plot import ggplot, geom_line, labs, ggsize, aes, scale_color_manual, geom_hline, geom_vline
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.signal import find_peaks
    from lets_plot.mapping import as_discrete
    import lmfit
    from lmfit import Model
    return (
        Model,
        StringIO,
        aes,
        find_peaks,
        geom_line,
        ggplot,
        ggsize,
        labs,
        lp,
        mo,
        np,
        pl,
        savgol_filter,
        scale_color_manual,
    )


@app.cell
def _(StringIO, pl):
    def process_coordinates_file(file_content):
        try:
            # Очищаем файл от лишних пробелов и табуляций
            cleaned_lines = []
            for line in file_content.split('\n'):
                if line.strip():  # Пропускаем пустые строки
                    # Разбиваем по пробелам и удаляем пустые элементы
                    parts = [part for part in line.split(' ') if part.strip()]
                    if len(parts) >= 2:  # Берем только первые два числа
                        cleaned_lines.append(f"{parts[0]} {parts[1]}")

            # Объединяем обратно в строку
            cleaned_content = '\n'.join(cleaned_lines)

            # Читаем очищенное содержимое
            df = pl.read_csv(
                StringIO(cleaned_content),
                has_header=False,
                separator=' ',
                new_columns=['x_coord', 'y_coord'],
                truncate_ragged_lines=True  # Игнорируем лишние колонки
            )

            # Извлекаем массивы координат
            x_coordinates = df['x_coord'].to_list()
            y_coordinates = df['y_coord'].to_list()

            return df, x_coordinates, y_coordinates, None

        except Exception as e:
            return None, None, None, f"Ошибка обработки файла: {str(e)}"
    return (process_coordinates_file,)


@app.cell
def _(mo):
    # Создаем элемент загрузки файла
    file_upload = mo.ui.file(
        filetypes=[".txt"],
        multiple=False,
        label="Загрузите файл с координатами (.txt)"
    )

    # Объединяем все элементы в один вывод
    mo.vstack([
        mo.md("# 📁 Загрузка файла с координатами"),
        mo.md("## 📤 Загрузка файла"),
        mo.md("Загрузите текстовый файл с координатами в формате: x1 y1 x2 y2, x3 y3..."),
        file_upload
    ])
    return (file_upload,)


@app.cell
def _(file_upload, mo, process_coordinates_file):
    # Объявляем переменные в глобальной области видимости
    df = None
    x_coords = None
    y_coords = None
    error = None
    file_info = None
    y_done = []
    # Проверяем, загружен ли файл
    if file_upload.value:
        # Получаем содержимое файла
        file_content_bytes = file_upload.value[0].contents
        file_content = file_content_bytes.decode('utf-8')
        file_info = f"Загружен файл: {file_upload.value[0].name}"

        # Обрабатываем данные
        df, x_coords, y_coords, error = process_coordinates_file(file_content)

    # Создаем результат для отображения
    if file_upload.value:
        if error:
            result_display = mo.md(f"**❌ Ошибка:** {error}")
        elif df is not None:
            # Упрощенный вариант без колонок
            result_display = mo.vstack([
                mo.md("## 📊 Результаты обработки"),
                mo.md(file_info),
                mo.md("### Таблица данных"),
                mo.ui.table(df),
                mo.md(f"**Всего обработано точек:** {len(df)}")
            ])
        else:
            result_display = mo.md("**⚠️ Не удалось обработать файл**")
    else:
        result_display = mo.md("**⏳ Ожидание загрузки файла...**")

    # Отображаем результат
    result_display
    return df, x_coords, y_coords


@app.cell
def _(aes, df, geom_line, ggplot, ggsize, labs, lp, mo, x_coords, y_coords):
    # Проверяем, что данные загружены
    if df is not None:
        try:
            lp.LetsPlot.setup_html()

            # Создаем данные для Lets-Plot
            plot_data = {
                'x_coord': x_coords,
                'y_coord': y_coords
            }

            # Строим график
            plot = ggplot(plot_data) + \
                   geom_line(aes(x='x_coord', y='y_coord'), color='blue', size=1) + \
                   labs(x='X координата', y='Y координата', title='График координат') + \
                   ggsize(1000, 500)

            # ВСЕ в одном vstack
            result_display1 = mo.vstack([
                mo.md("## 📈 График данных (Lets-Plot)"),
                plot
            ])

        except ImportError:
            result_display1 = mo.vstack([
                mo.md("## 📦 Требуется установка Lets-Plot"),
                mo.md("**Установите:** `pip install lets-plot`"),
                mo.md("После установки перезапустите ячейку")
            ])
        except Exception as e:
            result_display1 = mo.vstack([
                mo.md("## ❌ Ошибка построения графика"),
                mo.md(str(e))
            ])
            print(e)
    else:
        result_display1 = mo.md("**⏳ Данные еще не загружены**")

    result_display1
    return


@app.cell
def _(mo):
    iterations = mo.ui.slider(
            start=5, stop=150, step=5, value=30,
            label="Количество итераций SNIP"
        )
    iterations
    return (iterations,)


@app.cell
def _(
    aes,
    df,
    geom_line,
    ggplot,
    ggsize,
    iterations,
    labs,
    lp,
    mo,
    np,
    x_coords,
    y_coords,
):

    # --- Реализация SNIP ---
    def snip_baseline(y, iterations=30):
        """Реализация SNIP (Statistics-sensitive Nonlinear Iterative Peak-clipping)"""
        y = np.array(y, dtype=float)
        L = len(y)
        baseline = y.copy()

        for k in range(1, iterations + 1):
            temp = baseline.copy()
            for i in range(k, L - k):
                avg = 0.5 * (temp[i - k] + temp[i + k])
                if baseline[i] > avg:
                    baseline[i] = avg
        return baseline

    if df is not None and 'iterations' in locals():
        lp.LetsPlot.setup_html()
        # Вычисляем baseline
        baseline = snip_baseline(y_coords, iterations=iterations.value)
        y_corrected = np.array(y_coords) - baseline

        # Данные для графиков
        plot_dataa = {
            "x": x_coords * 3,
            "y": y_coords + baseline.tolist() + y_corrected.tolist(),
            "type": (["Raw data"] * len(x_coords)) +
                    (["Фон (SNIP)"] * len(x_coords)) +
                    (["Вычтенный фон"] * len(x_coords))
        }

        # График
        plot_1 = (
            ggplot(plot_dataa)
            + geom_line(aes(x="x", y="y", color="type"), size=1)
            + labs(x="X", y="Интенсивность", title=f"SNIP (итераций: {iterations.value})", color="Тип")
            + ggsize(1000, 500)
        )

        result = mo.vstack([
            mo.md("## 🧮 SNIP определение фона"),
            plot_1
        ])
    else:
        result = mo.md("**⏳ Ожидание сглаженных данных...**")
        y_corrected = np.array([1,2])
    y_corrected
    result
    return (y_corrected,)


@app.cell
def _(df, mo):
    # Импортируем необходимые библиотеки
    try:
        scipy_available = True
    except ImportError:
        scipy_available = False

    if df is not None and scipy_available:
        # Создаем интерфейс для выбора параметров фильтра
        window_length = mo.ui.slider(
            start=7, 
            stop=51, 
            step=2,  # только нечетные значения
            value=11,
            label="Длина окна (только нечетные числа)"
        )

        polyorder = mo.ui.slider(
            start=1, 
            stop= 6, 
            value=3,
            label="Порядок полинома"
        )

        # ВСЕ в одном vstack
        result_display6 = mo.vstack([
            mo.md("## 🔧 Настройка фильтра Савицкого-Голея для сглаживания графика"),
            window_length,
            polyorder
        ])
    else:
        if df is None:
            result_display6 = mo.md("**⏳ Данные еще не загружены**")
        else:
            result_display6 = mo.md("**📦 Установите scipy:** `pip install scipy`")

    result_display6
    return polyorder, scipy_available, window_length


@app.cell(hide_code=True)
def _(
    aes,
    df,
    geom_line,
    ggplot,
    ggsize,
    labs,
    lp,
    mo,
    polyorder,
    savgol_filter,
    scale_color_manual,
    scipy_available,
    window_length,
    x_coords,
    y_corrected,
):
    y_savgol = []
    if df is not None and scipy_available and 'window_length' in locals() and 'polyorder' in locals():
        try:        
            lp.LetsPlot.setup_html()

            # Получаем значения параметров
            wl = window_length.value
            po = polyorder.value

            # Применяем фильтр Савицкого-Голея
            y_smoothed = savgol_filter(y_corrected, window_length=wl, polyorder=po)
            y_savgol = y_smoothed.tolist()
            # Создаем данные для графика
            plot_data_2 = {
                'x': x_coords * 2,  # Удваиваем для двух линий
                'y': y_corrected.tolist() + y_smoothed.tolist(),
                'type': ['Исходные'] * len(x_coords) + ['Сглаженные'] * len(x_coords)
            }

            # Строим сравнительный график
            comparison_plot = ggplot(plot_data_2) + \
                   geom_line(aes(x='x', y='y', color='type'), size=1) + \
                   labs(
                       x='X координата', 
                       y='Y координата', 
                       title=f'Фильтр Савицкого-Голея (окно: {wl}, порядок: {po})',
                       color='Тип данных'
                   ) + \
                   ggsize(1000, 500) + \
                   scale_color_manual(values=['gray', 'blue'])

            # График только сглаженных данных
            smoothed_plot = ggplot({'x': x_coords, 'y': y_smoothed}) + \
                   geom_line(aes(x='x', y='y'), color='red', size=1) + \
                   labs(x='X координата', y='Y координата', title='Только сглаженные данные') + \
                   ggsize(1000, 500)

            # ВСЕ в одном vstack
            result_display7 = mo.vstack([
                mo.md("## 📊 Сравнение исходных и сглаженных данных"),
                mo.md(f"**Параметры:** окно = {wl}, порядок полинома = {po}"),
                comparison_plot,
                mo.md("### 📈 Только сглаженные данные"),
                smoothed_plot
            ])

        except Exception as e:
            result_display7 = mo.vstack([
                mo.md("## ❌ Ошибка применения фильтра"),
                mo.md(str(e))
            ])
    elif not scipy_available:
        result_display7 = mo.md("**📦 Установите scipy:** `pip install scipy`")
    else:
        result_display7 = mo.md("**⏳ Настройте параметры фильтра в предыдущих ячейках**")

    result_display7
    return (y_savgol,)


@app.cell
def _(find_peaks, mo, x_coords, y_corrected, y_savgol):
    peaks, properties = find_peaks(y_savgol, prominence=20, width=5, height=50)
    peaks_x = []
    for i in range (0, len(peaks)):
        peaks_x.append(x_coords[peaks[i]])
    peaks_y = y_corrected[peaks]  # Значения по оси Y для пиков
    for i in range (0, len(peaks)):
        print(f'{peaks_x[i]}, {peaks_y[i]}')
       # Итоговый вывод
    res = mo.vstack([
        mo.md(f"**Найдено пиков:** {len(peaks)}")
    ])
    res
    return (peaks,)


@app.cell(hide_code=True)
def _(np):
    # Функция для Псевдо-Войгт
    def pseudo_voigt(x, center, amplitude, width, eta):
        """Псевдо-Войгт функция"""
        # Гауссов компонент
        gauss = amplitude * np.exp(-4 * np.log(2) * ((x - center) / width)**2)

        # Лоренцев компонент
        lorentz = (amplitude / np.pi) * (width / ((x - center)**2 + (width / 2)**2))

        # Псевдо-Войгт
        return (1 - eta) * gauss + eta * lorentz
    return (pseudo_voigt,)


@app.cell
def _(mo):
    smooth_win_slider = mo.ui.slider(start=5, stop=101, step=2, value=13, label="Окно сглаживания (границы пика)")
    slope_frac_slider = mo.ui.slider(start=0.001, stop=2.5, step=0.005, value=0.02, label="Порог наклона (границы пика)")
    max_comps_slider = mo.ui.slider(start=1, stop=5, step=1, value=3, label="Максимум компонент на пик")
    result_ = mo.vstack([

        smooth_win_slider, slope_frac_slider, max_comps_slider])
    result_
    return max_comps_slider, slope_frac_slider, smooth_win_slider


@app.cell
def _(
    Model,
    aes,
    find_peaks,
    geom_line,
    ggplot,
    ggsize,
    labs,
    lp,
    max_comps_slider,
    mo,
    np,
    peaks,
    pseudo_voigt,
    savgol_filter,
    slope_frac_slider,
    smooth_win_slider,
    x_coords,
    y_corrected,
):
    # --- 1. Вспомогательные функции ---
    def find_bounds_by_slope_peaks(x, y, peak_idx, smooth_win=31, polyorder=2,
                                   slope_frac=0.02, N_consec=3, min_width_pts=5, expand_factor=2):
        """Находим границы пика по производной (устойчивая версия, с адаптивным порогом)"""
        try:
            if len(y) < 3:
                return None
            if smooth_win >= len(y):
                smooth_win = max(3, len(y)//2*2+1)
            if smooth_win % 2 == 0:
                smooth_win += 1

            y_smooth = savgol_filter(y, window_length=smooth_win, polyorder=min(polyorder, smooth_win - 2))
            dy = np.gradient(y_smooth, x)

            # вычисляем локальную амплитуду пика
            peak_amp = y[peak_idx] - np.median(y)
            # более адаптивный порог — зависит от амплитуды, а не от глобального max(|dy|)
            s_thresh =slope_frac * abs(peak_amp)

            idx, consec, left_idx = peak_idx, 0, peak_idx
            while idx > 0:
                if dy[idx] < -s_thresh:
                    consec = 0
                    left_idx = idx
                else:
                    consec += 1
                    if consec >= N_consec:
                        break
                idx -= 1
            left_idx = max(0, left_idx)

            idx, consec, right_idx = peak_idx, 0, peak_idx
            while idx < len(y) - 1:
                if dy[idx] > s_thresh:
                    consec = 0
                    right_idx = idx
                else:
                    consec += 1
                    if consec >= N_consec:
                        break
                idx += 1
            right_idx = min(len(y) - 1, right_idx)

            # если всё равно слишком узко — расширяем вручную
            if right_idx - left_idx < min_width_pts:
                left_idx = max(0, peak_idx - min_width_pts * expand_factor)
                right_idx = min(len(y) - 1, peak_idx + min_width_pts * expand_factor)

            return int(left_idx), int(right_idx), y_smooth
        except Exception as e:
            print(f"[WARN] find_bounds_by_slope_peaks error at peak {peak_idx}: {e}")
            return None



    def iterative_multi_fit_region_peaks(x_region, y_region, max_components=4,
                                         resid_prom_frac=0.18, rel_improve_thresh=0.05):
        """Итеративно добавляем компоненты Псевдо-Войгта, если остаток содержит пики"""
        model_local = None
        params_local = None
        comps_local = []
        fit_y_local = np.zeros_like(y_region)
        prev_rms = np.inf

        # ищем первый пик
        pks_local, props_local = find_peaks(y_region, prominence=np.std(y_region) * 1.5)
        if len(pks_local) == 0:
            return None, [], fit_y_local
        centers_local = [x_region[pks_local[np.argmax(props_local["prominences"])]]]

        for comp_idx in range(max_components):
            center0 = centers_local[-1]
            prefix = f"pv{comp_idx}_"

            new_model_local = Model(pseudo_voigt, prefix=prefix)
            if model_local is None:
                model_local = new_model_local
                params_local = new_model_local.make_params(center=center0, amplitude=np.max(y_region),
                                                           width=(x_region[-1] - x_region[0]) / 6, eta=0.5)
            else:
                model_local = model_local + new_model_local
                params_local.update(new_model_local.make_params(center=center0, amplitude=np.max(y_region) / 3,
                                                                width=(x_region[-1] - x_region[0]) / 6, eta=0.5))

            params_local[prefix + "width"].min = (x_region[1] - x_region[0])
            params_local[prefix + "width"].max = (x_region[-1] - x_region[0]) * 2

            fit_result_local = model_local.fit(y_region, params_local, x=x_region)
            fit_y_local = fit_result_local.best_fit
            residual_local = y_region - fit_y_local
            rms_local = np.sqrt(np.mean(residual_local ** 2))
            rel_improve = (prev_rms - rms_local) / prev_rms if prev_rms != np.inf else 1.0
            prev_rms = rms_local

            # ищем пики в остатке
            rpks, rprops = find_peaks(residual_local, prominence=np.std(residual_local) * 1.2)
            resid_has_peak = False
            if len(rpks) > 0:
                max_prom = np.max(rprops["prominences"])
                resid_has_peak = max_prom > resid_prom_frac * np.max(y_region)

            comps_temp = []
            for name, param in fit_result_local.params.items():
                if name.endswith("_center"):
                    pref = name[:-7]  # оставляем префикс с подчёркиванием
                    comps_temp.append({
                        "center": fit_result_local.params[pref + "_center"].value,
                        "amplitude": fit_result_local.params[pref + "_amplitude"].value,
                        "width": fit_result_local.params[pref + "_width"].value,
                        "eta": fit_result_local.params[pref + "_eta"].value
                    })

            comps_local = comps_temp

            if (not resid_has_peak) or (rel_improve < rel_improve_thresh):
                break
            centers_local.append(x_region[rpks[np.argmax(rprops["prominences"])]])


    # --- 2. UI параметры ---

    # --- 3. Основной цикл обработки ---
    peaks_data_combined = []
    fits_data_combined = []

    for pk_i in sorted(peaks, key=lambda idx: y_corrected[idx]):
        bounds_result = find_bounds_by_slope_peaks(
            x_coords, y_corrected, pk_i,
            smooth_win=smooth_win_slider.value, slope_frac=slope_frac_slider.value
        )
        if bounds_result is None:
            continue

        left_idx, right_idx, y_smooth_local = bounds_result
        x_local = np.array(x_coords[left_idx:right_idx])
        print(f"→ Пик {pk_i}: диапазон ({left_idx}-{right_idx}), длина {len(x_local)}")
        if right_idx - left_idx < 5:
            continue  # слишком короткий участок

        y_local = np.array(y_corrected[left_idx:right_idx])

        # отладка

        fit_result_local, comps_local, fit_y_local = iterative_multi_fit_region_peaks(
            x_local, y_local, max_components=max_comps_slider.value
        )

        if fit_result_local is None or len(comps_local) == 0:
            print(f"⚠️  Пик {pk_i}: не удалось аппроксимировать.")
            continue

        # сохраним
        peaks_data_combined.extend(comps_local)
        fits_data_combined.append((x_local, fit_y_local))

    # --- 4. Построение графика ---
    lp.LetsPlot.setup_html()
    x_all, y_all, type_all = [], [], []

    # исходные данные
    x_all.extend(x_coords)
    y_all.extend(y_corrected)
    type_all.extend(["Исходные данные"] * len(x_coords))

    # аппроксимации
    for pk_i, (x_fit, y_fit) in enumerate(fits_data_combined):
        if len(x_fit) != len(y_fit):  # выравнивание
            min_len = min(len(x_fit), len(y_fit))
            x_fit, y_fit = x_fit[:min_len], y_fit[:min_len]
        x_all.extend(x_fit.tolist())
        y_all.extend(y_fit.tolist())
        type_all.extend([f"Аппроксимация {pk_i+1}"] * len(x_fit))

    plot_dataset_peaks = {"x": x_all, "y": y_all, "type": type_all}

    combined_plot_peaks = (
        ggplot(plot_dataset_peaks)
        + geom_line(aes(x="x", y="y", color="type"), size=1)
        + geom_line(aes(xintercept="x"),
                     data={"x": [p["center"] for p in peaks_data_combined]},
                     linetype="dashed", size=0.5)
        + labs(x="X", y="Интенсивность", title="Мульти-Псевдо-Войгт аппроксимация всех пиков")
        + ggsize(1000, 500)
    )

    # --- 5. Таблица ---
    if len(peaks_data_combined) == 0:
        table_md_peaks = "_Компоненты не найдены. Попробуйте снизить порог наклона или увеличить окно._"
    else:
        table_md_peaks = "\n".join([
            f"**Пик {pk_i + 1}:** Центр = {p['center']:.2f}, Амплитуда = {p['amplitude']:.2f}, "
            f"Ширина = {p['width']:.2f}, η = {p['eta']:.2f}"
            for pk_i, p in enumerate(peaks_data_combined)
        ])

    result_block_peaks = mo.vstack([
        mo.md("## 🧮 Аппроксимация перекрывающихся пиков (Псевдо-Войгт)"),
        smooth_win_slider, slope_frac_slider, max_comps_slider,
        combined_plot_peaks,
        mo.md("## 📋 Параметры найденных компонент"),
        mo.md(table_md_peaks)
    ])
    result_block_peaks


    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
