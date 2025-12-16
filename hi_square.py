import json
import numpy as np
import math
from scipy.stats import chi2

# Параметры распределений (из кода генерации)
theta_uniform = 79  # параметр дискретного равномерного распределения
m_erlang = 3        # параметр формы распределения Эрланга
theta_erlang = 1/6  # параметр масштаба распределения Эрланга

def load_samples(filename='sample_generation.json'):
    """Загрузка сгенерированных выборок из файла"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sturges_rule(n):
    """Правило Старджесса для определения числа интервалов"""
    return max(2, 1 + int(np.floor(np.log2(n))))

def create_intervals(data, k, distribution_type, params=None):
    """
    Создание интервалов группировки для различных распределений
    
    Parameters:
    -----------
    data : list
        Выборка данных
    k : int
        Число интервалов
    distribution_type : str
        Тип распределения ('discrete_uniform' или 'erlang')
    params : dict
        Параметры распределения
    
    Returns:
    --------
    intervals : list of tuples
        Границы интервалов
    """
    if distribution_type == 'discrete_uniform':
        # Дискретное равномерное: значения от 1 до theta
        theta = params.get('theta', theta_uniform)
        # Создаем k интервалов примерно равной длины
        step = theta / k
        intervals = []
        for i in range(k):
            start = 1 + i * step
            end = 1 + (i + 1) * step
            if i == k - 1:  # Последний интервал включаем до theta
                end = theta + 0.5  # +0.5 для включения целых чисел
            intervals.append((start, end))
        return intervals
    
    elif distribution_type == 'erlang':
        # Распределение Эрланга: непрерывное на (0, ∞)
        # Используем квантили для создания равновероятных интервалов
        m = params.get('m', m_erlang)
        theta = params.get('theta', theta_erlang)
        
        # Для распределения Эрланга нет простой обратной функции CDF,
        # поэтому используем эмпирические квантили выборки
        sorted_data = sorted(data)
        n = len(data)
        
        # Определяем границы интервалов по квантилям
        intervals = []
        for i in range(k):
            lower_idx = int(np.floor(i * n / k))
            upper_idx = int(np.floor((i + 1) * n / k)) - 1
            
            if i == 0:
                lower_bound = sorted_data[0] - 1e-6
            else:
                lower_bound = intervals[-1][1]
            
            if i == k - 1:
                upper_bound = sorted_data[-1] + 1e-6
            else:
                upper_bound = sorted_data[upper_idx]
            
            intervals.append((lower_bound, upper_bound))
        
        return intervals
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def theoretical_probability(interval, distribution_type, params=None):
    """
    Вычисление теоретической вероятности попадания в интервал
    
    Parameters:
    -----------
    interval : tuple
        Границы интервала (a, b)
    distribution_type : str
        Тип распределения
    params : dict
        Параметры распределения
    
    Returns:
    --------
    probability : float
        Теоретическая вероятность
    """
    a, b = interval
    
    if distribution_type == 'discrete_uniform':
        theta = params.get('theta', theta_uniform)
        # Для дискретного равномерного: P = (кол-во целых чисел в интервале) / theta
        # Учитываем, что значения целые от 1 до theta
        count = 0
        for x in range(1, theta + 1):
            if a <= x < b:
                count += 1
        return count / theta
    
    elif distribution_type == 'erlang':
        # Для непрерывного распределения Эрланга
        m = params.get('m', m_erlang)
        theta = params.get('theta', theta_erlang)
        
        # Вычисляем вероятность через CDF распределения Эрланга
        # CDF(x) = 1 - e^{-θx} * Σ_{k=0}^{m-1} (θx)^k / k!
        from scipy.stats import gamma
        
        # Распределение Эрланга - частный случай гамма-распределения
        # с целым параметром формы
        cdf_b = gamma.cdf(b, m, scale=1/theta)
        cdf_a = gamma.cdf(a, m, scale=1/theta)
        
        return cdf_b - cdf_a
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def pearson_chi2_statistic(data, intervals, distribution_type, params=None):
    """
    Вычисление статистики хи-квадрат Пирсона
    
    Parameters:
    -----------
    data : list
        Выборка данных
    intervals : list of tuples
        Границы интервалов
    distribution_type : str
        Тип распределения
    params : dict
        Параметры распределения
    
    Returns:
    --------
    chi2_value : float
        Значение статистики хи-квадрат
    observed_freq : list
        Наблюдаемые частоты
    expected_freq : list
        Ожидаемые частоты
    """
    n = len(data)
    k = len(intervals)
    
    # Подсчет наблюдаемых частот
    observed_freq = [0] * k
    for value in data:
        for i, (a, b) in enumerate(intervals):
            if a <= value < b:
                observed_freq[i] += 1
                break
        else:
            # Если значение не попало ни в один интервал (крайний случай)
            if value >= intervals[-1][1]:
                observed_freq[-1] += 1
    
    # Вычисление теоретических вероятностей и ожидаемых частот
    expected_freq = []
    probs = []
    for interval in intervals:
        p = theoretical_probability(interval, distribution_type, params)
        probs.append(p)
        expected_freq.append(n * p)
    
    # Вычисление статистики хи-квадрат
    chi2_value = 0
    for obs, exp in zip(observed_freq, expected_freq):
        if exp > 0:  # Избегаем деления на ноль
            chi2_value += (obs - exp) ** 2 / exp
        # Если exp = 0 и obs = 0, слагаемое равно 0
        # Если exp = 0 и obs > 0, это проблема - объединяем интервалы
    
    return chi2_value, observed_freq, expected_freq, probs

def chi2_test(sample, distribution_type, k_values=None, alpha=0.05, params=None):
    """
    Проведение теста хи-квадрат для выборки с различными вариантами группировки
    """
    n = len(sample)
    
    if k_values is None:
        # По умолчанию: Старджесс и несколько вариантов вокруг него
        k_sturges = sturges_rule(n)
        k_values = [
            max(2, int(k_sturges * 0.5)),  # Половина от Старджесса
            k_sturges,                     # По правилу Старджесса
            min(n // 5, int(k_sturges * 1.5)),  # В 1.5 раза больше
            min(n // 3, int(k_sturges * 2)),    # В 2 раза больше
        ]
        k_values = sorted(set(k_values))  # Убираем дубликаты
    
    results = []
    
    for k in k_values:
        if k < 2:
            continue  # Минимум 2 интервала
        
        # Создаем интервалы
        intervals = create_intervals(sample, k, distribution_type, params)
        
        # Вычисляем статистику Пирсона
        chi2_stat, observed, expected, probs = pearson_chi2_statistic(
            sample, intervals, distribution_type, params
        )
        
        # Число степеней свободы
        df = k - 1
        
        # Критическое значение
        critical_value = float(chi2.ppf(1 - alpha, df))
        
        # P-значение
        p_value = float(1 - chi2.cdf(chi2_stat, df))
        
        # Проверяем условие ожидаемых частот (не менее 5 в каждом интервале)
        valid_test = all(exp >= 5 for exp in expected)
        
        # Решение: отвергаем H0 если chi2_stat > critical_value
        reject_h0 = float(chi2_stat) > critical_value
        
        results.append({
            'k': int(k),
            'chi2_statistic': float(chi2_stat),
            'degrees_of_freedom': int(df),
            'critical_value': critical_value,
            'p_value': p_value,
            'reject_h0': bool(reject_h0),  # Явное преобразование в bool
            'valid_test': bool(valid_test),  # Явное преобразование в bool
            'observed_freq': [int(x) for x in observed],
            'expected_freq': [float(x) for x in expected],
            'theoretical_probs': [float(x) for x in probs],
            'intervals': intervals
        })
    
    return results

def analyze_all_samples(data, alpha=0.05):
    """
    Анализ всех сгенерированных выборок
    """
    all_results = {}
    
    # Параметры для каждого распределения
    distribution_params = {
        'discrete_uniform': {'theta': theta_uniform},
        'erlang': {'m': m_erlang, 'theta': theta_erlang}
    }
    
    for dist_name, dist_data in data.items():
        print(f"\n{'='*60}")
        print(f"Анализ распределения: {dist_name}")
        print(f"{'='*60}")
        
        dist_results = {}
        
        for size, samples in dist_data.items():
            print(f"\nРазмер выборки: {size}")
            print(f"Количество выборок: {len(samples)}")
            
            size_results = []
            
            for sample_idx, sample in enumerate(samples[:3]):  # Анализируем первые 3 выборки
                print(f"\n  Выборка #{sample_idx + 1}:")
                
                # Вычисляем различные варианты k
                n = len(sample)
                k_sturges = sturges_rule(n)
                
                # Определяем диапазон k для тестирования
                k_min = max(2, int(k_sturges * 0.3))
                k_max = min(n // 3, int(k_sturges * 3))
                
                # Создаем список различных k для тестирования
                step = max(1, (k_max - k_min) // 5)
                k_values = list(range(k_min, k_max + 1, step))
                if k_sturges not in k_values:
                    k_values.append(k_sturges)
                k_values = sorted(k_values)
                
                # Проводим тест
                results = chi2_test(
                    sample, 
                    dist_name,
                    k_values=k_values,
                    alpha=alpha,
                    params=distribution_params[dist_name]
                )
                
                # Сохраняем результаты
                sample_result = {
                    'sample_index': int(sample_idx),
                    'sample_size': int(n),
                    'k_sturges': int(k_sturges),
                    'test_results': results
                }
                size_results.append(sample_result)
                
                # Выводим краткую информацию
                print(f"    Правило Старджесса: k = {k_sturges}")
                print(f"    Тестируемые значения k: {k_values}")
                
                # Находим результат для k по правилу Старджесса
                for res in results:
                    if res['k'] == k_sturges:
                        print(f"    χ² статистика (k={k_sturges}): {res['chi2_statistic']:.4f}")
                        print(f"    Критическое значение: {res['critical_value']:.4f}")
                        print(f"    P-значение: {res['p_value']:.4f}")
                        decision = 'Отвергаем H0' if res['reject_h0'] else 'Не отвергаем H0'
                        print(f"    Решение: {decision}")
                        valid = 'Да' if res['valid_test'] else 'Нет'
                        print(f"    Тест валиден: {valid}")
                        break
            
            dist_results[str(size)] = size_results  # Сохраняем размер как строку
        
        all_results[dist_name] = dist_results
    
    return all_results

def save_results_to_file(results, filename='chi2_test_results.json'):
    """Сохранение результатов в файл JSON"""
    # Преобразуем результаты для сериализации
    serializable_results = {}
    
    for dist_name, dist_data in results.items():
        serializable_results[dist_name] = {}
        
        for size, size_data in dist_data.items():
            serializable_results[dist_name][str(size)] = []  # Преобразуем размер в строку
            
            for sample_result in size_data:
                # Преобразуем результаты теста
                test_results_serializable = []
                for test_res in sample_result['test_results']:
                    serializable_test = {
                        'k': int(test_res['k']),
                        'chi2_statistic': float(test_res['chi2_statistic']),
                        'degrees_of_freedom': int(test_res['degrees_of_freedom']),
                        'critical_value': float(test_res['critical_value']),
                        'p_value': float(test_res['p_value']),
                        'reject_h0': bool(test_res['reject_h0']),  # Преобразуем в bool
                        'valid_test': bool(test_res['valid_test']),  # Преобразуем в bool
                        'observed_freq': [int(x) for x in test_res['observed_freq']],
                        'expected_freq': [float(x) for x in test_res['expected_freq']],
                        'theoretical_probs': [float(x) for x in test_res['theoretical_probs']],
                        'intervals': [(float(a) if a is not None else None, 
                                      float(b) if b is not None else None) 
                                     for a, b in test_res['intervals']]
                    }
                    test_results_serializable.append(serializable_test)
                
                serializable_sample = {
                    'sample_index': int(sample_result['sample_index']),
                    'sample_size': int(sample_result['sample_size']),
                    'k_sturges': int(sample_result['k_sturges']),
                    'test_results': test_results_serializable
                }
                serializable_results[dist_name][str(size)].append(serializable_sample)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в файл: {filename}")
    return serializable_results

def print_detailed_report(results):
    """Вывод детального отчета по результатам тестирования"""
    print("\n" + "="*80)
    print("ДЕТАЛЬНЫЙ ОТЧЕТ ПО РЕЗУЛЬТАТАМ ТЕСТА ХИ-КВАДРАТ")
    print("="*80)
    
    for dist_name, dist_data in results.items():
        print(f"\n{'='*60}")
        print(f"РАСПРЕДЕЛЕНИЕ: {dist_name.upper()}")
        print(f"{'='*60}")
        
        for size, size_data in dist_data.items():
            print(f"\nРазмер выборки: {size}")
            print("-"*40)
            
            for sample_result in size_data:
                sample_idx = sample_result['sample_index'] + 1
                n = sample_result['sample_size']
                k_sturges = sample_result['k_sturges']
                
                print(f"\n  Выборка #{sample_idx} (N={n}):")
                print(f"  Правило Старджесса рекомендует k = {k_sturges}")
                
                # Создаем таблицу результатов для разных k
                print(f"\n  {'k':^6} {'χ²':^12} {'df':^6} {'χ² крит':^10} {'p-value':^10} {'Решение':^12} {'Валиден':^8}")
                print(f"  {'-'*6} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
                
                for test_res in sample_result['test_results']:
                    k = test_res['k']
                    chi2_val = test_res['chi2_statistic']
                    df = test_res['degrees_of_freedom']
                    crit = test_res['critical_value']
                    pval = test_res['p_value']
                    decision = "Отвергнуть" if test_res['reject_h0'] else "Принять"
                    valid = "Да" if test_res['valid_test'] else "Нет"
                    
                    # Подсвечиваем строку с k по правилу Старджесса
                    if k == k_sturges:
                        star = "*"
                    else:
                        star = " "
                    
                    print(f"  {star}{k:^5} {chi2_val:^12.4f} {df:^6} {crit:^10.4f} {pval:^10.4f} {decision:^12} {valid:^8}")
                
                # Анализ чувствительности к выбору k
                print(f"\n  Анализ чувствительности к выбору k:")
                reject_counts = sum(1 for res in sample_result['test_results'] if res['reject_h0'])
                total_tests = len(sample_result['test_results'])
                print(f"    Количество тестов: {total_tests}")
                print(f"    Тестов с отвержением H0: {reject_counts}")
                print(f"    Доля отвержений: {reject_counts/total_tests:.2%}")
                
                # Проверяем согласованность решений
                decisions = [res['reject_h0'] for res in sample_result['test_results']]
                if all(decisions) or not any(decisions):
                    print(f"    Решения согласованы для всех k")
                else:
                    print(f"    Решения НЕ согласованы для разных k")
                    
# Основная функция для запуска анализа
def main():
    """Основная функция выполнения анализа"""
    print("ЗАГРУЗКА ДАННЫХ...")
    data = load_samples('sample_generation.json')
    
    print("ПРОВЕДЕНИЕ ТЕСТА ХИ-КВАДРАТ...")
    results = analyze_all_samples(data, alpha=0.05)
    
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
    save_results_to_file(results, 'chi2_test_results.json')
    
    print("ВЫВОД ДЕТАЛЬНОГО ОТЧЕТА...")
    print_detailed_report(results)
    
    print("\n" + "="*80)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
    print("="*80)

if __name__ == "__main__":
    main()