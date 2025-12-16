import json
import numpy as np
import math
from scipy.stats import chi2, gamma

# Параметры, которые СЧИТАЮТСЯ известными для проверки
# Но на самом деле мы их будем оценивать по выборке

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
    (Адаптированная версия из предыдущего кода)
    """
    if distribution_type == 'discrete_uniform':
        # Дискретное равномерное: значения от 1 до theta
        # Если theta неизвестен (сложная гипотеза), оцениваем по данным
        if params and 'theta' in params:
            theta = params['theta']
        else:
            # Оцениваем theta как максимальное значение + небольшой запас
            theta = max(data) + 1
        
        # Создаем k интервалов примерно равной длины
        step = theta / k
        intervals = []
        for i in range(k):
            start = 1 + i * step
            end = 1 + (i + 1) * step
            if i == k - 1:  # Последний интервал
                end = theta + 0.5
            intervals.append((start, end))
        return intervals
    
    elif distribution_type == 'erlang':
        # Распределение Эрланга: непрерывное на (0, ∞)
        # Используем квантили для создания интервалов
        sorted_data = sorted(data)
        n = len(data)
        
        # Определяем границы интервалов по квантилям
        intervals = []
        for i in range(k):
            lower_idx = int(np.floor(i * n / k))
            upper_idx = int(np.floor((i + 1) * n / k)) - 1
            
            if i == 0:
                lower_bound = max(0, sorted_data[0] - 1e-6)
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

def estimate_parameters(data, distribution_type):
    """
    Оценка параметров распределения по выборке
    
    Parameters:
    -----------
    data : list
        Выборка данных
    distribution_type : str
        Тип распределения
    
    Returns:
    --------
    params : dict
        Оцененные параметры
    """
    if distribution_type == 'discrete_uniform':
        # Для дискретного равномерного: θ = максимальное значение
        # Метод максимального правдоподобия
        theta_hat = int(max(data))
        return {'theta': theta_hat}
    
    elif distribution_type == 'erlang':
        # Для распределения Эрланга
        # m считается известным (из задания m=3)
        # θ оценивается через среднее: для Erlang(m, θ) среднее = m/θ
        m = 3  # Из задания
        mean_val = np.mean(data)
        theta_hat = m / mean_val
        return {'m': m, 'theta': theta_hat}
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def theoretical_probability_complex(interval, distribution_type, params):
    """
    Вычисление теоретической вероятности с ОЦЕНЕННЫМИ параметрами
    
    Parameters:
    -----------
    interval : tuple
        Границы интервала (a, b)
    distribution_type : str
        Тип распределения
    params : dict
        ОЦЕНЕННЫЕ параметры распределения
    
    Returns:
    --------
    probability : float
        Теоретическая вероятность
    """
    a, b = interval
    
    if distribution_type == 'discrete_uniform':
        theta_hat = params['theta']
        
        # Для дискретного равномерного: P = (кол-во целых чисел в интервале) / θ_hat
        count = 0
        for x in range(1, theta_hat + 1):
            if a <= x < b:
                count += 1
        return count / theta_hat
    
    elif distribution_type == 'erlang':
        # Для распределения Эрланга с оцененными параметрами
        m = params['m']
        theta_hat = params['theta']
        
        # Вычисляем вероятность через CDF гамма-распределения
        cdf_b = gamma.cdf(b, m, scale=1/theta_hat)
        cdf_a = gamma.cdf(a, m, scale=1/theta_hat)
        
        return cdf_b - cdf_a
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def pearson_chi2_complex(data, intervals, distribution_type):
    """
    Вычисление статистики хи-квадрат Пирсона для СЛОЖНОЙ гипотезы
    
    Parameters:
    -----------
    data : list
        Выборка данных
    intervals : list of tuples
        Границы интервалов
    distribution_type : str
        Тип распределения
    
    Returns:
    --------
    chi2_value : float
        Значение статистики хи-квадрат
    observed_freq : list
        Наблюдаемые частоты
    expected_freq : list
        Ожидаемые частоты
    params : dict
        Оцененные параметры
    """
    n = len(data)
    k = len(intervals)
    
    # 1. Оцениваем параметры по выборке
    params = estimate_parameters(data, distribution_type)
    
    # 2. Подсчет наблюдаемых частот
    observed_freq = [0] * k
    for value in data:
        for i, (a, b) in enumerate(intervals):
            if a <= value < b:
                observed_freq[i] += 1
                break
        else:
            if value >= intervals[-1][1]:
                observed_freq[-1] += 1
    
    # 3. Вычисление теоретических вероятностей с оцененными параметрами
    expected_freq = []
    probs = []
    for interval in intervals:
        p = theoretical_probability_complex(interval, distribution_type, params)
        probs.append(p)
        expected_freq.append(n * p)
    
    # 4. Вычисление статистики хи-квадрат
    chi2_value = 0
    for obs, exp in zip(observed_freq, expected_freq):
        if exp > 0:
            chi2_value += (obs - exp) ** 2 / exp
    
    return chi2_value, observed_freq, expected_freq, probs, params

def chi2_test_complex(sample, distribution_type, k_values=None, alpha=0.05):
    """
    Проведение теста хи-квадрат для СЛОЖНОЙ гипотезы
    
    Parameters:
    -----------
    sample : list
        Выборка данных
    distribution_type : str
        Тип проверяемого распределения
    k_values : list
        Список различных значений k (числа интервалов)
    alpha : float
        Уровень значимости
    
    Returns:
    --------
    results : list of dict
        Результаты теста для каждого k
    """
    n = len(sample)
    
    if k_values is None:
        k_sturges = sturges_rule(n)
        k_values = [
            max(2, int(k_sturges * 0.5)),
            k_sturges,
            min(n // 5, int(k_sturges * 1.5)),
            min(n // 3, int(k_sturges * 2)),
        ]
        k_values = sorted(set(k_values))
    
    results = []
    
    for k in k_values:
        if k < 2:
            continue
        
        # Создаем интервалы
        intervals = create_intervals(sample, k, distribution_type, None)
        
        # Вычисляем статистику для СЛОЖНОЙ гипотезы
        chi2_stat, observed, expected, probs, params = pearson_chi2_complex(
            sample, intervals, distribution_type
        )
        
        # Число степеней свободы для СЛОЖНОЙ гипотезы
        # Определяем число оцениваемых параметров
        if distribution_type == 'discrete_uniform':
            s = 1  # Оцениваем θ
        elif distribution_type == 'erlang':
            # В задании m=3 считается известным, оцениваем только θ
            s = 1
        else:
            s = 0
        
        df = k - 1 - s
        
        if df <= 0:
            continue  # Пропускаем случаи с неположительными степенями свободы
        
        # Критическое значение
        critical_value = float(chi2.ppf(1 - alpha, df))
        
        # P-значение
        p_value = float(1 - chi2.cdf(chi2_stat, df))
        
        # Проверяем условие ожидаемых частот
        valid_test = all(exp >= 5 for exp in expected)
        
        # Решение
        reject_h0 = float(chi2_stat) > critical_value
        
        results.append({
            'k': int(k),
            'chi2_statistic': float(chi2_stat),
            'degrees_of_freedom': int(df),
            'estimated_params': params,
            's': s,  # Число оцениваемых параметров
            'critical_value': critical_value,
            'p_value': p_value,
            'reject_h0': bool(reject_h0),
            'valid_test': bool(valid_test),
            'observed_freq': [int(x) for x in observed],
            'expected_freq': [float(x) for x in expected],
            'theoretical_probs': [float(x) for x in probs],
            'intervals': intervals
        })
    
    return results

def compare_simple_complex_hypotheses(data, alpha=0.05):
    """
    Сравнение простой и сложной гипотез для всех выборок
    
    Parameters:
    -----------
    data : dict
        Загруженные данные выборок
    alpha : float
        Уровень значимости
    
    Returns:
    --------
    comparison_results : dict
        Результаты сравнения
    """
    comparison_results = {}
    
    # Истинные параметры (из генерации данных)
    true_params = {
        'discrete_uniform': {'theta': 79},
        'erlang': {'m': 3, 'theta': 1/6}
    }
    
    for dist_name, dist_data in data.items():
        print(f"\n{'='*60}")
        print(f"СЛОЖНАЯ ГИПОТЕЗА: {dist_name}")
        print(f"Истинные параметры: {true_params[dist_name]}")
        print(f"{'='*60}")
        
        dist_results = {}
        
        for size, samples in dist_data.items():
            print(f"\nРазмер выборки: {size}")
            
            size_results = []
            
            # Анализируем первые 3 выборки каждого размера
            for sample_idx, sample in enumerate(samples[:3]):
                print(f"\n  Выборка #{sample_idx + 1} (N={len(sample)}):")
                
                n = len(sample)
                k_sturges = sturges_rule(n)
                
                # Тестируем с k по правилу Старджесса
                results = chi2_test_complex(
                    sample, 
                    dist_name,
                    k_values=[k_sturges],
                    alpha=alpha
                )
                
                if results:
                    result = results[0]
                    
                    # Выводим оцененные параметры
                    params = result['estimated_params']
                    print(f"    Оцененные параметры: {params}")
                    
                    # Сравнение с истинными параметрами
                    true_param = true_params[dist_name]
                    if 'theta' in params:
                        error = abs(params['theta'] - true_param['theta']) / true_param['theta'] * 100
                        print(f"    Ошибка оценки θ: {error:.2f}%")
                    
                    print(f"    χ² статистика: {result['chi2_statistic']:.4f}")
                    print(f"    Степени свободы: {result['degrees_of_freedom']} (k-1-s, s={result['s']})")
                    print(f"    Критическое значение: {result['critical_value']:.4f}")
                    print(f"    P-значение: {result['p_value']:.4f}")
                    
                    decision = 'Отвергаем H0' if result['reject_h0'] else 'Не отвергаем H0'
                    print(f"    Решение: {decision}")
                    
                    size_results.append({
                        'sample_index': sample_idx,
                        'sample_size': n,
                        'k_sturges': k_sturges,
                        'test_result': result
                    })
        
        comparison_results[dist_name] = dist_results
    
    return comparison_results

def analyze_power_complex_vs_simple(data, alpha=0.05):
    """
    Анализ мощности критерия для простой и сложной гипотез
    
    Parameters:
    -----------
    data : dict
        Загруженные данные выборок
    alpha : float
        Уровень значимости
    
    Returns:
    --------
    power_analysis : dict
        Анализ мощности
    """
    print("\n" + "="*80)
    print("СРАВНЕНИЕ ПРОСТОЙ И СЛОЖНОЙ ГИПОТЕЗ")
    print("="*80)
    
    # Загрузим результаты простой гипотезы из файла
    try:
        with open('chi2_test_results.json', 'r', encoding='utf-8') as f:
            simple_results = json.load(f)
    except FileNotFoundError:
        print("Файл с результатами простой гипотезы не найден")
        simple_results = {}
    
    power_analysis = {}
    
    for dist_name, dist_data in data.items():
        print(f"\n{'='*60}")
        print(f"РАСПРЕДЕЛЕНИЕ: {dist_name.upper()}")
        print(f"{'='*60}")
        
        rejections_simple = {}
        rejections_complex = {}
        
        # Анализируем первые 3 выборки каждого размера
        for size, samples in dist_data.items():
            print(f"\nРазмер выборки: {size}")
            print(f"{'-'*40}")
            
            rejections_simple[size] = []
            rejections_complex[size] = []
            
            for sample_idx, sample in enumerate(samples[:3]):
                n = len(sample)
                k_sturges = sturges_rule(n)
                
                # Тест для сложной гипотезы
                complex_results = chi2_test_complex(
                    sample, dist_name, k_values=[k_sturges], alpha=alpha
                )
                
                if complex_results:
                    complex_result = complex_results[0]
                    reject_complex = complex_result['reject_h0']
                    rejections_complex[size].append(reject_complex)
                    
                    # Ищем соответствующий результат для простой гипотезы
                    reject_simple = None
                    if dist_name in simple_results and str(size) in simple_results[dist_name]:
                        simple_size_results = simple_results[dist_name][str(size)]
                        if sample_idx < len(simple_size_results):
                            simple_sample_results = simple_size_results[sample_idx]
                            for test_res in simple_sample_results['test_results']:
                                if test_res['k'] == k_sturges:
                                    reject_simple = test_res['reject_h0']
                                    break
                    
                    if reject_simple is not None:
                        rejections_simple[size].append(reject_simple)
                        
                        print(f"  Выборка #{sample_idx+1}:")
                        print(f"    Простая гипотеза: {'Отвергаем' if reject_simple else 'Не отвергаем'}")
                        print(f"    Сложная гипотеза: {'Отвергаем' if reject_complex else 'Не отвергаем'}")
                        
                        if reject_simple != reject_complex:
                            print(f"    ⚠️  РАСХОЖДЕНИЕ в решениях!")
        
        # Анализ частот отвержения
        print(f"\n  Частота отвержения H0 по размерам выборок:")
        for size in sorted(dist_data.keys(), key=lambda x: int(x)):
            if rejections_simple[size] and rejections_complex[size]:
                freq_simple = sum(rejections_simple[size]) / len(rejections_simple[size])
                freq_complex = sum(rejections_complex[size]) / len(rejections_complex[size])
                print(f"    N={size}: Простая={freq_simple:.2%}, Сложная={freq_complex:.2%}")
    
    return power_analysis

def save_complex_results(results, filename='chi2_test_complex_results.json'):
    """Сохранение результатов сложной гипотезы в файл JSON"""
    serializable_results = {}
    
    for dist_name, dist_data in results.items():
        serializable_results[dist_name] = {}
        
        for size, size_data in dist_data.items():
            serializable_results[dist_name][str(size)] = []
            
            for sample_result in size_data:
                test_res = sample_result['test_result']
                serializable_test = {
                    'k': int(test_res['k']),
                    'chi2_statistic': float(test_res['chi2_statistic']),
                    'degrees_of_freedom': int(test_res['degrees_of_freedom']),
                    'estimated_params': {k: (float(v) if isinstance(v, (int, float)) else v) 
                                        for k, v in test_res['estimated_params'].items()},
                    's': int(test_res['s']),
                    'critical_value': float(test_res['critical_value']),
                    'p_value': float(test_res['p_value']),
                    'reject_h0': bool(test_res['reject_h0']),
                    'valid_test': bool(test_res['valid_test']),
                    'observed_freq': [int(x) for x in test_res['observed_freq']],
                    'expected_freq': [float(x) for x in test_res['expected_freq']],
                    'theoretical_probs': [float(x) for x in test_res['theoretical_probs']],
                    'intervals': [(float(a), float(b)) for a, b in test_res['intervals']]
                }
                
                serializable_sample = {
                    'sample_index': int(sample_result['sample_index']),
                    'sample_size': int(sample_result['sample_size']),
                    'k_sturges': int(sample_result['k_sturges']),
                    'test_result': serializable_test
                }
                serializable_results[dist_name][str(size)].append(serializable_sample)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nРезультаты сложной гипотезы сохранены в файл: {filename}")
    return serializable_results

def main_complex_hypothesis():
    """Основная функция для проверки сложной гипотезы"""
    print("ЗАГРУЗКА ДАННЫХ...")
    data = load_samples('sample_generation.json')
    
    print("\n" + "="*80)
    print("ПРОВЕРКА СЛОЖНОЙ ГИПОТЕЗЫ (НЕИЗВЕСТНЫЕ ПАРАМЕТРЫ)")
    print("="*80)
    
    # Сравнение простой и сложной гипотез
    comparison = compare_simple_complex_hypotheses(data, alpha=0.05)
    
    # Сохранение результатов
    save_complex_results(comparison)
    
    # Анализ мощности
    power_analysis = analyze_power_complex_vs_simple(data, alpha=0.05)
    
    print("\n" + "="*80)
    print("ОСНОВНЫЕ ВЫВОДЫ ПО СЛОЖНОЙ ГИПОТЕЗЕ:")
    print("="*80)
    
    print("""
    1. Для сложной гипотезы число степеней свободы уменьшается на 
       количество оцениваемых параметров: r = k - 1 - s
       
    2. Оценка параметров по выборке вносит дополнительную неопределенность,
       что может влиять на мощность критерия.
       
    3. Ожидается, что для выборок из проверяемых распределений:
       - Частота отвержения H0 должна быть близка к уровню значимости α
       - Сложная гипотеза может быть менее мощной, чем простая
       - Решения могут отличаться, особенно для малых выборок
       
    4. Особенности оценки параметров:
       - Для дискретного равномерного: оценка θ = max(X) смещена вниз
       - Для распределения Эрланга: оценка θ = m/mean состоятельна
    """)
    
    return comparison, power_analysis

if __name__ == "__main__":
    comparison, power_analysis = main_complex_hypothesis()