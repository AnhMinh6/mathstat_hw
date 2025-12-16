import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammainc
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ОСНОВНЫЕ ФУНКЦИИ КРИТЕРИЯ КОЛМОГОРОВА ДЛЯ РАСПРЕДЕЛЕНИЯ ЭРЛАНГА
# ============================================================================

def erlang_cdf(x, theta, m):
    """
    Функция распределения Эрланга
    F(x) = 1 - exp(-θx) * Σ_{k=0}^{m-1} (θx)^k / k!
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask = x > 0
    if np.any(mask):
        # Используем неполную гамма-функцию для точного вычисления
        result[mask] = gammainc(m, theta * x[mask])
    return result

def kolmogorov_statistic(sample, cdf_func, *args):
    """
    Вычисление статистики Колмогорова D_n
    D_n = sup_x |F_n(x) - F_0(x)|
    """
    n = len(sample)
    sample_sorted = np.sort(sample)
    
    # Теоретическая функция распределения
    F_theoretical = cdf_func(sample_sorted, *args)
    i = np.arange(1, n + 1)
    
    # Вычисляем D_n^+ и D_n^-
    Dn_plus = np.max(i/n - F_theoretical)
    Dn_minus = np.max(F_theoretical - (i-1)/n)
    Dn = max(Dn_plus, Dn_minus)
    
    return Dn, Dn_plus, Dn_minus

def bolshev_correction(Dn, n):
    """
    Поправка Большева для ускорения сходимости
    S_n = (6nD_n + 1) / (6√n)
    """
    return (6 * n * Dn + 1) / (6 * np.sqrt(n))

def kolmogorov_critical_value(alpha, n, use_bolshev=False):
    """
    Критические значения для критерия Колмогорова
    
    Аргументы:
    ----------
    alpha : float
        Уровень значимости
    n : int
        Объем выборки
    use_bolshev : bool
        Используется ли статистика с поправкой Большева
    """
    # Точные табличные значения для малых выборок (n ≤ 10)
    exact_critical_values = {
        0.20: [0.900, 0.684, 0.565, 0.494, 0.446, 0.410, 0.381, 0.358, 0.339, 0.323],
        0.10: [0.950, 0.776, 0.642, 0.564, 0.510, 0.470, 0.438, 0.411, 0.388, 0.368],
        0.05: [0.975, 0.842, 0.708, 0.624, 0.563, 0.521, 0.486, 0.457, 0.432, 0.409],
        0.01: [0.995, 0.929, 0.828, 0.733, 0.669, 0.618, 0.577, 0.543, 0.514, 0.490]
    }
    
    # Для n ≤ 10 используем точные табличные значения
    if n <= 10 and alpha in exact_critical_values:
        idx = min(n - 1, 9)
        critical_Dn = exact_critical_values[alpha][idx]
        
        if use_bolshev:
            # Для S_n преобразуем обратно к √n * D_n для сравнения
            return critical_Dn * np.sqrt(n)
        else:
            return critical_Dn
    
    # Асимптотические квантили распределения Колмогорова
    asymptotic_quantiles = {
        0.20: 1.073,
        0.10: 1.224,
        0.05: 1.358,
        0.02: 1.517,
        0.01: 1.628,
        0.005: 1.731,
        0.002: 1.859,
        0.001: 1.949
    }
    
    if alpha in asymptotic_quantiles:
        lambda_alpha = asymptotic_quantiles[alpha]
        if use_bolshev:
            # Для S_n критическое значение такое же как для √n * D_n
            return lambda_alpha
        else:
            # Для √n * D_n
            return lambda_alpha / np.sqrt(n)
    
    # Для произвольного alpha (аппроксимация)
    c_alpha = np.sqrt(-0.5 * np.log(alpha/2))
    if use_bolshev:
        return c_alpha
    else:
        return c_alpha / np.sqrt(n)

def perform_kolmogorov_test(sample, cdf_func, alpha=0.05, use_bolshev=True, *args):
    """
    Выполнение критерия Колмогорова для распределения Эрланга
    """
    n = len(sample)
    
    # Вычисляем D_n, D_n^+, D_n^-
    Dn, Dn_plus, Dn_minus = kolmogorov_statistic(sample, cdf_func, *args)
    
    # Вычисляем статистику
    if use_bolshev:
        # Статистика с поправкой Большева
        statistic = bolshev_correction(Dn, n)
        stat_name = "S_n (с поправкой Большева)"
        stat_symbol = "S_n"
        # Критическое значение для S_n
        critical_value = kolmogorov_critical_value(alpha, n, use_bolshev=True)
    else:
        # Обычная статистика √n * D_n
        statistic = np.sqrt(n) * Dn
        stat_name = "√n D_n"
        stat_symbol = "√n D_n"
        critical_value = kolmogorov_critical_value(alpha, n, use_bolshev=False)
    
    # Решение о гипотезе
    reject_H0 = statistic > critical_value
    
    # Вычисление p-value через распределение Колмогорова
    z = statistic  # Для S_n используем ту же формулу
    k_max = 100
    p_value = 0
    for k in range(1, k_max + 1):
        term = (-1)**(k-1) * np.exp(-2 * k**2 * z**2)
        p_value += term
    p_value = 2 * p_value
    p_value = max(0, min(1, p_value))
    
    # Результаты теста
    result = {
        'n': n,
        'D_n': Dn,
        'D_n_plus': Dn_plus,
        'D_n_minus': Dn_minus,
        'statistic': statistic,
        'stat_name': stat_name,
        'stat_symbol': stat_symbol,
        'alpha': alpha,
        'critical_value': critical_value,
        'reject_H0': reject_H0,
        'p_value': p_value,
        'use_bolshev': use_bolshev
    }
    
    return result

def compare_statistics_bolshev(sample, cdf_func, *args):
    """
    Сравнение обычной статистики √nD_n и статистики S_n с поправкой Большева
    """
    n = len(sample)
    
    # Вычисляем D_n
    Dn, Dn_plus, Dn_minus = kolmogorov_statistic(sample, cdf_func, *args)
    
    # Обычная статистика
    sqrt_n_Dn = np.sqrt(n) * Dn
    
    # Статистика с поправкой Большева
    S_n = bolshev_correction(Dn, n)
    
    return {
        'n': n,
        'D_n': Dn,
        'sqrt_n_D_n': sqrt_n_Dn,
        'S_n': S_n,
        'разница': S_n - sqrt_n_Dn,
        'относительная_разница': (S_n - sqrt_n_Dn) / sqrt_n_Dn if sqrt_n_Dn != 0 else 0
    }

# ============================================================================
# РАБОТА С ЗАГРУЖЕННЫМИ ВЫБОРКАМИ
# ============================================================================

def load_erlang_samples(filename='sample_generation.json'):
    """
    Загрузка выборок распределения Эрланга из JSON файла
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Параметры из файла sample_generations.py
    m_erlang = 3
    theta_erlang = 1/6
    
    # Преобразуем данные в удобный формат
    samples = {}
    
    for size_str, sample_list in data['erlang'].items():
        size = int(size_str)
        samples[size] = [np.array(sample) for sample in sample_list]
    
    return samples, m_erlang, theta_erlang

def run_kolmogorov_tests_for_erlang(samples, m, theta, alpha=0.05, use_bolshev=True):
    """
    Запуск критерия Колмогорова для всех выборок распределения Эрланга
    """
    results = {}
    
    print("=" * 80)
    print("КРИТЕРИЙ КОЛМОГОРОВА ДЛЯ РАСПРЕДЕЛЕНИЯ ЭРЛАНГА")
    print("=" * 80)
    print(f"Параметры распределения:")
    print(f"  m (параметр формы) = {m}")
    print(f"  θ (параметр скорости) = {theta:.6f}")
    print(f"  Среднее = m/θ = {m/theta:.2f}")
    print(f"  Дисперсия = m/θ² = {m/(theta**2):.2f}")
    print(f"Уровень значимости α = {alpha}")
    print(f"Используется поправка Большева: {use_bolshev}")
    print("=" * 80)
    
    for size, sample_list in samples.items():
        results[size] = []
        print(f"\nРазмер выборки: n = {size}")
        print("-" * 50)
        
        for i, sample in enumerate(sample_list):
            result = perform_kolmogorov_test(
                sample, 
                erlang_cdf, 
                alpha, 
                use_bolshev, 
                theta, 
                m
            )
            results[size].append(result)
            
            print(f"  Выборка #{i+1}:")
            print(f"    D_n = {result['D_n']:.6f}")
            print(f"    D_n⁺ = {result['D_n_plus']:.6f}")
            print(f"    D_n⁻ = {result['D_n_minus']:.6f}")
            print(f"    {result['stat_symbol']} = {result['statistic']:.6f}")
            print(f"    Критическое значение = {result['critical_value']:.6f}")
            print(f"    P-value = {result['p_value']:.6f}")
            
            decision = "ОТВЕРГАЕМ" if result['reject_H0'] else "НЕ ОТВЕРГАЕМ"
            print(f"    Решение: {decision} H₀ (выборка из распределения Эрланга)")
    
    return results

def analyze_erlang_results(results):
    """
    Анализ результатов тестов для распределения Эрланга
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ДЛЯ РАСПРЕДЕЛЕНИЯ ЭРЛАНГА")
    print("=" * 80)
    
    sizes = sorted(results.keys())
    
    print("\nЧастота отвержения гипотезы H₀:")
    print("(Ожидаемая частота ошибок I рода при α=0.05: 5%)")
    print("-" * 60)
    
    for size in sizes:
        result_list = results[size]
        reject_count = sum(1 for r in result_list if r['reject_H0'])
        total = len(result_list)
        rejection_rate = reject_count / total * 100
        
        # Статистика по D_n
        avg_Dn = np.mean([r['D_n'] for r in result_list])
        std_Dn = np.std([r['D_n'] for r in result_list])
        
        # Статистика по тестовой статистике
        avg_statistic = np.mean([r['statistic'] for r in result_list])
        avg_pvalue = np.mean([r['p_value'] for r in result_list])
        
        print(f"n = {size}:")
        print(f"  Отвержений: {reject_count}/{total} ({rejection_rate:.1f}%)")
        print(f"  Среднее D_n: {avg_Dn:.6f} ± {std_Dn:.6f}")
        print(f"  Средняя статистика: {avg_statistic:.6f}")
        print(f"  Средний p-value: {avg_pvalue:.6f}")

def visualize_erlang_results(results, samples, m, theta):
    """
    Визуализация результатов для распределения Эрланга
    """
    sizes = sorted(results.keys())
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Критерий Колмогорова для распределения Эрланга', fontsize=16, y=1.02)
    
    # 1. График частоты отвержения H₀
    ax = axes[0, 0]
    rejection_rates = []
    
    for size in sizes:
        result_list = results[size]
        reject_count = sum(1 for r in result_list if r['reject_H0'])
        rejection_rates.append(reject_count / len(result_list))
    
    ax.plot(sizes, rejection_rates, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.05, color='r', linestyle='--', label='Ожидаемое (α=0.05)')
    ax.set_xlabel('Объем выборки n')
    ax.set_ylabel('Частота отвержения H₀')
    ax.set_title('Частота ошибок I рода')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. График среднего D_n
    ax = axes[0, 1]
    avg_Dn_values = []
    
    for size in sizes:
        result_list = results[size]
        avg_Dn = np.mean([r['D_n'] for r in result_list])
        avg_Dn_values.append(avg_Dn)
    
    ax.plot(sizes, avg_Dn_values, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Объем выборки n')
    ax.set_ylabel('Среднее D_n')
    ax.set_title('Зависимость D_n от объема выборки')
    ax.grid(True, alpha=0.3)
    
    # 3. График средней статистики
    ax = axes[0, 2]
    avg_stat_values = []
    
    for size in sizes:
        result_list = results[size]
        avg_stat = np.mean([r['statistic'] for r in result_list])
        avg_stat_values.append(avg_stat)
    
    ax.plot(sizes, avg_stat_values, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Объем выборки n')
    ax.set_ylabel('Средняя статистика')
    ax.set_title('Зависимость статистики от объема выборки')
    ax.grid(True, alpha=0.3)
    
    # 4. Пример эмпирической vs теоретической CDF (n=100, первая выборка)
    ax = axes[1, 0]
    sample_size = 100
    if sample_size in samples:
        sample = samples[sample_size][0]
        sample_sorted = np.sort(sample)
        x = np.linspace(0, max(sample)*1.1, 1000)
        
        ax.step(sample_sorted, np.arange(1, len(sample)+1)/len(sample), 
                'b-', linewidth=2, label='Эмпирическая $\hat{F}_n(x)$')
        ax.plot(x, erlang_cdf(x, theta, m), 'r-', linewidth=2, 
                label=f'Теоретическая $F_0(x)$\n(m={m}, θ={theta:.3f})')
        ax.set_xlabel('x')
        ax.set_ylabel('F(x)')
        ax.set_title(f'Функции распределения (n={sample_size})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Разности D_n⁺ и D_n⁻
    ax = axes[1, 1]
    if sample_size in samples:
        sample = samples[sample_size][0]
        sample_sorted = np.sort(sample)
        F_theor = erlang_cdf(sample_sorted, theta, m)
        i = np.arange(1, len(sample)+1)
        
        Dn_plus = i/len(sample) - F_theor
        Dn_minus = F_theor - (i-1)/len(sample)
        
        ax.plot(sample_sorted, Dn_plus, 'g^', markersize=6, 
                label=f'$i/n - F_0(X_i)$ (max={np.max(Dn_plus):.4f})', alpha=0.7)
        ax.plot(sample_sorted, Dn_minus, 'mv', markersize=6, 
                label=f'$F_0(X_i) - (i-1)/n$ (max={np.max(Dn_minus):.4f})', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('Разность')
        ax.set_title(f'Разности для вычисления D_n (n={sample_size})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Гистограмма p-value
    ax = axes[1, 2]
    all_pvalues = []
    for size in sizes:
        result_list = results[size]
        all_pvalues.extend([r['p_value'] for r in result_list])
    
    ax.hist(all_pvalues, bins=20, color='skyblue', edgecolor='black', 
            alpha=0.7, density=True)
    ax.axvline(x=0.05, color='r', linestyle='--', linewidth=2, label='α=0.05')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Плотность')
    ax.set_title('Распределение p-value для всех выборок')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Сравнение √nD_n и S_n для разных объемов выборки
    ax = axes[2, 0]
    sqrt_n_Dn_avg = []
    S_n_avg = []
    
    for size in sizes:
        result_list = results[size]
        # Берем первую выборку для демонстрации
        if result_list:
            comp = compare_statistics_bolshev(
                samples[size][0], 
                erlang_cdf, 
                theta, 
                m
            )
            sqrt_n_Dn_avg.append(comp['sqrt_n_D_n'])
            S_n_avg.append(comp['S_n'])
    
    if sqrt_n_Dn_avg and S_n_avg:
        ax.plot(sizes[:len(sqrt_n_Dn_avg)], sqrt_n_Dn_avg, 'b-', 
                label='$√n D_n$', linewidth=2)
        ax.plot(sizes[:len(S_n_avg)], S_n_avg, 'r--', 
                label='$S_n$ (Большева)', linewidth=2)
        ax.axhline(y=1.358, color='g', linestyle=':', 
                  label='λ_α (α=0.05)', linewidth=2)
        ax.set_xlabel('Объем выборки n')
        ax.set_ylabel('Значение статистики')
        ax.set_title('Сравнение статистик для разных n')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 8. Разность S_n - √nD_n
    ax = axes[2, 1]
    if sqrt_n_Dn_avg and S_n_avg:
        diff = np.array(S_n_avg) - np.array(sqrt_n_Dn_avg)
        ax.plot(sizes[:len(diff)], diff, 'k-', linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Объем выборки n')
        ax.set_ylabel('$S_n - √n D_n$')
        ax.set_title('Эффект поправки Большева')
        ax.grid(True, alpha=0.3)
    
    # 9. Теоретическое распределение Эрланга
    ax = axes[2, 2]
    if sample_size in samples:
        sample = samples[sample_size][0]
        x = np.linspace(0, max(sample)*1.1, 1000)
        pdf = (theta**m / np.math.factorial(m-1)) * x**(m-1) * np.exp(-theta * x)
        
        ax.plot(x, pdf, 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Плотность распределения Эрланга\n(m={m}, θ={theta:.3f})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_erlang_results(results, filename='kolmogorov_erlang_results.json'):
    """
    Сохранение результатов тестов в JSON файл
    """
    serializable_results = {}
    
    for size, result_list in results.items():
        serializable_results[str(size)] = []
        for result in result_list:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.floating, np.integer)):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.bool_):
                    serializable_result[key] = bool(value)
                else:
                    serializable_result[key] = value
            serializable_results[str(size)].append(serializable_result)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в файл: {filename}")
    return serializable_results

def generate_report(results, m, theta):
    """
    Генерация итогового отчета
    """
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)
    
    sizes = sorted(results.keys())
    
    print("\n1. ОБЩАЯ СТАТИСТИКА:")
    print("-" * 60)
    
    total_tests = 0
    total_rejections = 0
    
    for size in sizes:
        result_list = results[size]
        reject_count = sum(1 for r in result_list if r['reject_H0'])
        total_tests += len(result_list)
        total_rejections += reject_count
    
    overall_rejection_rate = total_rejections / total_tests * 100
    print(f"Всего тестов: {total_tests}")
    print(f"Всего отвержений H₀: {total_rejections}")
    print(f"Общая частота отвержений: {overall_rejection_rate:.1f}%")
    print(f"Ожидаемая частота (α=0.05): 5.0%")
    
    print("\n2. ВЛИЯНИЕ ОБЪЕМА ВЫБОРКИ:")
    print("-" * 60)
    
    for size in sizes:
        result_list = results[size]
        reject_count = sum(1 for r in result_list if r['reject_H0'])
        rejection_rate = reject_count / len(result_list) * 100
        avg_Dn = np.mean([r['D_n'] for r in result_list])
        
        print(f"n = {size:4d}: {reject_count:2d}/{len(result_list):2d} отвержений "
              f"({rejection_rate:5.1f}%), среднее D_n = {avg_Dn:.4f}")
    
    print("\n3. ВЫВОДЫ:")
    print("-" * 60)
    print("""
    Критерий Колмогорова с поправкой Большева был применен 
    для проверки гипотезы о распределении Эрланга.
    
    Ключевые наблюдения:
    1. Для выборок малого объема (n ≤ 10) критерий может быть
       ненадежным из-за использования асимптотического приближения.
    
    2. Поправка Большева (S_n = (6nD_n + 1)/(6√n)) обеспечивает
       более быструю сходимость к предельному распределению.
    
    3. Частота отвержения гипотезы должна быть близка к уровню
       значимости α=0.05 при верной H₀.
    
    4. Критерий Колмогорова чувствителен к отклонениям в хвостах
       распределения, что важно для распределения Эрланга.
    """)

# ============================================================================
# ОСНОВНАЯ ЧАСТЬ ПРОГРАММЫ
# ============================================================================

if __name__ == "__main__":
    print("КРИТЕРИЙ КОЛМОГОРОВА С ПОПРАВКОЙ БОЛЬШЕВА")
    print("=" * 80)
    print("ТОЛЬКО ДЛЯ РАСПРЕДЕЛЕНИЯ ЭРЛАНГА (непрерывное распределение)")
    print("=" * 80)
    
    # 1. Загрузка выборок распределения Эрланга
    print("\n1. ЗАГРУЗКА ВЫБОРОК РАСПРЕДЕЛЕНИЯ ЭРЛАНГА...")
    try:
        samples, m, theta = load_erlang_samples('sample_generation.json')
        print(f"Успешно загружено {sum(len(v) for v in samples.values())} выборок")
        
        print("\nОбъемы выборок:")
        for size in sorted(samples.keys()):
            print(f"  n={size}: {len(samples[size])} выборок")
            
        # Пример первой выборки
        sample_size = list(samples.keys())[0]
        first_sample = samples[sample_size][0]
        print(f"\nПример выборки (n={sample_size}, первые 5 значений):")
        print(f"  {first_sample[:5]}")
        print(f"  Минимум: {np.min(first_sample):.2f}")
        print(f"  Максимум: {np.max(first_sample):.2f}")
        print(f"  Среднее: {np.mean(first_sample):.2f} (теоретическое: {m/theta:.2f})")
        
    except FileNotFoundError:
        print("Ошибка: Файл sample_generation.json не найден!")
        print("Сначала запустите sample_generations.py для генерации выборок.")
        exit(1)
    
    # 2. Выполнение критерия Колмогорова
    print("\n2. ВЫПОЛНЕНИЕ КРИТЕРИЯ КОЛМОГОРОВА...")
    print("   Используется поправка Большева для ускорения сходимости")
    
    results = run_kolmogorov_tests_for_erlang(
        samples=samples,
        m=m,
        theta=theta,
        alpha=0.05,
        use_bolshev=True
    )
    
    # 3. Анализ результатов
    analyze_erlang_results(results)
    
    # 4. Визуализация
    print("\n4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")
    visualize_erlang_results(results, samples, m, theta)
    
    # 5. Сохранение результатов
    save_erlang_results(results)
    
    # 6. Итоговый отчет
    generate_report(results, m, theta)
    
    print("\n" + "=" * 80)
    print("ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 80)