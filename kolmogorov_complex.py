import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammainc
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ФУНКЦИИ ДЛЯ СЛОЖНОЙ ГИПОТЕЗЫ (НЕИЗВЕСТНЫЙ ПАРАМЕТР θ)
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
        result[mask] = gammainc(m, theta * x[mask])
    return result

def estimate_theta_mle(sample, m):
    """
    Оценка параметра θ методом максимального правдоподобия
    для распределения Эрланга с известным m
    θ̂ = m / среднее_выборки
    """
    return m / np.mean(sample)

def kolmogorov_statistic_with_estimated_theta(sample, m):
    """
    Вычисление статистики Колмогорова для сложной гипотезы
    (параметр θ оценивается по выборке)
    """
    n = len(sample)
    sample_sorted = np.sort(sample)
    
    # Оцениваем θ по ВСЕЙ выборке
    theta_hat = estimate_theta_mle(sample, m)
    
    # Теоретическая CDF с оцененным параметром
    F_theoretical = erlang_cdf(sample_sorted, theta_hat, m)
    
    i = np.arange(1, n + 1)
    
    Dn_plus = np.max(i/n - F_theoretical)
    Dn_minus = np.max(F_theoretical - (i-1)/n)
    Dn = max(Dn_plus, Dn_minus)
    
    return Dn, Dn_plus, Dn_minus, theta_hat

def bolshev_correction(Dn, n):
    """
    Поправка Большева для ускорения сходимости
    S_n = (6nD_n + 1) / (6√n)
    """
    return (6 * n * Dn + 1) / (6 * np.sqrt(n))

def load_samples_from_your_file():
    """
    Загрузка выборок из файла, созданного sample_generation.py
    """
    filename = 'sample_generation.json'
    
    if not os.path.exists(filename):
        print(f"ОШИБКА: Файл {filename} не найден!")
        print("Сначала запустите sample_generation.py для генерации выборок")
        return None, None
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Файл {filename} успешно загружен")
    
    # Извлекаем только распределение Эрланга
    samples = {}
    for size_str, sample_list in data['erlang'].items():
        size = int(size_str)
        samples[size] = [np.array(sample) for sample in sample_list]
    
    return samples, 3  # m = 3 из вашего файла

def bootstrap_critical_value(sample, m, alpha=0.05, n_bootstrap=1000):
    """
    Бутстреп-оценка критического значения для сложной гипотезы
    """
    n = len(sample)
    bootstrap_stats = []
    
    # Оцениваем параметр по исходной выборке
    theta_hat = estimate_theta_mle(sample, m)
    
    for _ in range(n_bootstrap):
        # Генерируем бутстреп-выборку из распределения Эрланга
        # с оцененным параметром
        bootstrap_sample = np.random.gamma(shape=m, scale=1/theta_hat, size=n)
        
        # Для каждой бутстреп-выборки оцениваем параметр заново
        # и вычисляем статистику Колмогорова
        theta_bootstrap = estimate_theta_mle(bootstrap_sample, m)
        
        # Сортируем выборку
        sample_sorted = np.sort(bootstrap_sample)
        
        # Вычисляем теоретическую CDF с оцененным параметром
        F_theoretical = erlang_cdf(sample_sorted, theta_bootstrap, m)
        
        # Вычисляем D_n
        i = np.arange(1, n + 1)
        Dn_plus = np.max(i/n - F_theoretical)
        Dn_minus = np.max(F_theoretical - (i-1)/n)
        Dn = max(Dn_plus, Dn_minus)
        
        # Статистика с поправкой Большева
        S_n = bolshev_correction(Dn, n)
        bootstrap_stats.append(S_n)
    
    # Квантиль распределения бутстреп-статистик
    critical_value = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return critical_value, bootstrap_stats

def perform_complex_hypothesis_test(sample, m, alpha=0.05, use_bootstrap=True, n_bootstrap=500):
    """
    Выполнение критерия Колмогорова для сложной гипотезы
    """
    n = len(sample)
    
    # Вычисляем статистику с оцененным параметром
    Dn, Dn_plus, Dn_minus, theta_hat = kolmogorov_statistic_with_estimated_theta(sample, m)
    
    # Вычисляем статистику с поправкой Большева
    statistic = bolshev_correction(Dn, n)
    
    # Получаем критическое значение
    if use_bootstrap and n >= 20:  # Бутстреп для достаточно больших выборок
        critical_value, bootstrap_stats = bootstrap_critical_value(sample, m, alpha, n_bootstrap)
        bootstrap_used = True
    else:
        # Аппроксимация для малых выборок или без бутстрепа
        # Для сложной гипотезы критические значения обычно меньше
        simple_critical = 1.358  # λ_0.05 для простой гипотезы
        critical_value = simple_critical * 0.85  # Примерная коррекция
        bootstrap_stats = None
        bootstrap_used = False
    
    # Решение
    reject_H0 = statistic > critical_value
    
    # P-value (приближенное)
    z = statistic
    k_max = 100
    p_value = 0
    for k in range(1, k_max + 1):
        term = (-1)**(k-1) * np.exp(-2 * k**2 * z**2)
        p_value += term
    p_value = 2 * p_value
    p_value = max(0, min(1, p_value))
    
    # Результаты
    result = {
        'n': n,
        'm': m,
        'theta_true': 1/6,
        'theta_estimated': theta_hat,
        'estimation_error_%': abs(theta_hat - 1/6) / (1/6) * 100,
        'D_n': Dn,
        'D_n_plus': Dn_plus,
        'D_n_minus': Dn_minus,
        'statistic_S_n': statistic,
        'alpha': alpha,
        'critical_value': critical_value,
        'reject_H0': reject_H0,
        'p_value': p_value,
        'bootstrap_used': bootstrap_used,
        'hypothesis_type': 'complex'
    }
    
    return result

def run_complex_hypothesis_analysis():
    """
    Основной анализ для сложной гипотезы
    """
    print("=" * 80)
    print("КРИТЕРИЙ КОЛМОГОРОВА ДЛЯ СЛОЖНОЙ ГИПОТЕЗЫ")
    print("=" * 80)
    print("Проверка гипотезы: выборка из распределения Эрланга")
    print("Сложная гипотеза: параметр θ оценивается по выборке")
    print("Параметр m = 3 (известен)")
    print("=" * 80)
    
    # 1. Загрузка ВАШИХ выборок
    print("\n1. ЗАГРУЗКА ВАШИХ ВЫБОРОК ИЗ sample_generation.json")
    samples, m = load_samples_from_your_file()
    
    if samples is None:
        return
    
    print(f"\nЗагружено выборок распределения Эрланга:")
    total_samples = 0
    sizes = []
    for size in sorted(samples.keys()):
        count = len(samples[size])
        total_samples += count
        sizes.append(size)
        print(f"  n={size}: {count} выборок")
    print(f"Всего: {total_samples} выборок")
    
    # 2. Проведение тестов
    print("\n2. ВЫПОЛНЕНИЕ КРИТЕРИЯ КОЛМОГОРОВА ДЛЯ СЛОЖНОЙ ГИПОТЕЗЫ")
    print("   Уровень значимости: α = 0.05")
    print("   Используется поправка Большева: S_n = (6nD_n + 1)/(6√n)")
    print("   Для n ≥ 20 используется бутстреп для оценки критического значения")
    print("=" * 80)
    
    all_results = {}
    summary_by_size = {}
    
    for size in sorted(samples.keys()):
        print(f"\nРАЗМЕР ВЫБОРКИ: n = {size}")
        print("-" * 60)
        
        results_for_size = []
        rejections = 0
        
        for i, sample in enumerate(samples[size]):
            # Тест для сложной гипотезы
            use_bootstrap = size >= 20  # Используем бутстреп для n ≥ 20
            result = perform_complex_hypothesis_test(
                sample, m, alpha=0.05, 
                use_bootstrap=use_bootstrap, n_bootstrap=200
            )
            
            results_for_size.append(result)
            if result['reject_H0']:
                rejections += 1
            
            print(f"  Выборка #{i+1}:")
            print(f"    θ оцененное = {result['theta_estimated']:.6f}")
            print(f"    θ истинное  = {result['theta_true']:.6f}")
            print(f"    Ошибка оценки = {result['estimation_error_%']:.1f}%")
            print(f"    D_n = {result['D_n']:.4f}")
            print(f"    S_n = {result['statistic_S_n']:.4f}")
            print(f"    Крит. значение = {result['critical_value']:.4f}")
            if result['bootstrap_used']:
                print(f"    (использован бутстреп)")
            print(f"    P-value = {result['p_value']:.4f}")
            
            decision = "✓ ОТВЕРГАЕМ" if result['reject_H0'] else "✓ НЕ ОТВЕРГАЕМ"
            print(f"    Решение: {decision} H₀")
            print()
        
        all_results[size] = results_for_size
        
        # Сводка по размеру
        rejection_rate = rejections / len(samples[size]) * 100
        avg_theta = np.mean([r['theta_estimated'] for r in results_for_size])
        avg_error = np.mean([r['estimation_error_%'] for r in results_for_size])
        avg_Dn = np.mean([r['D_n'] for r in results_for_size])
        
        summary_by_size[size] = {
            'rejections': rejections,
            'total': len(samples[size]),
            'rejection_rate': rejection_rate,
            'avg_theta': avg_theta,
            'avg_error': avg_error,
            'avg_Dn': avg_Dn
        }
        
        print(f"  Сводка для n={size}:")
        print(f"    Отвержений: {rejections}/{len(samples[size])} ({rejection_rate:.1f}%)")
        print(f"    Среднее θ оцененное: {avg_theta:.6f}")
        print(f"    Средняя ошибка оценки: {avg_error:.1f}%")
        print(f"    Среднее D_n: {avg_Dn:.4f}")
    
    # 3. Анализ результатов
    print("\n" + "=" * 80)
    print("3. АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    total_rejections = 0
    total_tests = 0
    
    print("\nРезультаты по объемам выборок:")
    print("-" * 60)
    print(f"{'n':>6} {'Отвержений':>12} {'Частота':>10} {'Ср.θ':>12} {'Ошибкаθ%':>10} {'Ср.D_n':>8}")
    print("-" * 60)
    
    for size in sorted(summary_by_size.keys()):
        s = summary_by_size[size]
        total_rejections += s['rejections']
        total_tests += s['total']
        
        print(f"{size:6d} {s['rejections']:3d}/{s['total']:3d} "
              f"{s['rejection_rate']:9.1f}% {s['avg_theta']:11.6f} "
              f"{s['avg_error']:9.1f}% {s['avg_Dn']:8.4f}")
    
    overall_rate = total_rejections / total_tests * 100
    print("-" * 60)
    print(f"Итого: {total_rejections:3d}/{total_tests:3d} отвержений "
          f"({overall_rate:.1f}%)")
    
    # 4. Визуализация
    print("\n" + "=" * 80)
    print("4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    visualize_results(all_results, samples, m)
    
    # 5. Сохранение результатов
    save_results(all_results)
    
    return all_results, summary_by_size

def visualize_results(all_results, samples, m):
    """
    Визуализация результатов сложной гипотезы
    """
    sizes = sorted(all_results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Критерий Колмогорова: Сложная гипотеза (θ оценивается)', 
                 fontsize=16, y=1.02)
    
    # 1. Частота отвержений по размерам выборок
    ax = axes[0, 0]
    rejection_rates = []
    for size in sizes:
        results = all_results[size]
        reject_count = sum(1 for r in results if r['reject_H0'])
        rejection_rates.append(reject_count / len(results))
    
    ax.plot(sizes, rejection_rates, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.05, color='r', linestyle='--', label='Ожидаемое (α=0.05)')
    ax.set_xlabel('Объем выборки n')
    ax.set_ylabel('Частота отвержения H₀')
    ax.set_title('Частота ошибок I рода')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Точность оценки параметра θ
    ax = axes[0, 1]
    avg_errors = []
    for size in sizes:
        results = all_results[size]
        avg_error = np.mean([r['estimation_error_%'] for r in results])
        avg_errors.append(avg_error)
    
    ax.plot(sizes, avg_errors, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Объем выборки n')
    ax.set_ylabel('Средняя ошибка оценки θ (%)')
    ax.set_title('Точность оценки параметра')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 3. Зависимость D_n от объема выборки
    ax = axes[0, 2]
    avg_Dn = []
    for size in sizes:
        results = all_results[size]
        avg_Dn.append(np.mean([r['D_n'] for r in results]))
    
    ax.plot(sizes, avg_Dn, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Объем выборки n')
    ax.set_ylabel('Среднее D_n')
    ax.set_title('Зависимость D_n от объема выборки')
    ax.grid(True, alpha=0.3)
    
    # 4. Пример эмпирической и теоретической CDF (n=100)
    ax = axes[1, 0]
    if 100 in samples:
        sample = samples[100][0]
        sample_sorted = np.sort(sample)
        theta_hat = estimate_theta_mle(sample, m)
        x = np.linspace(0, max(sample)*1.1, 1000)
        
        ax.step(sample_sorted, np.arange(1, len(sample)+1)/len(sample), 
                'b-', linewidth=2, label='Эмпирическая $\hat{F}_n(x)$')
        ax.plot(x, erlang_cdf(x, theta_hat, m), 'r-', linewidth=2, 
                label=f'Теоретическая\n(θ оценен={theta_hat:.4f})')
        ax.set_xlabel('x')
        ax.set_ylabel('F(x)')
        ax.set_title('CDF с оцененным параметром (n=100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Гистограмма оценок θ
    ax = axes[1, 1]
    all_theta_estimates = []
    for size in sizes:
        results = all_results[size]
        all_theta_estimates.extend([r['theta_estimated'] for r in results])
    
    ax.hist(all_theta_estimates, bins=20, color='skyblue', 
            edgecolor='black', alpha=0.7)
    ax.axvline(x=1/6, color='r', linestyle='--', 
               linewidth=2, label='Истинное θ=1/6')
    ax.set_xlabel('Оценка параметра θ')
    ax.set_ylabel('Частота')
    ax.set_title('Распределение оценок параметра')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Сравнение с простой гипотезой
    ax = axes[1, 2]
    if 100 in samples:
        sample = samples[100][0]
        n = len(sample)
        
        # Простая гипотеза
        theta_true = 1/6
        sample_sorted = np.sort(sample)
        F_true = erlang_cdf(sample_sorted, theta_true, m)
        
        # Сложная гипотеза
        theta_hat = estimate_theta_mle(sample, m)
        F_hat = erlang_cdf(sample_sorted, theta_hat, m)
        
        i = np.arange(1, n + 1)
        
        ax.plot(i/n, F_true, 'b-', label=f'Простая (θ={theta_true:.4f})', alpha=0.7)
        ax.plot(i/n, F_hat, 'r--', label=f'Сложная (θ оценен={theta_hat:.4f})', alpha=0.7)
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
        ax.set_xlabel('i/n')
        ax.set_ylabel('F(X(i))')
        ax.set_title('Сравнение: простая vs сложная гипотеза')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_results(all_results):
    """
    Сохранение результатов в JSON файл
    """
    filename = 'kolmogorov_complex_results.json'
    
    serializable_results = {}
    for size, result_list in all_results.items():
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
    
    print(f"\n✓ Результаты сохранены в файл: {filename}")

def generate_report(all_results, summary_by_size):
    """
    Генерация итогового отчета
    """
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ: КРИТЕРИЙ КОЛМОГОРОВА ДЛЯ СЛОЖНОЙ ГИПОТЕЗЫ")
    print("=" * 80)
    
    total_rejections = 0
    total_tests = 0
    
    for size in summary_by_size:
        s = summary_by_size[size]
        total_rejections += s['rejections']
        total_tests += s['total']
    
    overall_rate = total_rejections / total_tests * 100
    
    print(f"\nОБЩАЯ СТАТИСТИКА:")
    print(f"  Всего тестов: {total_tests}")
    print(f"  Отвержений H₀: {total_rejections}")
    print(f"  Частота отвержений: {overall_rate:.1f}%")
    print(f"  Ожидаемая частота (α=0.05): 5.0%")
    
    print(f"\nВЫВОДЫ:")
    print("""
    1. Критерий Колмогорова для сложной гипотезы:
       - Параметр θ оценивается по выборке: θ̂ = m/mean
       - Критические значения отличаются от простой гипотезы
       - Рекомендуется использовать бутстреп для их оценки
    
    2. Особенности для распределения Эрланга:
       - Оценка θ̂ = m/mean является состоятельной
       - С ростом объема выборки точность оценки улучшается
       - Для малых выборок ошибка оценки может быть значительной
    
    3. Рекомендации:
       - Для n < 20 результаты могут быть ненадежными
       - Для n ≥ 20 рекомендуется использовать бутстреп
       - Учитывать дополнительную неопределенность из-за
         оценки параметров
    """)
    
    print("\n" + "=" * 80)
    print("КОД УСПЕШНО ВЫПОЛНЕН!")
    print("=" * 80)

# ============================================================================
# ЗАПУСК АНАЛИЗА
# ============================================================================

if __name__ == "__main__":
    # Запуск анализа сложной гипотезы
    all_results, summary_by_size = run_complex_hypothesis_analysis()
    
    # Генерация отчета
    if all_results:
        generate_report(all_results, summary_by_size)