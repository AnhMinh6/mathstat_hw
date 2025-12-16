import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy import stats

def analyze_scaled_homogeneity():
    """
    Анализ однородности для МАСШТАБИРОВАННЫХ значений Dm,n
    (значения уже домножены на √((m*n)/(m+n)))
    """
    # Загрузка результатов
    with open('d_statistics_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    distributions = ['discrete_uniform', 'erlang']
    sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
    
    print("="*80)
    print("АНАЛИЗ ОДНОРОДНОСТИ (с учетом масштабирования Dm,n)")
    print("="*80)
    
    for distribution in distributions:
        print(f"\n{'='*40}")
        dist_name = 'Дискретное равномерное' if distribution == 'discrete_uniform' else 'Эрланга'
        print(f"Распределение: {dist_name}")
        print(f"{'='*40}")
        
        # Анализируем статистики
        print("\nСтатистики по выборкам одинакового размера:")
        print("Размер | Среднее D | Критическое D (α=0.05) | Однородность")
        print("-"*70)
        
        for size in sizes:
            key = f'D_{size}_{size}'
            if key in data[distribution]:
                d_mean = data[distribution][key]['mean']
                
                # Критическое значение для НЕмасштабированной статистики
                # D_crit = C(α) * √((m+n)/(m*n)), где C(α) ≈ 1.36 для α=0.05
                # Но у нас D уже масштабирована: D_scaled = D_original * √((m*n)/(m+n))
                # Поэтому: D_original = D_scaled / √((m*n)/(m+n))
                
                # Для одинаковых размеров: m = n
                scaling_factor = np.sqrt((size * size) / (size + size))  # √((n*n)/(2n)) = √(n/2)
                d_original = d_mean / scaling_factor
                
                # Критическое значение для исходной статистики Колмогорова-Смирнова
                d_crit = 1.36 / np.sqrt(size/2)  # Приближенная формула
                
                # Оценка однородности
                if d_original < d_crit:
                    homogeneity = "ОДНОРОДНЫ"
                else:
                    homogeneity = "НЕ однородны"
                
                print(f"{size:6d} | {d_mean:9.6f} | {d_crit:20.6f} | {homogeneity}")
    
    # Визуализация
    visualize_scaled_results(data)

def visualize_scaled_results(data):
    """Визуализация результатов с учетом масштабирования"""
    distributions = ['discrete_uniform', 'erlang']
    sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, distribution in enumerate(distributions):
        dist_name = 'Дискретное равномерное' if distribution == 'discrete_uniform' else 'Эрланга'
        
        # Подготовка данных
        original_d_values = []
        scaled_d_values = []
        critical_values = []
        size_labels = []
        
        for size in sizes:
            key = f'D_{size}_{size}'
            if key in data[distribution]:
                # Масштабированное значение (из файла)
                d_scaled = data[distribution][key]['mean']
                scaled_d_values.append(d_scaled)
                
                # Пересчет в не масштабированное
                scaling_factor = np.sqrt((size * size) / (size + size))
                d_original = d_scaled / scaling_factor
                original_d_values.append(d_original)
                
                # Критическое значение
                d_crit = 1.36 / np.sqrt(size/2)
                critical_values.append(d_crit)
                
                size_labels.append(str(size))
        
        # График 1: Масштабированные значения
        ax = axes[idx, 0]
        x_pos = np.arange(len(size_labels))
        width = 0.35
        
        ax.bar(x_pos - width/2, scaled_d_values, width, label='D (масштабированная)', alpha=0.7)
        ax.bar(x_pos + width/2, original_d_values, width, label='D (исходная)', alpha=0.7)
        ax.set_title(f'Статистики D - {dist_name}')
        ax.set_xlabel('Размер выборки')
        ax.set_ylabel('Значение D')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(size_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # График 2: Сравнение с критическим значением
        ax = axes[idx, 1]
        x_pos = np.arange(len(size_labels))
        
        ax.plot(x_pos, original_d_values, 'bo-', label='D (исходная)', linewidth=2, markersize=6)
        ax.plot(x_pos, critical_values, 'r--', label='Критическое D (α=0.05)', linewidth=2)
        
        # Закрашиваем область однородности
        ax.fill_between(x_pos, 0, critical_values, alpha=0.2, color='green', label='Область однородности')
        
        ax.set_title(f'Проверка однородности - {dist_name}')
        ax.set_xlabel('Размер выборки')
        ax.set_ylabel('Значение D')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(size_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем аннотации
        for i, (d_orig, d_crit) in enumerate(zip(original_d_values, critical_values)):
            if d_orig < d_crit:
                ax.text(i, d_orig, '✓', ha='center', va='bottom', fontsize=12, color='green')
            else:
                ax.text(i, d_orig, '✗', ha='center', va='bottom', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.show()

def generate_final_conclusion():
    """Генерация итогового заключения"""
    print("\n" + "="*80)
    print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ ОБ ОДНОРОДНОСТИ")
    print("="*80)
    
    try:
        with open('d_statistics_results.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        distributions = ['discrete_uniform', 'erlang']
        sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
        
        print("\nРЕЗУЛЬТАТЫ ПРОВЕРКИ ОДНОРОДНОСТИ:")
        print("(на уровне значимости α=0.05)")
        print("-"*60)
        
        for distribution in distributions:
            dist_name = 'Дискретное равномерное' if distribution == 'discrete_uniform' else 'Эрланга'
            print(f"\n{dist_name} распределение:")
            
            homogeneous_counts = 0
            total_counts = 0
            
            for size in sizes:
                key = f'D_{size}_{size}'
                if key in data[distribution]:
                    total_counts += 1
                    
                    d_scaled = data[distribution][key]['mean']
                    scaling_factor = np.sqrt((size * size) / (size + size))
                    d_original = d_scaled / scaling_factor
                    d_crit = 1.36 / np.sqrt(size/2)
                    
                    if d_original < d_crit:
                        homogeneous_counts += 1
                        symbol = "✓"
                        status = "однородны"
                    else:
                        symbol = "✗"
                        status = "не однородны"
                    
                    print(f"  {symbol} Размер {size}: D={d_original:.4f}, D_crit={d_crit:.4f} -> {status}")
            
            percentage = (homogeneous_counts / total_counts) * 100
            print(f"  Всего однородных выборок: {homogeneous_counts}/{total_counts} ({percentage:.1f}%)")
        
        print("\n" + "="*80)
        
        # Сохраняем отчет
        save_report_to_file(data)
        
    except Exception as e:
        print(f"Ошибка при анализе: {e}")


if __name__ == "__main__":
    # Импортируем datetime для правильной работы
    import datetime
    
    print("Анализ однородности выборок с учетом масштабирования Dm,n")
    print("="*60)
    
    # Проводим анализ
    analyze_scaled_homogeneity()
    
    # Генерируем итоговое заключение
    generate_final_conclusion()