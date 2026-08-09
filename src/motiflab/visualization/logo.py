import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.colors as mcolors


def plot_motif_logo(ppm: np.ndarray, symbols: tuple, save_path: str | Path, title: str = "Motif Logo"):
    ppm_t = ppm.T
    df = pd.DataFrame(ppm_t, columns=list(symbols))
    info_df = logomaker.transform_matrix(df, from_type='probability', to_type='information')
    fig, ax = plt.subplots(figsize=(10, 3))
    logo = logomaker.Logo(info_df, ax=ax, color_scheme='classic', font_name='Arial Rounded MT Bold')
    
    
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    ax.set_ylabel('Information (bits)')
    ax.set_xlabel('Position')
    ax.set_title(title)
    ax.set_ylim(0, 2.0) 
    
    # Сохраняем
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mutagenesis_heatmap(
    delta_p_matrix: np.ndarray, 
    wt_sequence: str, 
    symbols: tuple, 
    save_path: str | Path, 
    title: str = "In Silico Mutagenesis (Delta P)"
):
    """Plots a heatmap of mutation effects."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Создаем красивую палитру: Синий (ухудшает), Белый (нет эффекта), Красный (улучшает)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Серый цвет для NaN (Wild-Type позиции)
    cmap.set_bad(color='#aaaaaa')
    
    # Рисуем тепловую карту
    sns.heatmap(
        delta_p_matrix, 
        cmap=cmap, 
        center=0, 
        annot=False, 
        yticklabels=list(symbols),
        xticklabels=list(wt_sequence),
        cbar_kws={'label': 'Delta Probability'},
        ax=ax
    )
    
    # Оформление осей
    ax.set_title(title)
    ax.set_ylabel("Mutated To")
    ax.set_xlabel("Position (Wild-Type sequence)")
    
    # Выравниваем метки X, чтобы они были под квадратиками
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()