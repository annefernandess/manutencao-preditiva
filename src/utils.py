"""
Módulo de funções utilitárias do projeto.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def set_plotting_style():
    """Define o estilo padrão para gráficos."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
