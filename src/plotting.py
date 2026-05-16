"""Утилиты для построения графиков."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style(cfg: dict) -> None:
    """Применить глобальные настройки matplotlib из cfg."""
    style = cfg["visualization"].get("style", "seaborn-v0_8-darkgrid")
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["figure.dpi"] = cfg["visualization"].get("dpi", 150)
    plt.rcParams["savefig.dpi"] = cfg["visualization"].get("dpi", 150)
    plt.rcParams["figure.figsize"] = cfg["visualization"].get("figsize", [10, 6])
    # Поддержка кириллицы
    plt.rcParams["font.family"] = "DejaVu Sans"


def save_fig(fig, path: str | Path, **kwargs) -> None:
    """Сохранить fig в файл, гарантируя что директория существует."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)
