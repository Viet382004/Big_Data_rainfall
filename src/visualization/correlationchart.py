import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_data():
    """Tải dữ liệu"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    encoded_file = project_root / "data" / "processed" / "rainfall_encoded.csv"
    df = pd.read_csv(encoded_file)
    return df


def create_correlation_chart(df):
    """Tạo ma trận tương quan"""
    corr_cols = ['rain', 'temp', 'humidity', 'wind', 'pressure']
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")

    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        linewidths=0.6,
        linecolor='white',
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
        cbar_kws={
            "shrink": 0.8,
            "aspect": 20,
            "label": "Hệ số tương quan (r)"
        }
    )

    plt.title("Ma trận tương quan giữa các biến thời tiết", fontsize=18, weight='bold', pad=20)
    plt.xticks(fontsize=12, rotation=15)
    plt.yticks(fontsize=12, rotation=0)

    for _, spine in plt.gca().spines.items():
        spine.set_visible(True)
        spine.set_color("#cccccc")

    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    create_correlation_chart(df)


if __name__ == "__main__":
    main()