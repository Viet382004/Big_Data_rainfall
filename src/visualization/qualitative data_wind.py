import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data():
    """Tải dữ liệu"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    encoded_file = project_root / "data" / "processed" / "rainfall_encoded.csv"
    df = pd.read_csv(encoded_file)
    return df


def draw_wind_pie(df):
    """Vẽ biểu đồ tròn cho wind"""
    wind = df['wind']
    bins = [0, 10, 20, 30, 100]
    labels = ['0–10 km/h', '10–20 km/h', '20–30 km/h', '>30 km/h']

    wind_groups = pd.cut(wind, bins=bins, labels=labels, include_lowest=True)
    counts = wind_groups.value_counts()

    plt.figure(figsize=(7, 7))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Phân phối tốc độ gió (wind)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    draw_wind_pie(df)


if __name__ == "__main__":
    main()