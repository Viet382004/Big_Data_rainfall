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


def draw_rain_by_month(df):
    """Vẽ biểu đồ cột lượng mưa theo tháng"""
    df['rain'] = pd.to_numeric(df['rain'], errors='coerce')
    df['date'] = pd.to_numeric(df['date'], errors='coerce')
    df = df.rename(columns={'date': 'Month'})

    rainfall_by_month = df.groupby('Month')['rain'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='Month',
        y='rain',
        data=rainfall_by_month,
        palette="GnBu",
        hue='Month',
        legend=False
    )

    plt.title("Lượng mưa trung bình theo tháng (0–13)", fontsize=14)
    plt.xlabel("Tháng (0–13)", fontsize=12)
    plt.ylabel("Lượng mưa trung bình (mm)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    draw_rain_by_month(df)


if __name__ == "__main__":
    main()