import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
encoded_file = project_root / "data" / "processed" / "rainfall_clean.csv"

def load_data():
    """Tải dữ liệu"""
    df = pd.read_csv(encoded_file)
    return df

def draw_box_plot(df):
    """Tạo boxplot cho các cột định lượng"""
    df_num = df.select_dtypes(include='number')

    plt.figure(figsize=(12, 6))
    plt.boxplot(df_num.values, patch_artist=True)
    plt.xlabel("Cột số", fontsize=12)
    plt.ylabel("Giá trị", fontsize=12)
    plt.title("Boxplot của các cột định lượng", fontsize=14, fontweight='bold')
    plt.grid(linestyle='solid', linewidth=0.4)
    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    draw_box_plot(df)

if __name__ == "__main__":
    main()