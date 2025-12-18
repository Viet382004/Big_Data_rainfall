import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Tải dữ liệu"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    encoded_file = project_root / "data" / "processed" / "rainfall_clean.csv"
    df = pd.read_csv(encoded_file)
    return df

def draw_pie_location(df):
    """Vẽ biểu đồ tròn cho location"""
    location_counts = df['location'].value_counts()

    plt.figure(figsize=(10, 6))
    plt.pie(
        location_counts,
        labels=location_counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Tỷ lệ phân bố theo Location", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    draw_pie_location(df)

if __name__ == "__main__":
    main()