import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"/data/processed/rainfall_encoded.csv")
def draw_bar_rain(df):

    location_counts = df['location'].value_counts()

    plt.figure(figsize=(8, 6))

    sns.barplot(
        x=location_counts.index,
        y=location_counts.values,
        hue=location_counts.index,
        palette="Blues_d",
        legend=False

    )

    # Thêm giá trị trên từng cột
    for i, v in enumerate(location_counts.values):
        plt.text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=10)

    plt.title("Phân bố số lượng dữ liệu theo tỉnh", fontsize=16, fontweight='bold')
    plt.xlabel("Tỉnh", fontsize=12)
    plt.ylabel("Số lượng bản ghi", fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(axis='y', linestyle='--', alpha=0.3)  # cho đẹp & dễ đọc giá trị
    plt.tight_layout()
    plt.show()

draw_bar_rain(df)