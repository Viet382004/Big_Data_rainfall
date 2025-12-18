import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. ĐỌC DỮ LIỆU GỐC
df = pd.read_csv(r"/data/raw/rainfall.csv")

# 2. CHUẨN HÓA CỘT NGÀY (LẤY THEO THÁNG)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M').astype(str)

# 3. GROUP THEO TỈNH + THÁNG
df_group = (
    df.groupby(['location', 'month'])['rainfall_mm']
      .mean()
      .reset_index()
)

# 4. PIVOT ĐỂ VẼ HEATMAP
df_pivot = df_group.pivot(
    index='location',
    columns='month',
    values='rainfall_mm'
)

# 5. VẼ HEATMAP
plt.figure(figsize=(10, 6))

sns.heatmap(
    df_pivot,
    cmap='YlGnBu',
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Heatmap phân bố lượng mưa trung bình theo tháng và tỉnh", fontsize=14)
plt.xlabel("Tháng", fontsize=10)
plt.ylabel("Tỉnh / Thành phố", fontsize=10)

plt.tight_layout()
plt.show()
