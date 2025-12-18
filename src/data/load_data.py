import os
import shutil
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, trim, initcap, year, month, when
from pyspark.sql.types import DoubleType, IntegerType
import pandas as pd

# Khởi tạo Spark
spark = SparkSession.builder \
    .appName("Rainfall Data Preprocessing") \
    .master("local[*]") \
    .getOrCreate()

# Đường dẫn
current_dir = Path(__file__).parent  # src/data
project_root = current_dir.parent.parent  # D:/SPARK
input_file = project_root / "data" / "raw" / "rainfall.csv"
output_file = project_root / "data" / "processed" / "rainfall_clean.csv"

print(f"Đang đọc file: {input_file}")

# Đọc file CSV
df = spark.read.option("header", True).option("inferSchema", True).csv(str(input_file))

print(f"\nDataset có {df.count()} dòng và {len(df.columns)} cột")

# Chuẩn hóa tên cột (lowercase và strip)
df = df.select([col(c).alias(c.lower().strip()) for c in df.columns])

# Đổi tên cột ngắn gọn (giữ nguyên để phù hợp với EDA)
# rename_mapping = {
#     'rainfall_mm': 'rain',
#     'temperature_c': 'temp',
#     'humidity_pct': 'humidity',
#     'wind_speed_kmh': 'wind',
#     'pressure_hpa': 'pressure'
# }

# for old_name, new_name in rename_mapping.items():
#     if old_name in df.columns:
#         df = df.withColumnRenamed(old_name, new_name)

# Chuẩn hóa kiểu dữ liệu
# Date
df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

# Tạo cột season từ month (giống EDA)
df = df.withColumn("month", month("date"))
df = df.withColumn(
    "season",
    when(col("month").isin(3, 4, 5), "spring")
    .when(col("month").isin(6, 7, 8), "summer")
    .when(col("month").isin(9, 10, 11), "autumn")
    .otherwise("winter")
)

# Chuẩn hóa kiểu dữ liệu số
numeric_cols = ['rainfall_mm', 'temperature_c', 'humidity_pct',
                'wind_speed_kmh', 'pressure_hpa']
for col_name in numeric_cols:
    if col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

# Chuẩn hóa text
text_cols = ['location', 'rain_category']
for col_name in text_cols:
    if col_name in df.columns:
        # Strip và title case
        df = df.withColumn(col_name, initcap(trim(col(col_name))))

# Loại dòng chứa NA
initial_count = df.count()
df_clean = df.dropna()
na_removed = initial_count - df_clean.count()
print(f"\nĐã xóa {na_removed} dòng có NA")

# Loại giá trị ngoại lai (cập nhật theo EDA)
outlier_count = df_clean.count()
df_clean = df_clean.filter(
    (col('pressure_hpa') > 950) &
    (col('pressure_hpa') < 1050) &
    (col('wind_speed_kmh') < 60) &
    (col('rainfall_mm') >= 0) &
    (col('rainfall_mm') <= 200)  # Giới hạn theo phân phối
)
outlier_removed = outlier_count - df_clean.count()
print(f"Đã xóa {outlier_removed} dòng có giá trị ngoại lai")

# Loại dòng trùng lặp
duplicate_count = df_clean.count()
df_clean = df_clean.dropDuplicates()
duplicate_removed = duplicate_count - df_clean.count()
print(f"Đã xóa {duplicate_removed} dòng trùng lặp")

# Lưu file sạch
print(f"\nĐang lưu file sạch...")
output_file.parent.mkdir(parents=True, exist_ok=True)

# Chuyển Spark DataFrame sang Pandas
df_pandas_clean = df_clean.toPandas()

# Ghi file CSV bằng Pandas
df_pandas_clean.to_csv(output_file, index=False, encoding='utf-8')
print(f"✓ File sạch đã được tạo: {output_file}")
print(f"✓ Số dòng: {len(df_pandas_clean)}, Số cột: {len(df_pandas_clean.columns)}")

# Hiển thị thông tin
print(f"\n=== THÔNG TIN FILE SẠCH ===")
print(f"5 dòng đầu của dữ liệu đã xử lý:")
print(df_pandas_clean.head().to_string())

print(f"\nCác cột hiện có:")
for i, col_name in enumerate(df_pandas_clean.columns, 1):
    print(f"  {i}. {col_name}")

# Dừng Spark
spark.stop()