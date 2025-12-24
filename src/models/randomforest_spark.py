from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofmonth, month, year
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import time
import os

# 1. SPARK SESSION
spark = SparkSession.builder \
    .appName("Rainfall RandomForest - Spark Version") \
    .getOrCreate()


#  LOAD DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "rainfall_encoded.csv"
)

df = spark.read.csv(
    DATA_PATH,
    header=True,
    inferSchema=True
)

#  HANDLE DATE
df = df.withColumn("date", col("date").cast("date"))

df = df \
    .withColumn("day", dayofmonth("date")) \
    .withColumn("month", month("date")) \
    .withColumn("year", year("date"))

#  FEATURE & LABEL
feature_cols = [
    "location", "temp", "humidity", "wind", "pressure",
    "day", "month", "year"
]

label_col = "rain"

#  VECTOR ASSEMBLER
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

df_features = assembler.transform(df).select("features", label_col)

# TRAIN / TEST SPLIT
#Chia dữ liệu
train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

# RANDOM FOREST
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=label_col,
    numTrees=200,
    minInstancesPerNode=2,
    seed=42
)

# TRAIN
start_time = time.time()
model = rf.fit(train_df)
training_time = time.time() - start_time

print(f"Thời gian huấn luyện (Spark): {training_time:.2f} giây")
# 1. Chuẩn bị đặc trưng
feature_cols = ["location", "temp", "humidity", "wind", "pressure",
                "day", "month", "year"]

# 2. Vector Assembler để chuyển đổi sang định dạng Spark ML
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# 3. Mô hình Random Forest với các tham số
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="rain",
    numTrees=200,    # Số lượng cây trong rừng
    minInstancesPerNode=2,    # Số mẫu tối thiểu trong nút lá
    seed=42    # Random seed để tái lập kết quả
)
# Chia dữ liệu
train_df, test_df = df_features.randomSplit(weights=[0.8, 0.2], seed=42)
# PREDICT
predictions = model.transform(test_df)

# METRICS
rmse_eval = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="rmse"
)

r2_eval = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="r2"
)

# Test set
rmse = rmse_eval.evaluate(predictions)
r2_test = r2_eval.evaluate(predictions)

# Train set
train_predictions = model.transform(train_df)
r2_train = r2_eval.evaluate(train_predictions)

print("===== KẾT QUẢ =====")
print(f"RMSE: {rmse:.4f}")
print(f"R Square (Test): {max(0, r2_test):.4f}")
print(f"R Square (Train): {max(0, r2_train):.4f}")

spark.stop()
