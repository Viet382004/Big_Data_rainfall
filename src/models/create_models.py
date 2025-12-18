import os
import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def train_rainfall_model():

    # Tạo SparkSession
    spark = SparkSession.builder \
        .appName("RainfallLinearRegression") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
        .getOrCreate()

    print("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")

    # Đọc dữ liệu
    df = spark.read.csv("D:/SPARK/data/processed/rainfall_clean.csv", header=True, inferSchema=True)

    # Tạo column features
    numeric_cols = ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa"]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    df_final = assembler.transform(df)

    # Cấu hình training
    feature_columns = numeric_cols
    label_column = "rainfall_mm"
    test_size = 0.2
    seed = 42
    scale = True

    # Chia dữ liệu
    train_df, test_df = df_final.randomSplit([1 - test_size, test_size], seed=seed)
    print(f"  Train: {train_df.count()} mẫu")
    print(f"  Test : {test_df.count()} mẫu")

    # Pipeline stages
    stages = []
    if scale:
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withMean=True,
            withStd=True
        )
        stages.append(scaler)
        features_col = "scaled_features"
    else:
        features_col = "features"

    # Linear Regression
    lr = LinearRegression(
        featuresCol=features_col,
        labelCol=label_column,
        solver="normal"
    )
    stages.append(lr)

    pipeline = Pipeline(stages=stages)

    # Huấn luyện
    pipeline_model = pipeline.fit(train_df)
    lr_model = pipeline_model.stages[-1]

    # Dự đoán và đánh giá
    predictions = pipeline_model.transform(test_df)
    evaluators = {
        "RMSE": RegressionEvaluator(labelCol=label_column, metricName="rmse"),
        "MAE": RegressionEvaluator(labelCol=label_column, metricName="mae"),
        "R2": RegressionEvaluator(labelCol=label_column, metricName="r2")
    }
    metrics = {}
    for name, evaluator in evaluators.items():
        value = evaluator.evaluate(predictions)
        metrics[name] = value

    # Thống kê
    summary = lr_model.summary
    importance = lr_model.coefficients.toArray().tolist()

    # Hiển thị kết quả
    print("\n=== KẾT QUẢ HUẤN LUYỆN ===")
    print("Metrics trên tập test:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nR² trên tập train: {summary.r2:.4f}")
    print(f"RMSE trên tập train: {summary.rootMeanSquaredError:.4f}")
    print(f"\nFeature importance: {importance}")

    spark.stop()

    return {
        "pipeline_model": pipeline_model,
        "linear_model": lr_model,
        "predictions": predictions,
        "metrics": metrics,
        "feature_importance": importance
    }


def main():
    result = train_rainfall_model()
    return result


if __name__ == "__main__":
    main()