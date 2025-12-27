import os
import json
import joblib
import findspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofmonth, month, year
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    StringIndexer,
    OneHotEncoder
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

findspark.init()


def train_rainfall_model():

    # ==================================================
    # 1. SPARK SESSION
    # ==================================================
    spark = SparkSession.builder \
        .appName("RainfallLinearRegression") \
        .getOrCreate()

    # ==================================================
    # 2. LOAD DATA
    # ==================================================
    df = spark.read.csv(
        "D:/SPARK/data/processed/rainfall_clean.csv",
        header=True,
        inferSchema=True
    )

    # ==================================================
    # 3. FEATURE ENGINEERING (DATE → DAY, MONTH, YEAR)
    # ==================================================
    df = df \
        .withColumn("day", dayofmonth("date")) \
        .withColumn("month", month("date")) \
        .withColumn("year", year("date"))

    label_col = "rainfall_mm"

    numeric_cols = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_kmh",
        "pressure_hpa",
        "day",
        "month",
        "year"
    ]

    # ==================================================
    # 4. LOCATION ENCODING
    # ==================================================
    location_indexer = StringIndexer(
        inputCol="location",
        outputCol="location_index",
        handleInvalid="keep"
    )

    location_encoder = OneHotEncoder(
        inputCols=["location_index"],
        outputCols=["location_ohe"]
    )

    # ==================================================
    # 5. ASSEMBLE & SCALE
    # ==================================================
    assembler = VectorAssembler(
        inputCols=numeric_cols + ["location_ohe"],
        outputCol="features_raw"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    # ==================================================
    # 6. LINEAR REGRESSION (RIDGE)
    # ==================================================
    lr = LinearRegression(
        featuresCol="features",
        labelCol=label_col,
        regParam=0.1,
        elasticNetParam=0.0
    )

    pipeline = Pipeline(stages=[
        location_indexer,
        location_encoder,
        assembler,
        scaler,
        lr
    ])

    # ==================================================
    # 7. TRAIN / TEST SPLIT
    # ==================================================
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # ==================================================
    # 8. TRAIN MODEL
    # ==================================================
    model = pipeline.fit(train_df)

    # ==================================================
    # 9. EVALUATION
    # ==================================================
    pred_test = model.transform(test_df)

    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction"
    )

    metrics = {
        "mse": evaluator.setMetricName("mse").evaluate(pred_test),
        "rmse": evaluator.setMetricName("rmse").evaluate(pred_test),
        "mae": evaluator.setMetricName("mae").evaluate(pred_test),
        "r2": evaluator.setMetricName("r2").evaluate(pred_test)
    }

    lr_model = model.stages[-1]
    scaler_model = model.stages[-2]

    # ==================================================
    # 10. SAVE MODEL PARAMS (CHO DEMO PYTHON / TKINTER)
    # ==================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_params = {
        "coefficients": lr_model.coefficients.toArray().tolist()[:len(numeric_cols)],
        "intercept": float(lr_model.intercept),
        "feature_names": numeric_cols,
        "scaler_mean": scaler_model.mean.toArray().tolist()[:len(numeric_cols)],
        "scaler_std": scaler_model.std.toArray().tolist()[:len(numeric_cols)]
    }

    with open(
        os.path.join(models_dir, "model_params.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(model_params, f, indent=2)

    joblib.dump(
        {k: round(v, 4) for k, v in metrics.items()},
        os.path.join(models_dir, "metrics.pkl")
    )

    # ==================================================
    # 11. PRINT (NGẮN – RÕ – ĐÚNG BÁO CÁO)
    # ==================================================
    print("\n📊 ĐÁNH GIÁ MÔ HÌNH (TEST)")
    print(f"MSE : {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"R²  : {metrics['r2']:.4f}")

    print("\n📐 HỆ SỐ HỒI QUY")
    print(f"Intercept (β₀): {lr_model.intercept:.4f}")
    for name, coef in zip(numeric_cols, lr_model.coefficients.toArray()):
        print(f"β ({name}): {coef:.4f}")

    spark.stop()
    return model


def main():
    train_rainfall_model()


if __name__ == "__main__":
    main()
