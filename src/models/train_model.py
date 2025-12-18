from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
import os
import shutil


def train_and_save(df, stages, label_col, model_path):

    # Huấn luyện và lưu mô hình Linear Regression

    print("\n===== HUẤN LUYỆN MÔ HÌNH =====")
    print(f"Số samples: {df.count():,}")
    print(f"Label column: {label_col}")

    # Khởi tạo Linear Regression với hyperparameters từ EDA
    lr = LinearRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
        solver="normal"
    )

    # Tạo pipeline
    pipeline = Pipeline(stages=stages + [lr])

    # Huấn luyện mô hình
    model = pipeline.fit(df)

    # Lấy mô hình Linear Regression
    lr_model = model.stages[-1]

    # Hiển thị thông tin mô hình
    print("\n=== THÔNG TIN MÔ HÌNH ===")
    print(f"R²: {lr_model.summary.r2:.4f}")
    print(f"RMSE: {lr_model.summary.rootMeanSquaredError:.4f}")
    print(f"Intercept: {lr_model.intercept:.4f}")
    print(f"Số coefficients: {len(lr_model.coefficients)}")

    # Lưu mô hình (ghi đè nếu tồn tại)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    model.write().overwrite().save(model_path)
    print(f"\n✓ Đã lưu mô hình tại: {model_path}")

    return model


