import pyspark.sql
from pyspark.ml.evaluation import RegressionEvaluator


def evaluate_model(model, df, label_col, model_name, model_path=None):

    # Đánh giá mô hình hồi quy tuyến tính

    print(f"\n===== ĐÁNH GIÁ MÔ HÌNH: {model_name} =====")

    # Dự đoán
    predictions = model.transform(df)

    # Tạo các evaluators
    evaluators = {
        "RMSE": RegressionEvaluator(labelCol=label_col, metricName="rmse"),
        "MAE": RegressionEvaluator(labelCol=label_col, metricName="mae"),
        "R2": RegressionEvaluator(labelCol=label_col, metricName="r2")
    }

    # Tính toán metrics
    metrics = {}
    for name, evaluator in evaluators.items():
        value = evaluator.evaluate(predictions)
        metrics[name] = value
        print(f"  {name}: {value:.4f}")

    # Thống kê mô hình
    lr_model = model.stages[-1]
    print("\n=== THỐNG KÊ MÔ HÌNH ===")
    print(f"  Số features: {len(lr_model.coefficients)}")
    print(f"  Intercept: {lr_model.intercept:.4f}")

    # Hiển thị một vài dự đoán mẫu
    print("\n=== DỰ ĐOÁN MẪU ===")
    predictions.select("prediction", label_col).show(5)

    return metrics


