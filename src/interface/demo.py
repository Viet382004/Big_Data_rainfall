import os
import json
import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# ==================================================
# MODEL LOADER
# ==================================================
class RainfallPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base_dir))
        model_path = os.path.join(project_root, "models", "model_params.json")

        with open(model_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.coefficients = np.array(params["coefficients"])
        self.intercept = params["intercept"]
        self.feature_names = params["feature_names"]
        self.scaler_mean = np.array(params["scaler_mean"])
        self.scaler_std = np.array(params["scaler_std"])

    def predict(self, t, h, w, p, day, month, year):
        x = np.array(
            [t, h, w, p, day, month, year],
            dtype=float
        )
        x = (x - self.scaler_mean) / self.scaler_std
        y = np.dot(self.coefficients, x) + self.intercept
        return max(0.0, float(y))


predictor = RainfallPredictor()

# ==================================================
# LOAD METRICS
# ==================================================
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    metrics = joblib.load(os.path.join(project_root, "models", "metrics.pkl"))
except:
    metrics = {}

# ==================================================
# UI ROOT
# ==================================================
root = tk.Tk()
root.title("🌧 Dự báo lượng mưa")
root.geometry("900x760")
root.configure(bg="#f1f5f9")

main = tk.Frame(root, bg="white", padx=20, pady=15)
main.pack(fill="both", expand=True, padx=15, pady=15)

# ==================================================
# TITLE
# ==================================================
tk.Label(
    main,
    text="🌧 HỆ THỐNG DỰ BÁO LƯỢNG MƯA",
    font=("Segoe UI", 22, "bold"),
    fg="#1e4f91",
    bg="white"
).pack()

tk.Label(
    main,
    text="Linear Regression – Demo Python",
    font=("Segoe UI", 10, "italic"),
    bg="white"
).pack(pady=(0, 10))

# ==================================================
# INPUT – TIME
# ==================================================
time_frame = tk.LabelFrame(main, text="🕒 Thời gian", bg="white")
time_frame.pack(fill="x", pady=8)

today = datetime.now()

day_cb = ttk.Combobox(time_frame, values=list(range(1, 32)), width=6, state="readonly")
month_cb = ttk.Combobox(time_frame, values=list(range(1, 13)), width=6, state="readonly")
year_cb = ttk.Combobox(time_frame, values=list(range(2020, 2031)), width=8, state="readonly")

day_cb.set(today.day)
month_cb.set(today.month)
year_cb.set(today.year)

tk.Label(time_frame, text="Ngày:", bg="white").grid(row=0, column=0, padx=8, pady=5)
day_cb.grid(row=0, column=1, padx=8)

tk.Label(time_frame, text="Tháng:", bg="white").grid(row=0, column=2, padx=8)
month_cb.grid(row=0, column=3, padx=8)

tk.Label(time_frame, text="Năm:", bg="white").grid(row=0, column=4, padx=8)
year_cb.grid(row=0, column=5, padx=8)

# ==================================================
# INPUT – WEATHER
# ==================================================
weather_frame = tk.LabelFrame(main, text="🌦 Thông tin thời tiết", bg="white")
weather_frame.pack(fill="x", pady=8)

def wrow(label, widget, r):
    tk.Label(weather_frame, text=label, bg="white", width=18, anchor="w") \
        .grid(row=r, column=0, padx=10, pady=4, sticky="w")
    widget.grid(row=r, column=1, padx=10, pady=4, sticky="w")

location_cb = ttk.Combobox(
    weather_frame,
    values=["Hà Nội", "Đà Nẵng", "Huế", "TP.HCM", "Cần Thơ"],
    state="readonly",
    width=22
)
location_cb.set("Hà Nội")

temp_e = ttk.Entry(weather_frame, width=25)
hum_e = ttk.Entry(weather_frame, width=25)
wind_e = ttk.Entry(weather_frame, width=25)
pres_e = ttk.Entry(weather_frame, width=25)

temp_e.insert(0, "28")
hum_e.insert(0, "85")
wind_e.insert(0, "15")
pres_e.insert(0, "1008")

wrow("Khu vực:", location_cb, 0)
wrow("Nhiệt độ (°C):", temp_e, 1)
wrow("Độ ẩm (%):", hum_e, 2)
wrow("Gió (km/h):", wind_e, 3)
wrow("Áp suất (hPa):", pres_e, 4)

# ==================================================
# RESULT LOGIC
# ==================================================
result_var = tk.StringVar(value="Nhập dữ liệu và nhấn Dự báo")

def predict():
    try:
        t = float(temp_e.get())
        h = float(hum_e.get())
        w = float(wind_e.get())
        p = float(pres_e.get())
        day = int(day_cb.get())
        month = int(month_cb.get())
        year = int(year_cb.get())

        rain = round(
            predictor.predict(t, h, w, p, day, month, year),
            2
        )

        if rain >= 50:
            level, color = "MƯA LỚN", "#dc2626"
        elif rain >= 20:
            level, color = "MƯA VỪA", "#ea580c"
        elif rain >= 5:
            level, color = "MƯA NHỎ", "#ca8a04"
        else:
            level, color = "ÍT MƯA", "#16a34a"

        result_label.config(fg=color)
        result_var.set(
            f"{level}\n"
            f"Lượng mưa: {rain} mm\n"
            f"Khu vực: {location_cb.get()}"
        )

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

# ==================================================
# BUTTON
# ==================================================
ttk.Button(main, text="🌧 DỰ BÁO", command=predict).pack(pady=12)

# ==================================================
# RESULT DISPLAY
# ==================================================
res = tk.LabelFrame(main, text="📊 Kết quả & đánh giá", bg="white")
res.pack(fill="x")

result_label = tk.Label(
    res,
    textvariable=result_var,
    font=("Segoe UI", 15, "bold"),
    bg="white",
    justify="center"
)
result_label.pack(pady=8)

# ==================================================
# METRICS
# ==================================================
metrics_frame = tk.LabelFrame(main, text="📈 Hiệu suất mô hình (Test)", bg="white")
metrics_frame.pack(fill="x", pady=6)

tk.Label(
    metrics_frame,
    text=(
        f"MSE : {metrics.get('mse', 'N/A')}\n"
        f"RMSE: {metrics.get('rmse', 'N/A')}\n"
        f"MAE : {metrics.get('mae', 'N/A')}\n"
        f"R²  : {metrics.get('r2', 'N/A')}"
    ),
    font=("Consolas", 10),
    bg="white",
    justify="left"
).pack(anchor="w", padx=10)

# ==================================================
# COEFFICIENTS
# ==================================================
coef_frame = tk.LabelFrame(main, text="📐 Tham số hồi quy", bg="white")
coef_frame.pack(fill="x", pady=6)

coef_text = f"Intercept (β₀): {predictor.intercept:.4f}\n\n"
for name, coef in zip(predictor.feature_names, predictor.coefficients):
    coef_text += f"β ({name}): {coef:.4f}\n"

tk.Label(
    coef_frame,
    text=coef_text,
    font=("Consolas", 10),
    bg="white",
    justify="left"
).pack(anchor="w", padx=10)

root.mainloop()