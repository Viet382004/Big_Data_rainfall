import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# ======================
# HÀM DỰ BÁO LƯỢNG MƯA
# ======================
def predict_rainfall():
    try:
        # Ngày tháng
        day = day_cb.get()
        month = month_cb.get()
        year = year_cb.get()

        if not day or not month or not year:
            raise ValueError("Vui lòng chọn đầy đủ ngày tháng năm")

        date = f"{day}/{month}/{year}"

        # Dữ liệu thời tiết
        temp = float(temp_entry.get())
        humidity = float(humidity_entry.get())
        wind = float(wind_entry.get())

        # ======================
        # CÔNG THỨC DỰ BÁO (DEMO)
        # ======================
        rainfall = (
            0.4 * humidity +
            0.3 * wind -
            0.2 * temp
        )

        rainfall = max(0, round(rainfall, 2))  # không âm

        # Hiển thị kết quả
        result_label.config(
            text=f"📅 Ngày: {date}\n🌧 Lượng mưa dự báo: {rainfall} mm",
            fg="#0b5394"
        )

    except ValueError as e:
        messagebox.showerror("Lỗi nhập liệu", str(e))


# ======================
# CỬA SỔ CHÍNH
# ======================
root = tk.Tk()
root.title("Dự báo lượng mưa")
root.geometry("720x650")
root.configure(bg="#eaf2f8")

# ======================
# FRAME CHÍNH
# ======================
main_frame = tk.Frame(
    root,
    bg="white",
    padx=30,
    pady=30,
    relief="groove",
    bd=2
)
main_frame.pack(padx=30, pady=30, fill="both", expand=True)

# ======================
# TIÊU ĐỀ
# ======================
title = tk.Label(
    main_frame,
    text="🌧 HỆ THỐNG DỰ BÁO LƯỢNG MƯA",
    font=("Segoe UI", 20, "bold"),
    fg="#1f4e79",
    bg="white"
)
title.pack(pady=15)

# ======================
# NGÀY THÁNG NĂM
# ======================
date_frame = tk.Frame(main_frame, bg="white")
date_frame.pack(fill="x", pady=10)

tk.Label(date_frame, text="Ngày:", bg="white").grid(row=0, column=0, padx=5)
tk.Label(date_frame, text="Tháng:", bg="white").grid(row=0, column=2, padx=5)
tk.Label(date_frame, text="Năm:", bg="white").grid(row=0, column=4, padx=5)

day_cb = ttk.Combobox(date_frame, width=5, values=[f"{i:02d}" for i in range(1, 32)])
month_cb = ttk.Combobox(date_frame, width=5, values=[f"{i:02d}" for i in range(1, 13)])
year_cb = ttk.Combobox(date_frame, width=8, values=[str(i) for i in range(2020, 2031)])

today = datetime.now()
day_cb.set(today.strftime("%d"))
month_cb.set(today.strftime("%m"))
year_cb.set(today.strftime("%Y"))

day_cb.grid(row=0, column=1)
month_cb.grid(row=0, column=3)
year_cb.grid(row=0, column=5)

# ======================
# INPUT
# ======================
def create_input(label):
    frame = tk.Frame(main_frame, bg="white")
    frame.pack(fill="x", pady=8)

    tk.Label(frame, text=label, bg="white", width=20, anchor="w")\
        .pack(side="left")
    entry = ttk.Entry(frame)
    entry.pack(side="right", fill="x", expand=True)
    return entry

temp_entry = create_input("Nhiệt độ (°C):")
humidity_entry = create_input("Độ ẩm (%):")
wind_entry = create_input("Tốc độ gió (km/h):")

# ======================
# BUTTON
# ======================
ttk.Button(
    main_frame,
    text="🌧 DỰ BÁO LƯỢNG MƯA",
    command=predict_rainfall
).pack(pady=20)

# ======================
# KẾT QUẢ
# ======================
result_label = tk.Label(
    main_frame,
    text="",
    font=("Segoe UI", 14, "bold"),
    bg="white"
)
result_label.pack(pady=15)

# ======================
root.mainloop()