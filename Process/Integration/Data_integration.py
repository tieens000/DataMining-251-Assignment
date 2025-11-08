import os
import requests
import pandas as pd
import numpy as np

# Định nghĩa các đường dẫn phù hợp cấu trúc project (Process/Raw_data/)
BASE_RAW_DIR = os.path.join("Process", "Raw_data")
RAW_FILE = os.path.join(BASE_RAW_DIR, "new.data")
OUTPUT_DIR = os.path.join(BASE_RAW_DIR, "to_csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "raw_heart_disease.csv")

def extract_heart_disease_data():
    """
    Tải file 'new.data' từ UCI về Process/Raw_data/
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/new.data"
    os.makedirs(BASE_RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        print(f"✅ Đã có file: {RAW_FILE}")
        return
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(RAW_FILE, "wb") as f:
            f.write(response.content)
        print(f"✅ Tải file thành công: {RAW_FILE}")
    except Exception as e:
        print(f"⚠️ Không thể tải file new.data: {e}")

def convert_to_heart_disease_csv():
    """
    Chuyển file new.data thành raw_heart_disease.csv (Process/Raw_data/to_csv/)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(RAW_FILE):
        print(f"❌ Không tìm thấy file: {RAW_FILE}")
        return None
    # Tên các thuộc tính (76 cột)
    col_names = [
        "id", "ccf", "age", "sex", "painloc", "painexer", "relrest", "pncaden",
        "cp", "trestbps", "htn", "chol", "smoke", "cigs", "years", "fbs", "dm",
        "famhist", "restecg", "ekgmo", "ekgday", "ekgyr", "dig", "prop", "nitr",
        "pro", "diuretic", "proto", "thaldur", "thaltime", "met", "thalach",
        "thalrest", "tpeakbps", "tpeakbpd", "dummy", "trestbpd", "exang", "xhypo",
        "oldpeak", "slope", "rldv5", "rldv5e", "ca", "restckm", "exerckm",
        "restef", "restwm", "exeref", "exerwm", "thal", "thalsev", "thalpul",
        "earlobe", "cmo", "cday", "cyr", "num", "lmt", "ladprox", "laddist",
        "diag", "cxmain", "ramus", "om1", "om2", "rcaprox", "rcadist", "lvx1",
        "lvx2", "lvx3", "lvx4", "lvf", "cathef", "junk", "name"
    ]
    # Đọc file với encoding linh hoạt
    raw_data = None
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            with open(RAW_FILE, 'r', encoding=encoding) as f:
                raw_data = f.read()
            break
        except UnicodeDecodeError:
            continue
    if raw_data is None:
        with open(RAW_FILE, 'rb') as f:
            raw_data = f.read().decode('latin-1', errors='replace')

    # Ghép từng record (kết thúc bởi từ 'name')
    records, cur = [], []
    for line in raw_data.strip().split('\n'):
        line = line.strip()
        if not line: continue
        if "name" in line:
            cur.append("name")
            records.append(' '.join(cur))
            cur = []
        else:
            cur += line.split()
    if cur:  # Phòng trường hợp dòng cuối không có 'name'
        records.append(' '.join(cur))

    # Đảm bảo mỗi dòng dữ liệu có đúng 76 cột
    data_rows = []
    for rec in records:
        vals = rec.split()
        if len(vals) < 76:
            vals += [np.nan] * (76 - len(vals))
        elif len(vals) > 76:
            vals = vals[:76]
        data_rows.append(vals)

    df = pd.DataFrame(data_rows, columns=col_names)
    df.to_csv(OUTPUT_CSV, index=False, escapechar='\\')
    print(f"✅ Đã tạo file: {OUTPUT_CSV} (Tổng số dòng: {len(df)})")
    return df

def extract_all_data():
    """
    Chạy toàn bộ pipeline: tải + chuyển đổi dữ liệu về CSV (Process/Raw_data/to_csv/)
    """
    print("🚀 Bắt đầu pipeline trích xuất dữ liệu heart-disease...")
    extract_heart_disease_data()
    df = convert_to_heart_disease_csv()
    print("🎉 Toàn bộ pipeline hoàn thành!")
    return df

def ensure_raw_heart_disease_exists():
    """
    Đảm bảo file raw_heart_disease.csv tồn tại; tạo mới nếu chưa có
    """
    if os.path.exists(OUTPUT_CSV):
        print(f"✅ Đã có file csv: {OUTPUT_CSV}")
        return OUTPUT_CSV
    print(f"⚠️ Chưa có file csv, bắt đầu tạo mới...")
    if not os.path.exists(RAW_FILE):
        print("📥 Chưa có file new.data, thực thi pipeline đầy đủ...")
        df = extract_all_data()
    else:
        print("🔗 Đã có new.data, chuyển đổi sang csv...")
        df = convert_to_heart_disease_csv()
    if df is not None:
        print(f"✅ Đã tạo csv thành công: {OUTPUT_CSV}")
        return OUTPUT_CSV
    else:
        print("❌ Không tạo được file csv")
        return None
