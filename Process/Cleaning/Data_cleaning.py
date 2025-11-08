from matplotlib.rcsetup import validate_any
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

# Các cột dữ liệu cần chọn để phân tích
SELECTED_COLUMNS = [
    'age', 'sex', 'num', 'cp', 'thal', 'ca', 'oldpeak', 'exang', 
    'trestbps', 'chol', 'thalach', 'slope', 'restecg', 'htn', 'dm', 
    'famhist', 'fbs', 'prop', 'nitr', 'pro', 
    'diuretic', 'xhypo'
]
# Các cột quan trọng cần giữ lại
TRULY_CRITICAL_COLUMNS = ['age', 'sex', 'num']

def lower_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển đổi tất cả tên cột thành chữ thường để chuẩn hóa định dạng."""
    df.columns = [col.lower() for col in df.columns] 
    return df

def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Chọn các cột trong DataFrame dựa trên danh sách columns cung cấp."""
    # Chỉ giữ các cột có trong DataFrame và columns đầu vào
    return df[[col for col in columns if col in df.columns]]

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các giá trị missing cho tất cả các cột."""
    missing_indicators = [
        '-9', -9, 'NULL', 'null', 'NA', 'N/A', '', ' ', '?', 'nan', 'NaN', 
        'none', 'None', 'NAN', 'Null', 'missing', 'Missing', 'MISSING',
        '.', '..', '...', 'undefined', 'Undefined', '#N/A', '#NULL!', '#DIV/0!'
    ]
    df_clean = df.copy()
    # Xử lý missing cho từng cột với đầy đủ điều kiện đặc thù
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(missing_indicators, np.nan)
        # thal chỉ cho phép 3,6,7 vì 3 là normal, 6 là reversible defect, 7 là permanent defect
        if col == 'thal':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean.loc[~df_clean[col].isin([3, 6, 7]), col] = np.nan  
        # slope chỉ cho phép 1,2,3 vì 1 là upsloping, 2 là flat, 3 là downsloping
        elif col == 'slope':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean.loc[df_clean[col] == 0, col] = np.nan  
        # ca chỉ cho phép 0,1,2,3 vì 0 là normal, 1 là single, 2 là double, 3 là triple
        elif col == 'ca' and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean.loc[(df_clean[col] < 0) | (df_clean[col] > 3), col] = np.nan  
    return df_clean

def handle_missing_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Xử lý các giá trị thiếu (missing data) trong DataFrame đầu vào."""
    stats = {"rows_removed": 0, "values_imputed": 0, "columns_dropped": 0}

    print("🧠 Processing missing data...")
    df_clean = clean_missing_values(df)

    before_empty = len(df_clean)
    df_clean = df_clean.dropna(how='all')  # Bỏ dòng toàn bộ đều thiếu
    empty_removed = before_empty - len(df_clean)

    # Xác định các cột quan trọng còn tồn tại
    critical_cols_exist = [col for col in TRULY_CRITICAL_COLUMNS if col in df_clean.columns]
    critical_removed = 0
    if critical_cols_exist:
        before_critical = len(df_clean)
        # Loại bỏ dòng thiếu tất cả các giá trị ở các cột quan trọng
        df_clean = df_clean[~df_clean[critical_cols_exist].isnull().all(axis=1)]
        critical_removed = before_critical - len(df_clean)
    stats["rows_removed"] = empty_removed + critical_removed

    if stats["rows_removed"] > 0:
        print(f"   🗑️  Removed {stats['rows_removed']} unusable rows because of missing data") 

    # Loại bỏ các cột mà toàn bộ giá trị là NaN, trừ cột cực kỳ quan trọng
    columns_to_drop = [
        col for col in df_clean.columns 
        if df_clean[col].isnull().all() and col not in TRULY_CRITICAL_COLUMNS
    ]
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        stats["columns_dropped"] = len(columns_to_drop)
        print(f"   ✅ Removed {len(columns_to_drop)} columns with all missing values: {columns_to_drop}")      


    # Tổng hợp danh sách dòng cần loại bỏ vì tỉ lệ missing lớn
    rows_to_remove = []
    for col in df_clean.columns:
        if not df_clean[col].isnull().any():
            continue  # Cột này không thiếu giá trị nào
        missing_count = df_clean[col].isnull().sum()
        missing_rate = missing_count / len(df_clean)
        print(f"   🔍 {col}: {missing_count} values ({missing_rate*100:.1f}% missing)")
        if missing_rate > 0.05:
            # Nếu tỉ lệ missing trong cột này >5%, đánh dấu các dòng cần loại bỏ (ưu tiên loại dòng)
            print(f"   ⚠️  {col}: High missing rate ({missing_rate*100:.1f}%) - marking rows for removal")
            rows_to_remove += df_clean[df_clean[col].isnull()].index.tolist()
            continue
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # Nếu là số, thay bằng median hoặc giá trị mặc định nếu không có median
            clinical_defaults = {'age': 55, 'trestbps': 130, 'chol': 200, 'thalach': 150}
            median_val = df_clean[col].median()
            fill_val = clinical_defaults.get(col, 0) if pd.isna(median_val) else median_val
            msg = (
                f"   🏥 {col}: filled {missing_count} values with clinical default ({fill_val})"
                if pd.isna(median_val) and col in clinical_defaults else
                (f"   ⚠️  {col}: filled {missing_count} values with 0 (no valid data for median)"
                 if pd.isna(median_val) else
                 f"   📊 {col}: filled {missing_count} values with median ({fill_val:.1f})"
                 )
            )
            print(msg)
            df_clean[col] = df_clean[col].fillna(fill_val)
        else:
            # Nếu là dạng phân loại hoặc ký tự, thay bằng mode nếu có
            mode_val = df_clean[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
            print(
                f"   📝 {col}: filled {missing_count} values with mode ({fill_val})"
                if not mode_val.empty else
                f"   ⚠️  {col}: filled {missing_count} values with 'Unknown' (no valid data for mode)"
            )
            df_clean[col] = df_clean[col].fillna(fill_val)
        stats["values_imputed"] += missing_count

    # Loại bỏ các dòng có tỉ lệ missing trên 5% ở một số cột (đã đánh dấu bên trên)
    if rows_to_remove:
        rows_to_remove = list(set(rows_to_remove))  # Loại bỏ lặp dòng
        rows_removed_count = len(rows_to_remove)
        df_clean = df_clean.drop(index=rows_to_remove).reset_index(drop=True)
        stats["rows_removed"] += rows_removed_count
        stats["rows_removed_high_missing"] = rows_removed_count
        print(f"   🗑️  Removed {rows_removed_count} rows with high missing rate data (>5%)")
    else:
        stats["rows_removed_high_missing"] = 0

    # Xử lý giá trị thiếu đặc biệt cho cột 'ca'
    if 'ca' in df_clean.columns and df_clean['ca'].isnull().any():
        missing_count = df_clean['ca'].isnull().sum()
        median_val = df_clean['ca'].median()
        fill_val = 0 if pd.isna(median_val) else median_val
        print(
            f"   🏥 ca: filled {missing_count} values with clinical default (0 vessels)"
            if pd.isna(median_val) else
            f"   🩺 ca: filled {missing_count} values with median ({fill_val:.0f} vessels)"
        )
        df_clean['ca'] = df_clean['ca'].fillna(fill_val)
        stats["values_imputed"] += missing_count

    # Chuyển kiểu dữ liệu các cột về đúng chuẩn, dùng high-order function try-except để tránh lỗi
    type_conversions = {
        'sex': 'int8', 'cp': 'int8', 'fbs': 'int8', 'restecg': 'int8',
        'exang': 'int8', 'slope': 'int8', 'thal': 'int8', 'ca': 'int8',
        'htn': 'int8', 'dm': 'int8', 'famhist': 'int8',
        'dig': 'int8', 'prop': 'int8', 'nitr': 'int8', 'pro': 'int8',
        'diuretic': 'int8', 'xhypo': 'int8'
    }
    # Sử dụng dict & loop cho chuyển kiểu và try/except để tránh crash
    for col, dtype in type_conversions.items():
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(dtype)
            except Exception:
                pass  # Nếu lỗi sẽ bỏ qua cột đó

    print(f"   ✅ Missing data processed: {stats['values_imputed']} values imputed")
    return df_clean, stats

def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Kiểm tra & làm sạch giá trị bất thường cho dự báo nguy cơ NMCT."""
    stats = {"invalid_values_corrected": 0}
    # Thiết lập các ngưỡng cho từng trường dữ liệu liên tục, nhị phân, phân loại.
    valid_ranges = {
        'age': (18, 100),                 # Chỉ chấp nhận bệnh nhân từ 18 đến 100 tuổi
        'trestbps': (70, 250),            # Chỉ chấp nhận huyết áp tâm thu từ 70 đến 250 mmHg
        'chol': (100, 600),               # Chỉ chấp nhận cholesterol từ 100 đến 600 mg/dL
        'thalach': (60, 220),             # Chỉ chấp nhận nhịp tim từ 60 đến 220 lần/phút
        'oldpeak': (0, 10),               # Chỉ chấp nhận oldpeak từ 0 đến 10
        'thal': (3, 7),                   # Chỉ chấp nhận thal từ 3 đến 7
    }
    binary_cols = [
        'sex', 'fbs', 'restecg', 'exang', 'htn',
        'dm', 'famhist', 'prop', 'nitr', 'pro', 
        'diuretic', 'xhypo'
    ]
    multiclass_ranges = {
        'cp': (0, 4),                     # Chỉ chấp nhận cp từ 0 đến 4
        'slope': (1, 3),                  # Chỉ chấp nhận slope từ 1 đến 3
        'num': (0, 4),                    # Chỉ chấp nhận num từ 0 đến 4
        'ca': (0, 3),
    }
    print("🔍 Validating data ranges for MI risk prediction...")

    # Kiểm tra & chỉnh sửa cột liên tục (continuous)
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            outlier_mask = (df[col] < min_val) | (df[col] > max_val)
            cnt = outlier_mask.sum()
            if cnt:
                df.loc[outlier_mask, col] = df[col].median()
                stats["invalid_values_corrected"] += cnt
                print(f"   🔧 {col}: fixed {cnt} values outside [{min_val}, {max_val}]")

    # Các cột dạng nhị phân (0 hoặc 1)
    for col in binary_cols:
        if col in df.columns:
            invalid_mask = ~df[col].isin([0, 1])
            cnt = invalid_mask.sum()
            if cnt:
                df.loc[invalid_mask, col] = 0
                stats["invalid_values_corrected"] += cnt
                print(f"   🔧 {col}: fixed {cnt} non-binary values")

    # Các cột dạng multiclass (categorical)
    for col, (min_val, max_val) in multiclass_ranges.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            cnt = invalid_mask.sum()
            if cnt:
                # Thay giá trị sai bằng mode của cột, hoặc 0 nếu không có mode
                mode = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                df.loc[invalid_mask, col] = mode
                stats["invalid_values_corrected"] += cnt
                print(f"   🔧 {col}: fixed {cnt} categorical values")

    # Đảm bảo 'num' chỉ là 0 hoặc 1 (bài toán nhị phân)
    if 'num' in df.columns:
        bin_mask = df['num'] > 1
        cnt = bin_mask.sum()
        if cnt:
            df.loc[bin_mask, 'num'] = 1
            stats["invalid_values_corrected"] += cnt
            print(f"   🎯 num: converted {cnt} values > 1 to 1 (binary classification)")
    return df, stats

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Loại bỏ các hàng trùng lặp ra khỏi DataFrame."""
    df_new = df.drop_duplicates()  # Trả về DataFrame mới không có dòng trùng
    removed = len(df) - len(df_new)
    if removed:
        print(f"   🗑️  Removed {removed} duplicate rows")
    return df_new, {"duplicates_removed": removed}
