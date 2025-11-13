import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from pathlib import Path



def create_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    df['high_chol_flag'] = (df['chol'] > 240).astype(int)
    df['low_hr_flag'] = (df['thalach'] < 100).astype(int)
    df['oldpeak_risk'] = (df['oldpeak'] > 2).astype(int)
    
    # DỰA TRÊN PHÂN TÍCH CORRELATION:
    # Chúng ta đã quyết định loại bỏ các thuộc tính tương tác/tỷ lệ
    # vì 'chol' và 'trestbps' có tương quan yếu với 'num'.
    
    # df['age_x_chol'] = df['age'] * df['chol']
    # df['bp_chol_ratio'] = np.where(df['chol'] > 0, df['trestbps'] / df['chol'], 0)
    return df

def create_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các thuộc tính dạng 'giỏ' (binned) cho các cột số."""
    age_bins = [0, 40, 55, np.inf]
    age_labels = ['Age_Young', 'Age_Middle', 'Age_Old']
    df['age_binned'] = pd.cut(
        df['age'],
        bins=age_bins,
        labels=age_labels,
        right=True
    )

    # Chol Binning (dùng pd.qcut để chia 3 nhóm bằng nhau)
    chol_labels = ['Chol_Low', 'Chol_Medium', 'Chol_High']
    df['chol_binned'] = pd.qcut(
        df['chol'],
        q=3,
        labels=chol_labels,
        duplicates='drop'
    )
    return df

def encode_categorical_features(df: pd.DataFrame, cols_to_encode: List[str]) -> pd.DataFrame:
    """Mã hóa One-Hot các cột phân loại."""    
    df = pd.get_dummies(
        df, 
        columns=cols_to_encode, 
        drop_first=True # Tránh đa cộng tuyến
    )
    return df

def standardize_numeric_features(df: pd.DataFrame, cols_to_scale: List[str]) -> pd.DataFrame:
    """Chuẩn hóa Z-score (StandardScaler) các cột số liên tục."""    
    # Chỉ scale các cột có trong DataFrame
    valid_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
    
    scaler = StandardScaler()
    df[valid_cols_to_scale] = scaler.fit_transform(df[valid_cols_to_scale])
    
    return df


def run_transformation_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    
    df_transformed = df.copy()
    # Cột liên tục gốc để chuẩn hóa
    cols_for_scaling = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Cột phân loại gốc để mã hóa
    cols_for_encoding = ['cp', 'thal', 'slope', 'restecg']
    
    # Bước 1: Tạo thuộc tính cờ
    df_transformed = create_risk_flags(df_transformed)
    
    # Bước 2: Tạo thuộc tính binned
    df_transformed = create_binned_features(df_transformed)
    
    # Cập nhật danh sách mã hóa: Thêm các cột binned mới
    new_binned_cols = ['age_binned', 'chol_binned']
    cols_for_encoding.extend(new_binned_cols)

    # Bước 3: Mã hóa One-Hot
    df_transformed = encode_categorical_features(df_transformed, cols_for_encoding)

    # Bước 4: Chuẩn hóa
    # (Lưu ý: Chúng ta không thêm các cột cờ mới vào cols_for_scaling
    # vì chúng đã là dạng 0/1)
    df_transformed = standardize_numeric_features(df_transformed, cols_for_scaling)
    
    output_dir = Path("Process/Transformed_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "transformed_heart_disease.csv"
    df_transformed.to_csv(output_file, index=False)
    print(f"Kết quả đã được lưu tại: {output_file}")
    
    return df_transformed

