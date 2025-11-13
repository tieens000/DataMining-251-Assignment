import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pathlib import Path


def run_feature_selection(df: pd.DataFrame, target_col: str):
    """
    Thực hiện Feature Selection bằng 2 hướng:
    1. Filter (Correlation): chọn feature tương quan mạnh với target.
    2. Embedded (Random Forest): chọn feature có độ quan trọng cao khi huấn luyện.
    """    
    # Tạo thư mục lưu kết quả
    output_dir = Path("Process/Reduced_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chia tập dữ liệu thành features (X) và target (y)
    X = df.drop(columns=target_col)
    y = df[target_col]

    # ---------------------------------------------------------
    # (1/2) FILTER METHOD: Dựa vào hệ số tương quan
    # ---------------------------------------------------------
    print("Đang chạy Filter Method (Correlation)...")
    
    # Chỉ tính tương quan trên các cột dạng số
    numeric_df = df.select_dtypes(include=[np.number])

    # Tính độ tương quan tuyệt đối giữa các cột và target
    corr_with_target = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    
    #  Ngưỡng này dựa trên quan sát EDA: loại bỏ feature yếu (corr < 0.1)
    CORR_THRESHOLD = 0.1
    relevant_features_filter = corr_with_target[corr_with_target > CORR_THRESHOLD].index
    
    # Loại bỏ cột target khỏi danh sách
    relevant_features_filter = relevant_features_filter.drop(target_col, errors='ignore')
    
    # Tạo DataFrame mới chỉ chứa các cột được giữ lại
    df_reduced_filter = X[relevant_features_filter]
    df_reduced_filter[target_col] = y
    
    # Lưu file
    output_file_filter = output_dir / "reduced_heart_disease_FILTERED.csv"
    df_reduced_filter.to_csv(output_file_filter, index=False)
    print(f"Đã lưu file Filtered: {len(relevant_features_filter)} cột được giữ lại.")
    
    # Huấn luyện mô hình Random Forest để đánh giá feature importance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Trích xuất độ quan trọng của từng feature
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Ngưỡng quan trọng: giữ lại feature có importance > 0.01
    # 👉 Dựa trên quan sát — loại bỏ cột ít ảnh hưởng đến dự đoán.
    IMPORTANCE_THRESHOLD = 0.01
    relevant_features_embedded = feature_importance_df[
        feature_importance_df['Importance'] > IMPORTANCE_THRESHOLD
    ]['Feature']
    
    # Tạo DataFrame sau khi giảm chiều
    df_reduced_embedded = X[relevant_features_embedded]
    df_reduced_embedded[target_col] = y
    
    # Lưu file
    output_file_embedded = output_dir / "reduced_heart_disease_EMBEDDED.csv"
    df_reduced_embedded.to_csv(output_file_embedded, index=False)
    print(f"      ✅ Đã lưu file Embedded: {len(relevant_features_embedded)} cột được giữ lại.")
    
    print("      🔍 Top 5 cột quan trọng nhất theo Random Forest:")
    print(feature_importance_df.head(5).to_string(index=False))


# --- HÀM 2: PCA REDUCTION ---

def run_pca_reduction(df: pd.DataFrame, target_col: str):
    """
    Thực hiện giảm chiều bằng PCA, giữ lại 95% phương sai.
    PCA giúp trích xuất đặc trưng tổng hợp, giảm nhiễu, tăng tốc huấn luyện.
    """    
    output_dir = Path("Process/Reduced_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=target_col)
    y = df[target_col]
    
    # ⚠️ PCA không xử lý được giá trị NaN — kiểm tra trước
    
    # Khởi tạo PCA với mục tiêu giữ lại 95% phương sai dữ liệu
    pca = PCA(n_components=0.95)
    
    # Fit & transform
    X_pca = pca.fit_transform(X)
    
    # Số thành phần PCA được giữ lại
    n_components = pca.n_components_
    print(f"   -> PCA đã nén {X.shape[1]} cột gốc xuống còn {n_components} cột (giữ 95% variance).")
    
    # Tạo DataFrame kết quả
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    df_reduced_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_reduced_pca[target_col] = y.values
    
    # Lưu file kết quả
    output_file_pca = output_dir / "reduced_heart_disease_PCA.csv"
    df_reduced_pca.to_csv(output_file_pca, index=False)
    print(f"Đã lưu vào file PCA: {output_file_pca}")


def run_reduction_pipeline(df: pd.DataFrame, target_col: str):
    """
    Hàm chính: Tải dữ liệu đã transformed và chạy cả 2 phương pháp reduction.
    """
    # Chạy 2 phương pháp giảm chiều
    run_feature_selection(df, target_col)
    run_pca_reduction(df, target_col)
    
    print(" 3 file kết quả đã được lưu tại thư mục 'Process/Reduced_data/'")
