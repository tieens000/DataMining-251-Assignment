import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any

# Tạo thư mục gốc một lần
OUTPUT_DIR = "Process/visualization/plot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HÀM MỚI: SO SÁNH DỮ LIỆU MISSING ---

def plot_missing_data_comparison(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    print("\n--- 📊 Đang vẽ So sánh Dữ liệu Missing (Trước vs Sau) ---")
    
    # ... (giữ nguyên logic tính toán của bạn) ...
    raw_missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
    raw_missing_df = raw_missing_pct.reset_index()
    raw_missing_df.columns = ['Column', 'Percentage']
    raw_missing_df['State'] = 'Raw'
    
    clean_missing_pct = (df_clean.isnull().sum() / len(df_clean)) * 100
    clean_missing_df = clean_missing_pct.reset_index()
    clean_missing_df.columns = ['Column', 'Percentage']
    clean_missing_df['State'] = 'Cleaned'
    
    all_cols = df_clean.columns.tolist()
    combined_df = pd.concat([raw_missing_df, clean_missing_df])
    combined_df = combined_df[combined_df['Column'].isin(all_cols)]
    
    cols_with_missing = raw_missing_df[raw_missing_df['Percentage'] > 0]['Column'].unique()
    if len(cols_with_missing) == 0:
        print("   ✅ Không tìm thấy dữ liệu missing ở dữ liệu Raw. Bỏ qua biểu đồ.")
        return
        
    combined_df = combined_df[combined_df['Column'].isin(cols_with_missing)]

    # 4. Vẽ biểu đồ
    # Tạo fig để có thể close nó sau
    fig = plt.figure(figsize=(15, 8))
    sns.barplot(
        x='Column', 
        y='Percentage', 
        hue='State', 
        data=combined_df, 
        palette={'Raw': 'salmon', 'Cleaned': 'lightgreen'}
    )
    plt.title('So sánh Tỉ lệ Missing Data (Trước vs. Sau Clean)', fontsize=16)
    plt.ylabel('Tỉ lệ Missing (%)')
    plt.xlabel('Cột')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # === SỬA LỖI 1 ===
    # LƯU TRƯỚC KHI HIỂN THỊ
    plt.savefig(
        f'{OUTPUT_DIR}/missing_data_comparison.png', 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig) # Đóng fig để giải phóng bộ nhớ

def plot_continuous_distributions_cleaned(df_clean: pd.DataFrame, continuous_cols: List[str]):
    print("\n--- 📊 Đang vẽ Phân phối Biến liên tục (Sau Clean) ---")
    num_cols = len(continuous_cols)
    
    # Lấy fig và axes để kiểm soát
    fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, num_cols * 4))
    if num_cols == 1: axes = [axes] 

    for i, col in enumerate(continuous_cols):
        ax = axes[i]
        if col in df_clean.columns:
            sns.kdeplot(df_clean[col].dropna(), ax=ax, label='Cleaned', color='green', fill=True, alpha=0.5)
            ax.set_title(f'Phân phối của {col} (Sau Clean)', fontsize=14)
            ax.legend()
        
    plt.tight_layout()
    
    # === SỬA LỖI 1 ===
    plt.savefig(
        f'{OUTPUT_DIR}/continuous_distributions_cleaned.png',
        dpi=300, 
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig) # Đóng fig

def plot_categorical_distributions_cleaned(df_clean: pd.DataFrame, categorical_cols: List[str]):
    print("\n--- 📊 Đang vẽ Phân phối Biến phân loại (Sau Clean) ---")
    
    for col in categorical_cols:
        if col not in df_clean.columns:
            continue
            
        # Tạo fig mới cho MỖI vòng lặp
        fig = plt.figure(figsize=(10, 5))
        
        try:
            all_categories = sorted(list(df_clean[col].dropna().unique()))
        except Exception:
            all_categories = None 

        ax = sns.countplot(x=col, data=df_clean, palette='deep', order=all_categories)
        ax.set_title(f'Phân phối {col} (Sau Clean)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        
        # === SỬA LỖI 1 và 2 ===
        # Lỗi 2: Thêm {col} vào tên file để không bị ghi đè
        file_name = f'categorical_dist_{col}.png'
        plt.savefig(
            f'{OUTPUT_DIR}/{file_name}',
            dpi=300, 
            bbox_inches='tight'
        )
        
        # Lỗi 1: show() sau khi save()
        plt.show()
        plt.close(fig) # Đóng fig này trước khi bắt đầu vòng lặp mới

def plot_boxplots_cleaned(df_clean: pd.DataFrame, continuous_cols: List[str]):
    print("\n--- 📊 Đang vẽ Boxplots (Sau Clean) ---")
    
    for col in continuous_cols:
        if col not in df_clean.columns:
            continue
            
        # Tạo fig mới cho MỖI vòng lặp
        fig = plt.figure(figsize=(7, 5))
        ax = sns.boxplot(y=df_clean[col], color='lightgreen')
        ax.set_title(f'Boxplot {col} (Sau Clean)', fontsize=14)
        plt.tight_layout()
        
        # === SỬA LỖI 1 và 2 ===
        # Lỗi 2: Thêm {col} vào tên file
        file_name = f'boxplot_{col}.png'
        plt.savefig(
            f'{OUTPUT_DIR}/{file_name}',
            dpi=300, 
            bbox_inches='tight'
        )
        
        # Lỗi 1: show() sau khi save()
        plt.show()
        plt.close(fig) # Đóng fig

def plot_correlation_cleaned(df_clean: pd.DataFrame):
    print("\n--- 📊 Đang vẽ Ma trận tương quan (Sau Clean) ---")
    
    df_clean_numeric = df_clean.select_dtypes(include=np.number)
    corr_clean = df_clean_numeric.corr()

    # Tạo fig
    fig = plt.figure(figsize=(15, 12))
    sns.heatmap(corr_clean, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Ma trận Tương quan (Sau Clean)', fontsize=16)
    plt.tight_layout()
    
    # === SỬA LỖI 1 ===
    # (Lưu ý: Tôi thống nhất lưu vào cùng thư mục OUTPUT_DIR)
    plt.savefig(
        f'{OUTPUT_DIR}/correlation_cleaned.png',
        dpi=300, 
        bbox_inches='tight'
    )
    plt.show()
    plt.close(fig)
# Hàm chính để gọi tất cả các biểu đồ
def run_visualizations(df_raw: pd.DataFrame, df_clean: pd.DataFrame, continuous_cols: List[str], categorical_cols: List[str]):
    plot_missing_data_comparison(df_raw, df_clean)
    plot_continuous_distributions_cleaned(df_clean, continuous_cols)
    plot_categorical_distributions_cleaned(df_clean, categorical_cols)
    plot_boxplots_cleaned(df_clean, continuous_cols)
    plot_correlation_cleaned(df_clean)