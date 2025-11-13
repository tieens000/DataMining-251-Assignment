import os
import pandas as pd
import tkinter as tk

from Process.Integration.Data_integration import extract_all_data, OUTPUT_CSV
import Process.visualization.visualization as viz   
import Process.Tranformation.tranformation as trans
import Process.Reduction.reduction as red


def main():
    df = extract_all_data()
    if df is not None:
        print(f"📊 Thông tin dữ liệu:")
        print(f"   - Số dòng: {len(df)}")
        print(f"   - Số cột: {len(df.columns)}")
        print(f"   - File output: {OUTPUT_CSV}")
    else:
        print("❌ Không tạo được file csv")
    from Process.Cleaning.Data_cleaning import (
    lower_case_columns,
    select_columns,
    handle_missing_data,
    validate_and_clean_data,
    remove_duplicates
)
    
    df_raw = pd.read_csv('Process/Raw_data/to_csv/raw_heart_disease.csv')
    df_clean = pd.read_csv('Process/Cleaned_data/cleaned_heart_disease.csv')
    df_transformed = trans.run_transformation_pipeline(df_clean)
    red.run_reduction_pipeline(df_transformed,'num')
    

if __name__ == "__main__":
    main()