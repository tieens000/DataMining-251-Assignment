from Process.Integration.Data_integration import extract_all_data, OUTPUT_CSV

def main():
    df = extract_all_data()
    if df is not None:
        print(f"📊 Thông tin dữ liệu:")
        print(f"   - Số dòng: {len(df)}")
        print(f"   - Số cột: {len(df.columns)}")
        print(f"   - File output: {OUTPUT_CSV}")
    else:
        print("❌ Không tạo được file csv")

if __name__ == "__main__":
    main()