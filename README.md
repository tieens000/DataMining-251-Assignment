# DataMining_251 - Dự án Khai thác Dữ liệu Bệnh Tim

Dự án khai thác dữ liệu và phân tích bệnh tim sử dụng bộ dữ liệu Heart Disease từ UCI Machine Learning Repository. Dự án bao gồm pipeline xử lý dữ liệu từ giai đoạn thu thập, tích hợp, làm sạch và chuẩn bị dữ liệu cho các mô hình machine learning.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Dependencies](#dependencies)
- [Nguồn dữ liệu](#nguồn-dữ-liệu)
- [Quy trình xử lý](#quy-trình-xử-lý)

## 🎯 Tổng quan

Dự án này tập trung vào việc xây dựng một pipeline xử lý dữ liệu hoàn chỉnh cho bộ dữ liệu bệnh tim, bao gồm:

- **Tích hợp dữ liệu (Data Integration)**: Tải và chuyển đổi dữ liệu thô từ UCI về định dạng CSV
- **Làm sạch dữ liệu (Data Cleaning)**: Xử lý missing values, validate dữ liệu, loại bỏ outliers và duplicates
- **Chuẩn bị dữ liệu**: Chọn lọc các thuộc tính quan trọng cho mô hình dự đoán nguy cơ nhồi máu cơ tim

## ✨ Tính năng

### Data Integration
- Tự động tải dữ liệu từ UCI Machine Learning Repository
- Chuyển đổi file `.data` sang định dạng CSV có cấu trúc
- Xử lý encoding linh hoạt (UTF-8, Latin-1, ISO-8859-1)
- Tự động tạo thư mục và quản lý đường dẫn

### Data Cleaning
- **Xử lý Missing Values**:
  - Nhận diện và chuẩn hóa các giá trị missing đa dạng
  - Imputation thông minh: median cho số liệu, mode cho categorical
  - Sử dụng giá trị mặc định lâm sàng khi cần
  - Loại bỏ dòng/cột có tỷ lệ missing quá cao (>5%)

- **Validation & Cleaning**:
  - Kiểm tra và sửa các giá trị ngoài phạm vi hợp lệ
  - Validate các cột nhị phân (0/1)
  - Validate các cột phân loại (categorical)
  - Chuyển đổi bài toán đa lớp thành nhị phân (num: 0/1)

- **Loại bỏ trùng lặp**: Tự động phát hiện và xóa các bản ghi trùng lặp

## 📁 Cấu trúc dự án

```
DataMining_251/
│
├── Main.py                          # File chính để chạy pipeline
├── README.md                        # Tài liệu dự án
├── requirements.txt                 # Danh sách các thư viện cần thiết
│
└── Process/
    ├── Integration/
    │   └── Data_integration.py      # Module tích hợp dữ liệu
    │
    ├── Cleaning/
    │   └── Data_cleaning.py         # Module làm sạch dữ liệu
    │
    └── Raw_data/
        ├── new.data                 # File dữ liệu thô từ UCI
        └── to_csv/
            └── raw_heart_disease.csv # File CSV đã chuyển đổi
```

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.7 trở lên
- pip (Python package manager)

### Các bước cài đặt

1. **Clone hoặc tải dự án về máy**

2. **Cài đặt các thư viện cần thiết**:
```bash
pip install -r requirements.txt
```

Hoặc cài đặt từng thư viện:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly jupyter
```

## 💻 Sử dụng

### Chạy pipeline tích hợp dữ liệu

Để tải và chuyển đổi dữ liệu từ UCI về CSV:

```bash
python Main.py
```

Script sẽ:
1. Tự động tải file `new.data` từ UCI (nếu chưa có)
2. Chuyển đổi sang file CSV có cấu trúc
3. Hiển thị thông tin về số dòng, số cột và đường dẫn file output

### Sử dụng các module riêng lẻ

#### Data Integration
```python
from Process.Integration.Data_integration import extract_all_data, ensure_raw_heart_disease_exists

# Tải và chuyển đổi dữ liệu
df = extract_all_data()

# Hoặc đảm bảo file CSV tồn tại (tự động tạo nếu chưa có)
csv_path = ensure_raw_heart_disease_exists()
```

#### Data Cleaning
```python
from Process.Cleaning.Data_cleaning import (
    lower_case_columns,
    select_columns,
    handle_missing_data,
    validate_and_clean_data,
    remove_duplicates
)
import pandas as pd

# Đọc dữ liệu
df = pd.read_csv('Process/Raw_data/to_csv/raw_heart_disease.csv')

# Chuẩn hóa tên cột
df = lower_case_columns(df)

# Chọn các cột cần thiết
SELECTED_COLUMNS = [
    'age', 'sex', 'num', 'cp', 'thal', 'ca', 'oldpeak', 'exang', 
    'trestbps', 'chol', 'thalach', 'slope', 'restecg', 'htn', 'dm', 
    'famhist', 'fbs', 'prop', 'nitr', 'pro', 
    'diuretic', 'xhypo'
]
df = select_columns(df, SELECTED_COLUMNS)

# Xử lý missing values
df, missing_stats = handle_missing_data(df)

# Validate và làm sạch dữ liệu
df, validation_stats = validate_and_clean_data(df)

# Loại bỏ duplicates
df, duplicate_stats = remove_duplicates(df)

# Lưu dữ liệu đã làm sạch
df.to_csv('Process/Cleaned_data/cleaned_heart_disease.csv', index=False)
```

## 📦 Dependencies

Các thư viện chính được sử dụng trong dự án:

- **pandas** (>=1.3.0): Xử lý và phân tích dữ liệu
- **numpy** (>=1.21.0): Tính toán số học
- **matplotlib** (>=3.5.0): Vẽ biểu đồ cơ bản
- **seaborn** (>=0.11.0): Vẽ biểu đồ thống kê nâng cao
- **scikit-learn** (>=1.0.0): Machine learning algorithms
- **scipy** (>=1.7.0): Thống kê và tính toán khoa học
- **plotly** (>=5.0.0): Biểu đồ tương tác
- **jupyter** (>=1.0.0): Môi trường notebook
- **requests**: Tải dữ liệu từ web (được sử dụng trong Data_integration.py)

## 📊 Nguồn dữ liệu

Bộ dữ liệu Heart Disease được lấy từ **UCI Machine Learning Repository**:
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
- **File**: `new.data`
- **Số lượng thuộc tính**: 76 cột
- **Mô tả**: Dữ liệu về các bệnh nhân tim mạch với các thông tin lâm sàng và kết quả chẩn đoán

### Các thuộc tính quan trọng được sử dụng:
- `age`: Tuổi
- `sex`: Giới tính (0: nữ, 1: nam)
- `num`: Mức độ bệnh tim (0: không bệnh, 1-4: có bệnh)
- `cp`: Loại đau ngực
- `trestbps`: Huyết áp tâm thu khi nghỉ
- `chol`: Cholesterol
- `thalach`: Nhịp tim tối đa đạt được
- `oldpeak`: ST depression
- `exang`: Đau thắt ngực do gắng sức
- `slope`: Độ dốc của đoạn ST
- `ca`: Số mạch máu chính được nhuộm màu
- `thal`: Thalassemia

## 🔄 Quy trình xử lý

### 1. Data Integration
```
UCI Repository (new.data) 
    ↓
[Download & Parse]
    ↓
CSV Format (raw_heart_disease.csv)
```

### 2. Data Cleaning Pipeline
```
Raw CSV
    ↓
[Chuẩn hóa tên cột]
    ↓
[Chọn các cột quan trọng]
    ↓
[Xử lý Missing Values]
    ├── Nhận diện missing values
    ├── Imputation (median/mode)
    └── Loại bỏ dòng/cột có missing rate cao
    ↓
[Validation & Cleaning]
    ├── Kiểm tra phạm vi giá trị
    ├── Sửa outliers
    └── Validate binary/categorical
    ↓
[Loại bỏ duplicates]
    ↓
Cleaned Data (sẵn sàng cho ML)
```

## 📝 Ghi chú

- Dự án được thiết kế để dễ dàng mở rộng với các module mới (như Data Transformation, Feature Engineering, Modeling)
- Tất cả các đường dẫn được quản lý tự động, đảm bảo tính nhất quán
- Code được viết với nhiều comment và thông báo để dễ theo dõi quá trình xử lý

## 👤 Tác giả

DataMining_251 - Dự án Khai thác Dữ liệu

## 📄 License

Dự án này được sử dụng cho mục đích học tập và nghiên cứu.

>>>>>>> 3fa28c8 (Initial)
