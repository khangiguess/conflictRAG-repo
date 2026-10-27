# Báo Cáo Kết Quả Thực Nghiệm: Retrieval-Augmented Generation Benchmark (RGB)

## 1. Thông Tin Chung Về Thí Nghiệm
* **Mô hình thực nghiệm (Evaluated Model):** `GPT 6.1-sol`
* **Cổng kết nối (API Provider):** Thông qua `CetriX API Key`
* **Bộ dữ liệu đánh giá:** Retrieval-Augmented Generation Benchmark (RGB) - Bản tiếng Anh (`en`)
* **Tham số cấu hình:** 
  * Nhiệt độ (`temperature`): $0.2$
  * Số lượng tài liệu truy xuất (`passage`): $5$ đoạn văn bản cho mỗi câu hỏi
* **Tệp dữ liệu tổng hợp:** `RGB_Final_Summary.csv` (Được tổng hợp tự động từ 12 tệp kết quả JSON riêng lẻ tương ứng với các kịch bản thí nghiệm khác nhau).

---

## 2. Giải Mã Quy Tắc Đặt Tên File & Phân Loại Thí Nghiệm

Mặc dù tên file đầu ra được giữ theo định dạng mặc định của framework RGB (`prediction_en_..._chatgpt_...`), toàn bộ kết quả thực tế bên trong đều được sinh bởi **GPT 6.1-sol** qua **CetriX API Key**. 

Cấu trúc tên file:
`prediction_[ngôn_ngữ]_[loại_testbed]_chatgpt_temp0.2_noise[tỷ_lệ_nhiễu]_passage5_correct0.0_[loại_đánh_giá].json`

* **Phân biệt hậu tố đánh giá:**
  * `_result.json`: Đánh giá bằng phương pháp so khớp chính xác (**Exact Matching**) với đáp án chuẩn hoặc câu mẫu quy định trong System Prompt.
  * `_chatgptresult.json`: Đánh giá nâng cao bằng LLM để kiểm tra xem mô hình có thực sự từ chối trả lời (`Rej*`) hoặc phát hiện lỗi sai sự thật (`ED*`) về mặt ngữ nghĩa hay không.

### Bảng ánh xạ các file trong `RGB_Final_Summary.csv`

| STT | Tên File trong `RGB_Final_Summary.csv` | Nhóm Thí Nghiệm (RGB Testbed) | Số mẫu (`Samples`) | Ý nghĩa tham số |
| :--- | :--- | :--- | :---: | :--- |
| **0 - 4** | `prediction_en_chatgpt_temp0.2_noise0.[0-8]_passage5_correct0.0_result.json` | **1. Noise Robustness**<br>*(Độ bền vững trước nhiễu)* | $300$ | Đánh giá độ chính xác khi tỷ lệ tài liệu nhiễu tăng dần từ $0.0 \rightarrow 0.8$. |
| **5 - 6** | `prediction_en_chatgpt_temp0.2_noise1.0_passage5_correct0.0_[result / chatgptresult].json` | **2. Negative Rejection**<br>*(Khả năng từ chối khi thiếu thông tin)* | $300$ | Tỷ lệ nhiễu $1.0$ ($100\%$ tài liệu không chứa đáp án). Đo lường khả năng từ chối trả lời. |
| **7 - 8** | `prediction_en_fact_chatgpt_temp0.2_noise0.0_passage5_correct0.0_[result / chatgptresult].json` | **4. Counterfactual Robustness**<br>*(Khả năng xử lý thông tin sai sự thật)* | $100$ | Từ khóa `_fact_`. Kiểm tra khả năng phát hiện và tự sửa lỗi khi tài liệu chứa thông tin sai lệch. |
| **9 - 11** | `prediction_en_int_chatgpt_temp0.2_noise0.[0-4]_passage5_correct0.0_result.json` | **3. Information Integration**<br>*(Khả năng tích hợp thông tin)* | $100$ | Từ khóa `_int_`. Đánh giá khả năng tổng hợp đáp án từ nhiều tài liệu ở các mức nhiễu $0.0, 0.2, 0.4$. |

---

## 3. Chi Tiết Kết Quả Thực Nghiệm (GPT 6.1-sol)

### 3.1. Thí nghiệm 1: Noise Robustness (Độ bền vững trước nhiễu)
Đánh giá trên $300$ câu hỏi cơ sở khi tăng dần số lượng văn bản gây nhiễu:

| Tỷ lệ nhiễu (`Noise Ratio`) | Tên File tương ứng | Accuracy (%) |
| :---: | :--- | :---: |
| **$0.0$** | `prediction_en_chatgpt_temp0.2_noise0.0_passage5_correct0.0_result.json` | **$95.33\%$** |
| **$0.2$** | `prediction_en_chatgpt_temp0.2_noise0.2_passage5_correct0.0_result.json` | **$94.67\%$** |
| **$0.4$** | `prediction_en_chatgpt_temp0.2_noise0.4_passage5_correct0.0_result.json` | **$95.00\%$** |
| **$0.6$** | `prediction_en_chatgpt_temp0.2_noise0.6_passage5_correct0.0_result.json` | **$93.67\%$** |
| **$0.8$** | `prediction_en_chatgpt_temp0.2_noise0.8_passage5_correct0.0_result.json` | **$91.33\%$** |

* **Nhận xét:** Mô hình **GPT 6.1-sol** thể hiện độ bền vững rất cao trước nhiễu. Ngay cả khi $80\%$ tài liệu đưa vào là nhiễu ($4/5$ đoạn văn bản không chứa đáp án), độ chính xác chỉ giảm nhẹ $4.00\%$ (từ $95.33\%$ xuống $91.33\%$), vượt trội hơn nhiều so với kết quả baseline gốc trong bài báo RGB ($76.00\%$).
* **Thống kê mô tả:** Trung bình ($\text{Mean}$) = $94.00\%$, Độ lệch chuẩn ($\text{Std}$) = $1.61\%$, Trung vị ($\text{Median}$) = $94.67\%$.

---

### 3.2. Thí nghiệm 2: Negative Rejection (Khả năng từ chối khi thiếu thông tin)
Đánh giá trên $300$ câu hỏi với tỷ lệ nhiễu `noise = 1.0` (toàn bộ tài liệu cung cấp đều không có thông tin trả lời):

| Phương pháp đánh giá | Tên File tương ứng | Chỉ số trong CSV | Giá trị (%) |
| :--- | :--- | :--- | :---: |
| **Exact Match (`Rej`)** | `..._noise1.0_passage5_correct0.0_result.json` | `Accuracy` (Khớp câu mẫu từ chối) | **$75.33\%$** |
| **LLM Eval (`Rej*`)** | `..._noise1.0_passage5_correct0.0_chatgptresult.json` | `RejStar` / `Accuracy` | **$32.00\%$** / **$44.00\%$** |

---

### 3.3. Thí nghiệm 3: Information Integration (Khả năng tích hợp thông tin)
Đánh giá trên $100$ câu hỏi phức hợp đòi hỏi tổng hợp thông tin từ nhiều đoạn văn bản khác nhau (`_int_`):

| Tỷ lệ nhiễu (`Noise Ratio`) | Tên File tương ứng | Accuracy (%) |
| :---: | :--- | :---: |
| **$0.0$** | `prediction_en_int_chatgpt_temp0.2_noise0.0_passage5_correct0.0_result.json` | **$81.00\%$** |
| **$0.2$** | `prediction_en_int_chatgpt_temp0.2_noise0.2_passage5_correct0.0_result.json` | **$79.00\%$** |
| **$0.4$** | `prediction_en_int_chatgpt_temp0.2_noise0.4_passage5_correct0.0_result.json` | **$72.00\%$** |

* **Nhận xét:** Khả năng tổng hợp đa tài liệu của **GPT 6.1-sol** đạt mức tốt ($81.00\%$ khi không nhiễu và giữ được $72.00\%$ ở mức nhiễu $0.4$).
* **Thống kê mô tả:** Trung bình ($\text{Mean}$) = $77.33\%$, Độ lệch chuẩn ($\text{Std}$) = $4.73\%$, Trung vị ($\text{Median}$) = $79.00\%$.

---

### 3.4. Thí nghiệm 4: Counterfactual Robustness (Xử lý thông tin sai sự thật)
Đánh giá trên $100$ câu hỏi mà tài liệu truy xuất bị chèn thông tin sai lệch so với thực tế (`_fact_`):

| Chỉ số đánh giá | Tên cột & File nguồn trong CSV | Giá trị (%) | Ý nghĩa |
| :--- | :--- | :---: | :--- |
| **$\text{Acc}_{\text{doc}}$** | `Accuracy` (`_fact_..._result.json`) | **$96.00\%$** | Tỷ lệ trả lời đúng câu hỏi bất chấp tài liệu bị làm sai lệch. |
| **$\text{ED}$ (Error Detection)** | `ED` (`_fact_..._result.json`) | **$99.00\%$** | Tỷ lệ phát hiện có lỗi sai sự thật trong tài liệu (khớp mẫu). |
| **$\text{ED}^*$ (LLM Error Detection)** | `RejStar` (`_fact_..._chatgptresult.json`) | **$99.00\%$** | Tỷ lệ phát hiện lỗi sai sự thật được đánh giá qua bộ chấm LLM. |
| **$\text{CR}$ (Error Correction Rate)** | `CR` (`_fact_..._result.json`) | **$96.97\%$** | Tỷ lệ sửa lại đáp án đúng thành công sau khi đã phát hiện ra lỗi trong tài liệu. |

---

## 4. Mã Nguồn Python Vẽ Biểu Đồ Trực Quan Hóa

Đoạn mã dưới đây đọc trực tiếp file `RGB_Final_Summary.csv`, phân tách từng loại thí nghiệm theo quy tắc tên file và xuất ra 2 cụm biểu đồ tổng hợp:

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc file dữ liệu tổng hợp
df = pd.read_csv('RGB_Final_Summary.csv')

# Thiết lập giao diện biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# ==============================================================================
# CỤM BIỂU ĐỒ 1: NOISE ROBUSTNESS & INFORMATION INTEGRATION (LINE CHARTS)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Thí nghiệm 1: Noise Robustness ---
nr_df = df[df['File'].str.contains('prediction_en_chatgpt_temp0.2_noise0.[0-8]')].copy()
nr_df['Noise Ratio'] = nr_df['File'].str.extract(r'noise([0-9.]+)_').astype(float)
nr_df = nr_df.sort_values('Noise Ratio')

ax1.plot(nr_df['Noise Ratio'], nr_df['Accuracy'], marker='o', linewidth=2.5, 
         markersize=8, color='#2b5c8f', label='GPT 6.1-sol (CetriX)')
for x, y in zip(nr_df['Noise Ratio'], nr_df['Accuracy']):
    ax1.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0, 8), 
                 ha='center', fontweight='bold')

ax1.set_title('Exp 1: Noise Robustness (GPT 6.1-sol)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Noise Ratio', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax1.set_ylim(85, 100)
ax1.legend(loc='lower left')

# --- Thí nghiệm 3: Information Integration ---
ii_df = df[df['File'].str.contains('prediction_en_int_chatgpt')].copy()
ii_df['Noise Ratio'] = ii_df['File'].str.extract(r'noise([0-9.]+)_').astype(float)
ii_df = ii_df.sort_values('Noise Ratio')

ax2.plot(ii_df['Noise Ratio'], ii_df['Accuracy'], marker='s', linewidth=2.5, 
         markersize=8, color='#d95f02', label='GPT 6.1-sol (CetriX)')
for x, y in zip(ii_df['Noise Ratio'], ii_df['Accuracy']):
    ax2.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0, 8), 
                 ha='center', fontweight='bold')

ax2.set_title('Exp 3: Information Integration (GPT 6.1-sol)', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Noise Ratio', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xticks([0.0, 0.2, 0.4])
ax2.set_ylim(65, 90)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig('RGB_Exp1_Exp3_LineCharts.png', dpi=300)
plt.show()

# ==============================================================================
# CỤM BIỂU ĐỒ 2: NEGATIVE REJECTION & COUNTERFACTUAL ROBUSTNESS (BAR CHARTS)
# ==============================================================================
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Thí nghiệm 2: Negative Rejection ---
rej_exact = df[df['File'].str.contains('noise1.0.*_result.json')]['Accuracy'].values[0]
rej_chat_row = df[df['File'].str.contains('noise1.0.*_chatgptresult.json')]
rej_star = rej_chat_row['RejStar'].values[0]
rej_chat_acc = rej_chat_row['Accuracy'].values[0]

metrics_nr = ['Rej (Exact Match\n/ Result Accuracy)', 'Rej* (LLM Eval\nRejStar)', 'LLM Eval\nAccuracy']
values_nr = [rej_exact, rej_star, rej_chat_acc]
colors_nr = ['#4a7c59', '#68a67d', '#8fc9a3']

bars3 = ax3.bar(metrics_nr, values_nr, color=colors_nr, width=0.55, edgecolor='black', linewidth=0.8)
for bar in bars3:
    height = bar.get_height()
    ax3.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

ax3.set_title('Exp 2: Negative Rejection (GPT 6.1-sol, Noise = 1.0)', fontsize=13, fontweight='bold', pad=12)
ax3.set_ylabel('Percentage (%)', fontsize=12)
ax3.set_ylim(0, 90)

# --- Thí nghiệm 4: Counterfactual Robustness ---
fact_exact = df[df['File'].str.contains('_fact_.*_result.json')]
fact_chat = df[df['File'].str.contains('_fact_.*_chatgptresult.json')]

metrics_cf = ['Acc_doc\n(With Counterfactual)', 'ED\n(Exact Match)', 'ED*\n(LLM Eval)', 'CR\n(Error Correction)']
values_cf = [
    fact_exact['Accuracy'].values[0],
    fact_exact['ED'].values[0],
    fact_chat['RejStar'].values[0],
    fact_exact['CR'].values[0]
]
colors_cf = ['#7570b3', '#9e9ac8', '#bcbddc', '#54278f']

bars4 = ax4.bar(metrics_cf, values_cf, color=colors_cf, width=0.55, edgecolor='black', linewidth=0.8)
for bar in bars4:
    height = bar.get_height()
    ax4.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

ax4.set_title('Exp 4: Counterfactual Robustness (GPT 6.1-sol)', fontsize=13, fontweight='bold', pad=12)
ax4.set_ylabel('Percentage (%)', fontsize=12)
ax4.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('RGB_Exp2_Exp4_BarCharts.png', dpi=300)
plt.show()
```