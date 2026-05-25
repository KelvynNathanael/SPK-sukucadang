# SPK Suku Cadang Mobil — Streamlit App

UI Streamlit untuk skripsi **Kelvyn Nathanael Maulana (NIM 535220062)**:
*Sistem Pendukung Keputusan Penentuan Prioritas Model Kendaraan untuk Perencanaan
Suku Cadang Fast Moving Menggunakan Metode Fuzzy AHP dan Holt Winters.*

## 📁 Struktur File

```
spk_app/
├── app.py              # Streamlit app (UI utama)
├── fuzzy_ahp.py        # Modul Fuzzy AHP (Chang's Extent Analysis)
├── requirements.txt    # Dependencies Python
└── README.md
```

## 🚀 Cara Install & Run

```bash
# 1. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit
streamlit run app.py
```

App akan terbuka otomatis di browser di `http://localhost:8501`.

## 🎯 Fitur

### Sidebar — Input Preferensi Bengkel
- **Upload File Gaikindo** (.xlsx)
- **Slider Bobot Kriteria** Fuzzy AHP (skala -9 sampai +9)
  - Negatif → C2 (Kapasitas CC) lebih penting
  - Positif → C1 (Tren Populasi) lebih penting
- **Parameter EOQ:** target servis/bulan, biaya pesan, biaya simpan
- **Toggle Forecast** — aktifkan kalau mau pakai Holt-Winters

### Main Area — 4 Tab
1. **🏆 Ranking Prioritas** — tabel ranking model dengan progress bar untuk skor
2. **📈 Visualisasi Tren** — grafik historis + forecast (max 5 model sekaligus)
3. **🔢 Detail Fuzzy AHP** — matriks TFN, Si, bobot, uji konsistensi
4. **📦 Rekomendasi EOQ** — perhitungan kuantitas optimal per komponen

## 🔍 Tips

- Forecast Holt-Winters memakan waktu beberapa menit untuk dataset besar (ratusan
  model). Hasilnya di-cache, jadi run kedua untuk slider preferensi yang sama akan
  langsung cepat.
- Format kolom bulan di Excel harus `JAN_2012`, `FEB_2012`, dst.
- Model dengan data < 24 bulan otomatis difilter (syarat Holt-Winters).
- Hasil ranking bisa di-download sebagai CSV.

## 📊 Format Data Gaikindo yang Didukung

Kolom yang wajib ada di Excel:
- `BRAND`, `MODEL`, `CC`
- Kolom bulanan dengan format `JAN_YYYY`, `FEB_YYYY`, ..., `DEC_YYYY`
- Opsional: `CATEGORYTYPE` (untuk filter kendaraan berat)

## 🛠️ Customization

Mau tambah komponen baru di tab EOQ? Edit dictionary `komponen` di `app.py`:

```python
komponen = {
    "Filter Oli":   2.0,    # frekuensi ganti per tahun
    "Oli Mesin":    2.0,
    "Filter Udara": 1.0,
    "Kampas Rem":   0.5,
    "Busi":         0.5,
    # tambah baris baru di sini
}
```

Mau ubah skala TFN? Edit `TFN_SCALE` di `fuzzy_ahp.py`.
