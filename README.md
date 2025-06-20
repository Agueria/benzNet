
# BenzNet: Deep Learning Car Price Prediction

A machine learning regression project that predicts used car prices using the Mercedes-Benz dataset (`merc.xlsx`).  
This project demonstrates modern data preprocessing and deep neural network training with Keras/TensorFlow and scikit-learn.

---

## Features

- **Automatic data cleaning**: Removes outliers and missing/invalid data.
- **Categorical encoding**: One-hot encoding for categorical features.
- **Feature scaling**: Uses MinMaxScaler for stable neural network training.
- **Deep neural network**: Multi-layer regression model with regularization.
- **Performance visualization**: Plots loss curves and prediction results.
- **Full reproducibility**: Only `data.xlsx` and this code are required.

---

## Requirements

- Python 3.9+  
- See [`requirements.txt`](./requirements.txt) for all dependencies.

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

1. Place your `data.xlsx` data file in the project root directory.
2. Run the script:
   ```bash
   python benznet.py
   ```
3. The script will automatically clean, preprocess, train, and visualize results.

---

## Output

- Training and validation loss plots.
- Scatter plot of true vs predicted prices.
- Prints MAE, RMSE, and R² metrics in the console.

---

## File Structure

```
main/
│
├── benznet.py
├── requirements.txt
├── data.xlsx
└── README.md
```

---

## About the Dataset

The `benznet.xlsx` file should contain used car data (e.g., Mercedes-Benz).  
Required columns:  
- `price` (target value)
- Additional features (categorical/numeric, e.g. year, mileage, fuel type, model, etc.)

---

## License

This project is open source and free to use for academic or personal purposes.

---

## Contact

Feel free to open an issue or contact the project owner for support or questions.

---

---

# BenzNet: Derin Öğrenme ile Araba Fiyat Tahmini

Kullanılmış Mercedes-Benz araç fiyatlarını (`data.xlsx`) makine öğrenmesi ile tahmin eden bir regresyon projesi.  
Bu proje, modern veri ön işleme ve derin sinir ağı eğitimi adımlarını Keras/TensorFlow ve scikit-learn ile örnekler.

---

## Özellikler

- **Otomatik veri temizleme**: Aykırı ve eksik/veri hatalı kayıtlar temizlenir.
- **Kategorik kodlama**: Kategorik değişkenler one-hot encoding ile dönüştürülür.
- **Özellik ölçekleme**: Sinir ağı eğitimi için MinMaxScaler ile ölçekleme yapılır.
- **Derin sinir ağı**: Çok katmanlı regresyon modeli (düzenlileme içerir).
- **Performans görselleştirme**: Loss eğrileri ve tahmin sonuçları grafiklerle gösterilir.
- **Tam tekrar üretilebilirlik**: Sadece `data.xlsx` ve bu kod yeterlidir.

---

## Gereksinimler

- Python 3.9+  
- Tüm bağımlılıklar için [`requirements.txt`](./requirements.txt) dosyasına bakabilirsiniz.

Bağımlılıkları yüklemek için:
```bash
pip install -r requirements.txt
```

---

## Kullanım

1. `data.xlsx` veri dosyanızı proje ana dizinine koyun.
2. Scripti çalıştırın:
   ```bash
   python benznet.py
   ```
3. Kod otomatik olarak veri temizleme, ön işleme, eğitim ve görselleştirme işlemlerini yapacaktır.

---

## Çıktılar

- Eğitim ve doğrulama loss grafiklerini çizer.
- Gerçek ve tahmin fiyatlarının saçılım grafiğini gösterir.
- MAE, RMSE ve R² metriklerini konsolda yazdırır.

---

## Dosya Yapısı

```
main/
│
├── benznet.py
├── requirements.txt
├── data.xlsx
└── README.md
```

---

## Veri Seti Hakkında

`data.xlsx` dosyası, kullanılmış araç (tercihen Mercedes-Benz) verilerini içermelidir.  
Gerekli sütunlar:  
- `price` (tahmin edilecek hedef değer)
- Diğer özellikler (kategorik/sayısal, ör. yıl, kilometre, yakıt tipi, model vs.)

---

## Lisans

Bu proje açık kaynaklıdır ve akademik/kişisel kullanım için ücretsizdir.

---

## İletişim

Destek veya soru için issue açabilir veya proje sahibine ulaşabilirsiniz.
