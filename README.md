
# Smart Product Pricing Multimodal ML Model

**multimodal machine learning system** that predicts e-commerce product prices using **textual** (product titles, descriptions) and **visual** (product images) data.  
It combines **NLP-based feature extraction**, **CNN image embeddings**, and a **gradient-boosted regression model** for pricing predictions.

---

## Features

- **Multimodal Fusion:** Combines text and image features for richer representations.  
- **Text Feature Engineering:** Extracts brand, quantity, size, and category cues using regex-based parsing and hashing vectorization.  
- **Image Understanding:** Uses **MobileNetV2** pretrained on ImageNet to generate image embeddings.  
- **Optimized Training:** Employs **LightGBM** with cross-validation and log-transform targets for stable regression.  
- **Caching Support:** Stores precomputed text and image embeddings for faster retraining.  
- **Custom Evaluation Metric:** Uses **SMAPE (Symmetric Mean Absolute Percentage Error)** for fair price accuracy evaluation.

---

## Model Architecture

```text
Product Title/Text → OptimizedTextFeatureExtractor → Hashing + Numeric + Category Features
Product Image → CNNImageFeatureExtractor → MobileNetV2 Embeddings
↓
Concatenation (Text + Image)
↓
LightGBM Regressor
↓
Predicted Price (USD)
````

---

## 📁 Project Structure

```
├── smart_pricing_multimodal.py     # Main pipeline script
├── dataset/
│   ├── train.csv                   # Training data
│   ├── test.csv                    # Test data
│   ├── images/                     # Image folder
│   └── cache_text_1k/              # Cached text features
│   └── cache_cnn_1k/               # Cached image features
└── sample_test.csv                 # Sample data file (for demo)
```

---

## 🧪 Example Dataset Format

| sample_id | catalog_content               | image_link           | price |
| --------- | ----------------------------- | -------------------- | ----- |
| 1         | "Organic Green Tea 100g Pack" | images/green_tea.jpg | 5.99  |
| 2         | "Premium Dark Chocolate 250g" | images/dark_choc.jpg | 8.49  |
| ...       | ...                           | ...                  | ...   |

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SaarthChahal/ML-price-predictor.git
cd ML-price-predictor
pip install -r requirements.txt
```

### Main Dependencies

* Python ≥ 3.9
* TensorFlow ≥ 2.10
* scikit-learn ≥ 1.4
* LightGBM ≥ 4.0
* NumPy, Pandas, TQDM, Pillow, SciPy

---

## Usage

### Run Training and Prediction

```bash
python smart_pricing_multimodal.py
```

This will:

1. Load and preprocess the dataset.
2. Extract text and image features.
3. Train the multimodal model using 3-fold cross-validation.
4. Generate predictions and save them to:

   ```
   dataset/predictions.csv
   ```

---

## Technical Details

| Component            | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **Text Encoder**     | Uses `HashingVectorizer` for tokenized features + regex-based numeric and categorical cues. |
| **Image Encoder**    | Extracts 1280-dimensional features using `MobileNetV2`’s global average pooling layer.      |
| **Regressor**        | `LightGBMRegressor` with log-transformed targets for price prediction.                      |
| **Metric**           | SMAPE (Symmetric Mean Absolute Percentage Error).                                           |
| **Cross-validation** | 3-fold `KFold` split for model stability.                                                   |

---

## Output

The script produces a CSV file with predicted prices:

```csv
sample_id,price
1,5.83
2,8.11
3,4.29
...
```

---

## Customization

You can modify these in `smart_pricing_multimodal.py`:

* `n_splits` → Number of cross-validation folds.
* `max_hash_features` → Dimensionality of text feature space.
* `image_dir` → Directory containing product images.
* `cache_text_dir` / `cache_cnn_dir` → Paths for saving cached embeddings.

---

## 🤝 Acknowledgments

* [TensorFlow MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
* [scikit-learn](https://scikit-learn.org/)
