# License: Apache-2.0
# Smart Product Pricing Challenge — PRODUCTION-READY MODEL
# Auto-uses 10K samples for speed, with emergency diagnostics
import os
import re
import sys
import warnings
import pickle
from pathlib import Path
from typing import Dict
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 #type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #type: ignore
from tensorflow.keras.utils import load_img, img_to_array #type: ignore

def debug_print(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

debug_print("="*60, "START")
debug_print("="*60, "START")

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# SMAPE metric
def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    out = np.zeros_like(denom)
    out[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(out) * 100.0)

# Patterns
_QUANTITY_PATTERNS = [
    (r"(\d+(?:\.\d+)?)\s*(kg|kilogram|kilograms)\b", 1000.0, "g"),
    (r"(\d+(?:\.\d+)?)\s*(g|gram|grams)\b", 1.0, "g"),
    (r"(\d+(?:\.\d+)?)\s*(lb|pound|pounds)\b", 453.592, "g"),
    (r"(\d+(?:\.\d+)?)\s*(oz|ounce|ounces)\b", 28.3495, "g"),
    (r"(\d+(?:\.\d+)?)\s*(l|liter|litre|liters|litres)\b", 1000.0, "ml"),
    (r"(\d+(?:\.\d+)?)\s*(ml|milliliter|millilitre)\b", 1.0, "ml"),
    (r"(\d+(?:\.\d+)?)\s*(fl\s*oz)\b", 29.5735, "ml")]
_MULTIPACK_PATTERNS = [r"(\d+)[- ]pack\b", r"\bpack\s+of\s+(\d+)", r"(\d+)\s*count\b", r"(\d+)[- ]ct\b"]
_SIZE_INDICATORS = {'mini': 0.5, 'small': 0.7, 'large': 1.5, 'xl': 2.0, 'family': 2.0, 'bulk': 4.0}
_QUALITY_TIERS = {'premium': ['premium', 'gourmet', 'luxury'], 'organic': ['organic', 'natural'], 'budget': ['value', 'economy']}
_CATEGORIES = {'food_snack': ['snack', 'cookie', 'chocolate'], 'beverage': ['coffee', 'tea', 'juice'],
    'personal_care': ['shampoo', 'soap', 'lotion'], 'supplement': ['vitamin', 'protein']}
_BRAND_HINT = re.compile(r"\b(by|from)\s+([A-Z][a-z]{2,})")

class OptimizedTextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_hash_features=2**14, cache_file=None):  # Reduced to 2^14 for speed
        self.max_hash_features = max_hash_features
        self.cache_file = cache_file
        self.hash_vec = HashingVectorizer(n_features=max_hash_features, alternate_sign=False, norm=None,
            lowercase=True, analyzer="word", ngram_range=(1,2), stop_words="english")  # Reduced to bigrams
        self.brand_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.category_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.top_brands = None
        self.brand_stats = None

    @staticmethod
    def _normalize_text(s):
        return re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip()

    def _extract_multipack_count(self, s):
        for pattern in _MULTIPACK_PATTERNS:
            m = re.search(pattern, s.lower())
            if m and 1 < float(m.group(1)) <= 100:
                return float(m.group(1))
        return 1.0

    def _extract_size_multiplier(self, s):
        for size, mult in _SIZE_INDICATORS.items():
            if size in s.lower():
                return mult
        return 1.0

    def _detect_category(self, s):
        s_l = s.lower()
        for category, keywords in _CATEGORIES.items():
            if any(kw in s_l for kw in keywords):
                return category
        return "other"

    def _extract_brand(self, s):
        m = _BRAND_HINT.search(s)
        if m:
            return m.group(2).lower()
        for word in s.split()[:10]:
            if word and len(word) > 2 and word[0].isupper():
                return word.lower()
        return "unknown"

    def _extract_numeric_features(self, s):
        s_l = s.lower()
        qty_g = qty_ml = 0.0
        for pat, factor, unit in _QUANTITY_PATTERNS:
            for m in re.finditer(pat, s_l):
                val = float(m.group(1))
                if unit == "g": qty_g += val * factor
                elif unit == "ml": qty_ml += val * factor
        pack_count = self._extract_multipack_count(s)
        size_mult = self._extract_size_multiplier(s)
        brand = self._extract_brand(s)
        quality_feats = {f'quality_{tier}': float(any(kw in s_l for kw in keywords)) for tier, keywords in _QUALITY_TIERS.items()}
        return {
            "qty_g": qty_g, "qty_ml": qty_ml, "pack_count": pack_count, "size_multiplier": size_mult,
            "total_weight_g": pack_count * qty_g * size_mult, "has_quantity": float(qty_g > 0 or qty_ml > 0),
            "has_known_brand": float(brand != "unknown"), "brand": brand, "category": self._detect_category(s),
            **quality_feats}

    def fit(self, X, y=None, prices=None):
        debug_print(f"Fitting on {len(X)} samples")
        X = X.fillna("").astype(str)
        brand_counts, brands_list, categories_list = Counter(), [], []
        for s in tqdm(X, desc="Extracting features"):
            feats = self._extract_numeric_features(self._normalize_text(s))
            brand_counts[feats["brand"]] += 1
            brands_list.append(feats["brand"])
            categories_list.append(feats["category"])
        self.top_brands = {brand for brand, count in brand_counts.items() if count >= 10}
        debug_print(f"Found {len(self.top_brands)} brands")
        if prices is not None:
            self.brand_stats = pd.DataFrame({'brand': brands_list, 'price': prices}).groupby('brand')['price'].agg(['mean', 'std']).fillna(0)
        brands_mapped = [b if b in self.top_brands else "other" for b in brands_list]
        self.brand_encoder.fit(np.array(brands_mapped).reshape(-1, 1))
        self.category_encoder.fit(np.array(categories_list).reshape(-1, 1))
        return self

    def transform(self, X):
        X = X.fillna("").astype(str)
        if self.cache_file and Path(self.cache_file).exists():
            debug_print(f"Loading cache: {self.cache_file}", "CACHE")
            X_all = sparse.load_npz(self.cache_file)
            if X_all.shape[0] == len(X):
                return X_all
        text_norm = [self._normalize_text(s) for s in tqdm(X, desc="Normalizing")]
        X_hash = self.hash_vec.transform(text_norm)
        num_cols = ["qty_g", "qty_ml", "pack_count", "size_multiplier", "total_weight_g", "has_quantity", "has_known_brand"]
        data_num, brands, categories, brand_price_feats = [], [], [], []
        for s in tqdm(text_norm, desc="Parsing"):
            f = self._extract_numeric_features(s)
            data_num.append([f[c] for c in num_cols])
            brand = f["brand"] if f["brand"] in self.top_brands else "other"
            brands.append(brand)
            categories.append(f["category"])
            if self.brand_stats is not None and f["brand"] in self.brand_stats.index:
                stats = self.brand_stats.loc[f["brand"]]
                brand_price_feats.append([stats['mean'], stats['std']])
            else:
                brand_price_feats.append([0.0, 0.0])
        X_num = np.array(data_num, dtype=float)
        X_brand_price = np.array(brand_price_feats, dtype=float)
        X_brand = self.brand_encoder.transform(np.array(brands).reshape(-1, 1))
        X_category = self.category_encoder.transform(np.array(categories).reshape(-1, 1))
        X_all = sparse.hstack([X_hash, X_brand, X_category, X_num, X_brand_price], format="csr")
        debug_print(f"Features: {X_all.shape}")
        if self.cache_file:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(self.cache_file, X_all)
        return X_all

class CNNImageFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, image_root, cache_file=None):
        self.image_root = Path(image_root)
        self.cache_file = cache_file
        self.cached_features_ = None
        self.cnn_model = None
        
        if cache_file and Path(cache_file).exists():
            with open(cache_file, 'rb') as f:
                self.cached_features_ = pickle.load(f)
            debug_print(f"Loaded {len(self.cached_features_)} CNN features", "CACHE")
        else:
            debug_print("Loading MobileNetV2...")
            self.cnn_model = MobileNetV2(
                weights='imagenet', 
                include_top=False, 
                pooling='avg', 
                input_shape=(224, 224, 3)
            )

    def _extract_cnn_features(self, img_path):
        """Extract CNN features with PROPER preprocessing"""
        try:
            # Load image
            img = load_img(img_path, target_size=(224, 224))
            
            # Convert to array
            x = img_to_array(img)
            
            # Add batch dimension
            x = np.expand_dims(x, axis=0)
            
            # CRITICAL: Preprocess for MobileNetV2
            x = preprocess_input(x)
            
            # Extract features
            features = self.cnn_model.predict(x, verbose=0)
            
            # Flatten to 1D
            features_flat = features.flatten()
            
            # Sanity check
            if np.max(np.abs(features_flat)) > 1000:
                debug_print(f"WARNING: Extreme CNN values detected: {np.max(features_flat)}", "WARN")
                return np.zeros(1280, dtype=np.float32)
            
            return features_flat.astype(np.float32)
            
        except Exception as e:
            debug_print(f"CNN extraction failed: {str(e)}", "WARN")
            return np.zeros(1280, dtype=np.float32)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.cached_features_ is not None:
            n = min(len(X), len(self.cached_features_))
            return sparse.csr_matrix(self.cached_features_[:n])
        
        X = X.fillna("").astype(str)
        feat_rows = []
        missing = 0
        
        for link in tqdm(X, desc="CNN"):
            if isinstance(link, str) and link:
                img_path = self.image_root / Path(link).name
                
                if img_path.exists():
                    feats = self._extract_cnn_features(img_path)
                else:
                    feats = np.zeros(1280, dtype=np.float32)
                    missing += 1
            else:
                feats = np.zeros(1280, dtype=np.float32)
                missing += 1
            
            feat_rows.append(feats)
        
        if missing > 0:
            debug_print(f"Missing images: {missing}/{len(X)}", "WARN")
        
        X_cnn = np.array(feat_rows, dtype=np.float32)
        
        # SANITY CHECK
        debug_print(f"CNN features range: {X_cnn.min():.3f} to {X_cnn.max():.3f}")
        if X_cnn.max() > 1000:
            debug_print("ERROR: CNN features corrupted! Range too large!", "ERROR")
        
        if self.cache_file:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(X_cnn, f)
            debug_print(f"Cached CNN features to {self.cache_file}", "CACHE")
        
        return sparse.csr_matrix(X_cnn)

class OptimizedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=500,  # Increased back
            learning_rate=0.05,
            num_leaves=127,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            verbose=-1,
            n_jobs=-1)
    
    def fit(self, X, y):
        debug_print(f"Training on {X.shape}")
        y = np.asarray(y, dtype=float)
        # LOG TRANSFORM - properly this time
        self.model.fit(X, np.log1p(y))  # log1p handles small values better
        return self
    
    def predict(self, X):
        y_log_pred = self.model.predict(X)
        y_pred = np.expm1(y_log_pred)  # Inverse of log1p
        return np.maximum(y_pred, 0.01)

def train_and_predict(train_df, test_df, image_dir, text_cache_dir=None, cnn_cache_dir=None, n_splits=3):  # Reduced to 3 folds
    from sklearn.model_selection import KFold  # Simple KFold for speed
    
    # Cache paths
    tr_text_cache = te_text_cache = tr_cnn_cache = te_cnn_cache = None
    if text_cache_dir:
        text_cache_dir = Path(text_cache_dir)
        text_cache_dir.mkdir(parents=True, exist_ok=True)
        tr_text_cache = str(text_cache_dir / "train_text.npz")
        te_text_cache = str(text_cache_dir / "test_text.npz")
    if cnn_cache_dir:
        cnn_cache_dir = Path(cnn_cache_dir)
        cnn_cache_dir.mkdir(parents=True, exist_ok=True)
        tr_cnn_cache = str(cnn_cache_dir / "train_cnn.pkl")
        te_cnn_cache = str(cnn_cache_dir / "test_cnn.pkl")
    
    # Features
    debug_print("\n--- TEXT FEATURES ---")
    txt = OptimizedTextFeatureExtractor(cache_file=tr_text_cache)
    txt.fit(train_df["catalog_content"], prices=train_df["price"])
    X_txt_tr = txt.transform(train_df["catalog_content"])
    txt.cache_file = te_text_cache
    X_txt_te = txt.transform(test_df["catalog_content"])
    
    debug_print("\n--- CNN FEATURES ---")
    imgx_train = CNNImageFeatureExtractor(image_root=image_dir, cache_file=tr_cnn_cache)
    X_img_tr = imgx_train.transform(train_df["image_link"])
    imgx_test = CNNImageFeatureExtractor(image_root=image_dir, cache_file=te_cnn_cache)
    X_img_te = imgx_test.transform(test_df["image_link"])
    
    # Fuse & convert to dense
    debug_print("\n--- FUSION & CONVERSION ---")
    X_tr = sparse.hstack([X_txt_tr, X_img_tr], format="csr").toarray()
    X_te = sparse.hstack([X_txt_te, X_img_te], format="csr").toarray()
    debug_print(f"Dense: train={X_tr.shape}, test={X_te.shape}")
    
    # FEATURE DIAGNOSTICS
    debug_print("\n=== FEATURE CHECK ===", "DIAG")
    debug_print(f"NaN: {np.isnan(X_tr).any()}, Inf: {np.isinf(X_tr).any()}")
    debug_print(f"Range: {X_tr.min():.3f} to {X_tr.max():.3f}")
    
    y = train_df["price"].astype(float).values
    debug_print(f"Price: ${y.min():.2f}-${y.max():.2f} (mean: ${y.mean():.2f})")
    
    # Train
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof, preds = np.zeros(len(train_df)), np.zeros(len(test_df))
    
    for fold, (tr, va) in enumerate(kf.split(X_tr), 1):
        debug_print(f"\n--- Fold {fold}/{n_splits} ---")
        model = OptimizedRegressor()
        model.fit(X_tr[tr], y[tr])
        va_pred = model.predict(X_tr[va])
        oof[va] = va_pred
        preds += model.predict(X_te) / n_splits
        debug_print(f"Fold {fold} SMAPE: {smape(y[va], va_pred):.3f}%, Pred: ${va_pred.min():.2f}-${va_pred.max():.2f}")
    
    debug_print(f"\nOVERALL SMAPE: {smape(y, oof):.3f}%", "RESULT")
    return np.maximum(preds, 0.01)

def main():
    data_dir = Path("dataset")
    images_dir = Path("dataset/images")
    
    debug_print("Loading datasets...")
    train_df = pd.read_csv(data_dir / "train.csv").head(75000)
    test_df = pd.read_csv(data_dir / "test.csv").head(75000)
    debug_print(f"Using {len(train_df)} train, {len(test_df)} test samples")
    
    preds = train_and_predict(train_df, test_df, images_dir, 
                              "dataset/cache_text_1k", 
                              "dataset/cache_cnn_1k",  # Separate cache!
                              n_splits=3)
    
    out = pd.DataFrame({"sample_id": test_df["sample_id"], "price": preds})
    out.to_csv("dataset/predictions_1k.csv", index=False)
    debug_print(f"✓ Saved predictions", "DONE")

if __name__ == "__main__":
    main()
