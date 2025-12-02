
# WSM — Weak Signal Miner for Emerging Risk Detection

WSM (Weak Signal Miner) is a full end-to-end framework for **detecting, scoring, and interpreting weak signals of emerging risks** in large scientific bodies (e.g., PubMed). It combines topic modeling, temporal analysis, recency-weighted magnitude scoring, and LLM-based interpretation to support early detection of hazards in regulatory science.

---

## 🔍 Key Features

### 1. Automated Literature Retrieval
- Query PubMed or custom corpora for risk-related search terms  
- Monthly or yearly aggregation  
- Transparent count tracking and result logging  

### 2. Semantic Topic Modeling
- BERTopic with configurable UMAP + HDBSCAN parameters  
- Designed to avoid over-segmentation  
- Domain-guided topic merging support  

### 3. Temporal Bucketing & Normalization
- Flexible time-bucket assignment (monthly, weekly, yearly)  
- Normalization for comparability across topics  
- Produces interpretable trend curves  

### 4. Topic Emergence Scoring
A core methodological component:
- Exponential recency weighting  
- Magnitude-based jump detection  
- Suddenness emphasis through relative change scoring  

### 5. Weak Signal Ranking
- Hard filtering by count bounds  
- Score thresholding  
- Even distribution across allowed count range  
- Returns top‑K curated weak signals  

### 6. LLM Weak Signal Interpreter (optional)
- LangChain + OpenAI-based  
- Generates:
  - Novelty score  
  - Severity score  
  - Narrative explanation  
- Appended directly to weak signal dataframe  

### 7. Visual Analytics
- Histograms (monthly, yearly, half-year)  
- Topic trend plots  
- Ranked weak-signal tables with full representation  

---

## 📁 Repository Structure

```
WSM/
│
├── WSM_final.ipynb           # Main workflow notebook
├── src/                      # Optional modular Python implementation
│   ├── retrieval.py
│   ├── topic_modeling.py
│   ├── scoring.py
│   ├── weak_signal_ranker.py
│   └── llm_interpreter.py
├── data/                     # Local caches or datasets
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone
```bash
git clone https://github.com/<yourname>/<yourrepo>.git
cd <yourrepo>
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set OpenAI Key
```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Run Notebook
Open `WSM_final.ipynb` in Jupyter or VS Code.

---

## ⚙️ Configuration Overview

### Topic Modeling
```python
UMAP_N_NEIGHBORS = 10
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = 'cosine'

HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = 'euclidean'
HDBSCAN_CLUSTER_SELECTION_METHOD = 'leaf'
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.1
```

### Recency & Magnitude
```python
HALF_LIFE = 1.0
magnitude_ALPHA = 1.5
magnitude_EPS = 1e-9
ALPHA = 1
```

### Weak Signal Ranking
```python
TOP_K_WEAK = 30
WEAK_MIN_COUNT = 10
WEAK_MAX_COUNT_QUANTILE = 0.8
WEAK_MIN_UPPER_COUNT = 30
SCORE_MIN_THRESHOLD = 0.004
```

---

## 📊 Workflows (ASCII)

### Temporal Bucketing
```
Raw Docs → Assign Time Buckets → Topic Counts → Normalization
```

### Recency-and-Magnitude Scoring
```
Topic Frequencies → Recency Weights → Magnitude Change → Final Score
```

### Weak Signal Ranking
```
Filter by Count → Score Threshold → Sort → Top-K Weak Signals
```

---

## 📘 Citation

Menning, A., Farkas, Z., NAME, K., Reale, F., van der Velden, B.  
*Topical over-segmentation and suddenness scoring leads to trustworthy Weak Signal Detection in Regulatory Risk Assessment.*  
Preprint.

---

## 🤝 Contributing

Contributions are welcome: add cases, improve scoring, or extend the notebook.

---

## 📄 License

MIT License (or your choice)
