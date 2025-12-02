
# WSM — Weak Signal Miner for Emerging Risk Detection

WSM (Weak Signal Miner) is a full end-to-end framework for **detecting, scoring, and interpreting weak signals of emerging risks** in large scientific bodies (e.g., PubMed). It combines topic modeling, temporal analysis, recency-weighted magnitude scoring, and LLM-based interpretation to support early detection of hazards in regulatory science.

---

## 🔍 Key Features
### 1. Micro Topic Modeling with BERTopic (https://maartengr.github.io/BERTopic)
- BERTopic with configurable UMAP + HDBSCAN parameters  
- Designed to avoid over-segmentation  

### 2. Temporal Bucketing & Normalization
- Flexible time-bucket assignment (monthly, weekly, yearly)  
- Normalization for comparability across topics  
- Produces interpretable trend curves  

### 3. Topic Emergence Scoring
A core methodological component:
- Exponential recency weighting  
- Magnitude-based jump detection  
- Suddenness emphasis through relative change scoring  

### 4. Top-k Weak Signal Ranking
- Hard filtering by count bounds  
- Score thresholding  
- Even distribution across allowed count range  
- Returns top‑K weak signals  

### 6. LLM Weak Signal Interpreter (coming soon)
- LangChain + OpenAI-based  
- Generates:
  - Novelty score  
  - Severity score  
  - Narrative explanation  
- Appended directly to weak signal dataframe  

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

## 🤝 Contributing

Contributions are welcome: add cases, improve scoring, or extend the notebook.

---

## 📄 License

CC BY 4.0 for methodology, MIT for code components
