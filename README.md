
# WSM — Weak Signal Miner for Emerging Risk Detection

WSM (Weak Signal Miner) is a full end-to-end framework for **detecting, scoring, and interpreting weak signals of emerging risks** in large scientific bodies (e.g., PubMed). It combines topic modeling, temporal analysis, recency-weighted magnitude scoring, and LLM-based interpretation to support early detection of hazards in regulatory science.

---

## 📊 Workflow Overview
WSM follows a structured pipeline that transforms raw scientific text into ranked weak signals of emerging risk. The process integrates micro-topic modelling, temporal normalisation, emergence scoring, and final weak-signal selection.

### 1. Micro Topic Modeling
WSM identifies fine-grained thematic units (“micro-topics”) using BERTopic (https://maartengr.github.io/BERTopic)
```
Raw Documents
      ↓
Sentence Transformer (embeddings)
      ↓
UMAP (dimensionality reduction)
      ↓
HDBSCAN (density-based clustering)
      ↓
Micro-Topics with representative terms
```
### 2. Temporal Bucketing & Normalization
Each document retains its publication date and is mapped to a temporal bucket (daily, weekly, monthly, etc.). The system automatically selects the finest granularity with sufficient data density.
```
Micro-Topics + Dates
      ↓
Automatic (or forced) bucket selection
      ↓
Assign documents to time buckets
      ↓
Count topic occurrences per bucket (nₖ,ₜ)
      ↓
Compute topic shares per bucket
      ↓
Topic-by-time dataframe
```
Using shares instead of raw counts ensures robustness to fluctuations in overall publication volume.

### 3. Topic Emergence Scoring
Emergence is detected through a combination of recency (persistence in recent buckets) and magnitude (sudden short-term increase):

```
Topic-by-time dataframe
      ↓
Exponential recency weighting (half-life h)
      ↓
Recency score (recent sustained attention)
      ↓
Magnitude score (positive jump in last interval)
      ↓
Normalize both components to [0,1]
      ↓
Final topic emergence score
```
This scoring highlights topics that are both currently active and accelerating.

### 4. Top-k Weak Signal Ranking
Finally, topics are filtered and ranked to identify the most relevant weak signals: 
```
Topics + emergence scores
      ↓
Count bounds & score threshold
      ↓
Sort by emergence score via "round robin method"
      ↓
Select Top-K weak signals
```
These weak signals represent early indicators of emerging scientific developments that merit further expert analysis.

### 6. LLM Weak Signal Interpreter
- LangChain + OpenAI OR Gemini 
- Generates:
  - Topic label
  - Novelty score  
  - Severity score  
  - Novelty reasoning
  - Severity reasoning
- Appended to weak signal dataframe

---

## 🤝 Contributing

Contributions are welcome: add cases, improve scoring, or extend the notebook.

---

## 📄 License

CC BY 4.0 for methodology, MIT for code components
