import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text_to_words(text_series):
    # Standard light cleaning similar to notebooks
    cleaned = text_series.str.lower()
    cleaned = cleaned.str.replace(r"[^\w\s-]", " ", regex=True) # remove punctuation except hyphens
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned

def get_word_shares(df, text_col, date_col, freq, min_doc_freq=10, max_doc_freq=1000):
    df = df.copy()
    df[text_col] = clean_text_to_words(df[text_col])
    
    # Define time buckets
    if freq == "H2":
        df["Bucket"] = (
            df[date_col].dt.year.astype(str)
            + "H"
            + np.where(df[date_col].dt.month <= 6, "1", "2")
        )
    else:
        df["Bucket"] = df[date_col].dt.to_period(freq).dt.to_timestamp()

    # Get total documents per bucket
    docs_per_bucket = df.groupby("Bucket").size()
    
    # Vocabulary and document index for each word
    word_doc_map = {}
    
    # Exclude standard English stopwords
    stopwords = set(list(ENGLISH_STOP_WORDS) + ["food", "feed", "diet", "dietary", "study", "results", "effect", "effects", "associated", "associated", "analysis", "group", "groups", "concentration", "concentrations", "levels", "level", "using", "use", "used", "high", "low", "different", "similar", "control", "controls", "supplement", "supplements", "significant", "significantly"])
    
    # We want to identify for each document in each bucket, which words it contains
    bucket_word_counts = []
    
    print("Tokenizing documents and counting words...")
    for idx, row in df.iterrows():
        bucket = row["Bucket"]
        text_val = str(row[text_col])
        words = set([w for w in text_val.split() if w.isalpha() and len(w) > 2 and w not in stopwords])
        for w in words:
            bucket_word_counts.append({"Bucket": bucket, "Word": w, "DocID": idx})
            
    word_df = pd.DataFrame(bucket_word_counts)
    
    # Filter words by total document count across the whole corpus
    word_totals = word_df.groupby("Word")["DocID"].nunique()
    valid_words = word_totals[(word_totals >= min_doc_freq) & (word_totals <= max_doc_freq)].index.tolist()
    
    word_df = word_df[word_df["Word"].isin(valid_words)]
    
    # Count of documents per (Bucket, Word)
    counts = word_df.groupby(["Bucket", "Word"]).size().unstack(fill_value=0)
    
    # Align buckets and calculate shares (document frequency / total documents in bucket)
    shares = counts.div(docs_per_bucket, axis=0).fillna(0.0)
    
    # Word totals across whole corpus
    counts_total = counts.sum(axis=0)
    
    return shares, counts_total

def rank_weak_signals_pfas(shares, counts_total):
    # Replicate PFAS scoring
    shares_sorted = shares.sort_index()
    n_buckets = len(shares_sorted)
    
    HALF_LIFE = 1.0
    MAG_ALPHA = 4.0
    RECENT_WINDOW = 2
    
    # Recency
    ages = np.arange(n_buckets - 1, -1, -1, dtype=float)
    w = 0.5 ** (ages / max(HALF_LIFE, 1e-6))
    w = w / w.sum()
    recency_score = (shares_sorted.mul(w, axis=0)).sum(axis=0)
    
    # Jump
    if n_buckets > RECENT_WINDOW:
        recent_mean = shares_sorted.tail(RECENT_WINDOW).mean(axis=0)
        past_mean = shares_sorted.iloc[:-RECENT_WINDOW].mean(axis=0)
    else:
        recent_mean = shares_sorted.tail(1).iloc[0]
        past_mean = pd.Series(0.0, index=shares_sorted.columns)
        
    jump_strength = recent_mean - past_mean
    
    # Penalty
    lifetime = (shares_sorted > 0).sum(axis=0).astype(float)
    penalty = 1.0 / np.log1p(lifetime)
    penalty.replace([np.inf, -np.inf], 0.0, inplace=True)
    penalty = penalty.fillna(0.0)
    
    # Combined score
    recency_jump_score = (recency_score + MAG_ALPHA * jump_strength) * penalty
    recency_jump_score = recency_jump_score.fillna(0.0)
    
    # Weak signal ranking parameters
    TOP_K_WEAK = 10
    WEAK_MIN_COUNT = 10
    WEAK_MIN_UPPER_COUNT = 30
    SCORE_MIN_THRESHOLD = 0.004
    
    df_rank = pd.DataFrame({
        "score": recency_jump_score,
        "Count": counts_total
    }).reset_index()
    
    df_rank = df_rank.rename(columns={"Word": "Name"})
    
    # Filters
    candidates = df_rank[
        (df_rank["Count"] >= WEAK_MIN_COUNT) &
        (df_rank["Count"] <= WEAK_MIN_UPPER_COUNT) &
        (df_rank["score"] >= SCORE_MIN_THRESHOLD)
    ].copy()
    
    print(f"PFAS Keyword Candidates count: {len(candidates)}")
    if candidates.empty:
        return pd.DataFrame(), df_rank
        
    c_min, c_max = candidates["Count"].min(), candidates["Count"].max()
    if c_max > c_min:
        count_norm = 1.0 - (candidates["Count"] - c_min) / (c_max - c_min)
    else:
        count_norm = pd.Series(1.0, index=candidates.index)
        
    s = candidates["score"]
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        score_norm = (s - s_min) / (s_max - s_min)
    else:
        score_norm = pd.Series(1.0, index=candidates.index)
        
    BETA = 0.5
    candidates["weak_signal_score"] = BETA * score_norm + (1.0 - BETA) * count_norm
    
    # Round-robin
    def round_robin_by_count(pool, k):
        if pool.empty or k <= 0:
            return []
        groups = {}
        for c, sub in pool.groupby("Count"):
            sub_sorted = sub.sort_values("weak_signal_score", ascending=False)
            groups[c] = list(sub_sorted.index)
        selected = []
        pointers = {c: 0 for c in groups}
        while len(selected) < k:
            any_picked = False
            for c in sorted(groups):
                idx_list = groups[c]
                pos = pointers[c]
                if pos < len(idx_list):
                    selected.append(idx_list[pos])
                    pointers[c] = pos + 1
                    any_picked = True
                    if len(selected) >= k:
                        break
            if not any_picked:
                break
        return selected

    selected_idx = round_robin_by_count(candidates, TOP_K_WEAK)
    ws_top = candidates.loc[selected_idx].copy()
    ws_top = ws_top.sort_values("weak_signal_score", ascending=False).reset_index(drop=True)
    
    return ws_top, df_rank

def rank_weak_signals_ryr(shares, counts_total):
    # Replicate RYR scoring
    shares_sorted = shares.sort_index()
    
    HALF_LIFE = 1.0
    JUMP_EPS = 1e-5
    ALPHA = 0.5
    
    # Recency
    ages = np.arange(len(shares_sorted))[::-1]
    weights = pd.Series(0.5 ** (ages / HALF_LIFE), index=shares_sorted.index)
    weights = weights / weights.sum()
    recency_score = (shares_sorted.T @ weights).rename("recency_score")
    
    # Jump
    last_share = shares_sorted.iloc[-1]
    if len(shares_sorted) > 1:
        prev_share = shares_sorted.iloc[-2]
    else:
        prev_share = 0.0 * last_share
    rel_jump = (last_share - prev_share) / (prev_share + JUMP_EPS)
    jump_strength = rel_jump.clip(lower=0.0).rename("jump_strength")
    
    # Normalize components
    r = recency_score.clip(lower=0.0)
    j = jump_strength.clip(lower=0.0)
    
    r_norm = r / r.max() if r.max() > 0 else r
    j_norm = j / j.max() if j.max() > 0 else j
    
    recency_jump_score = (ALPHA * r_norm + (1 - ALPHA) * j_norm).rename("recency_jump_score")
    
    # Weak signal ranking parameters
    TOP_K_WEAK = 10
    WEAK_MIN_COUNT = 10
    WEAK_MIN_UPPER_COUNT = 30
    SCORE_MIN_THRESHOLD = 0.004
    
    df_rank = pd.DataFrame({
        "score": recency_jump_score,
        "Count": counts_total
    }).reset_index()
    
    df_rank = df_rank.rename(columns={"Word": "Name"})
    
    # Filters
    candidates = df_rank[
        (df_rank["Count"] >= WEAK_MIN_COUNT) &
        (df_rank["Count"] <= WEAK_MIN_UPPER_COUNT) &
        (df_rank["score"] >= SCORE_MIN_THRESHOLD)
    ].copy()
    
    print(f"RYR Keyword Candidates count: {len(candidates)}")
    if candidates.empty:
        return pd.DataFrame(), df_rank
        
    c_min, c_max = candidates["Count"].min(), candidates["Count"].max()
    if c_max > c_min:
        count_norm = 1.0 - (candidates["Count"] - c_min) / (c_max - c_min)
    else:
        count_norm = pd.Series(1.0, index=candidates.index)
        
    s = candidates["score"]
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        score_norm = (s - s_min) / (s_max - s_min)
    else:
        score_norm = pd.Series(1.0, index=candidates.index)
        
    BETA = 0.5
    candidates["weak_signal_score"] = BETA * score_norm + (1.0 - BETA) * count_norm
    
    # Round-robin
    def round_robin_by_count(pool, k):
        if pool.empty or k <= 0:
            return []
        groups = {}
        for c, sub in pool.groupby("Count"):
            sub_sorted = sub.sort_values("weak_signal_score", ascending=False)
            groups[c] = list(sub_sorted.index)
        selected = []
        pointers = {c: 0 for c in groups}
        while len(selected) < k:
            any_picked = False
            for c in sorted(groups):
                idx_list = groups[c]
                pos = pointers[c]
                if pos < len(idx_list):
                    selected.append(idx_list[pos])
                    pointers[c] = pos + 1
                    any_picked = True
                    if len(selected) >= k:
                        break
            if not any_picked:
                break
        return selected

    selected_idx = round_robin_by_count(candidates, TOP_K_WEAK)
    ws_top = candidates.loc[selected_idx].copy()
    ws_top = ws_top.sort_values("weak_signal_score", ascending=False).reset_index(drop=True)
    
    return ws_top, df_rank

# --- PFAS Execution ---
print("=== PFAS Dataset ===")
pfas_df = pd.read_csv("c:\\Users\\axelm\\Documents\\Github\\WSM_paper\\data\\pfas_in_food.csv")
pfas_df["Date"] = pd.to_datetime(pfas_df["Date"], errors="coerce")
pfas_df = pfas_df.dropna(subset=["Abstract", "Date"])
pfas_df["Year"] = pfas_df["Date"].dt.year
pfas_df = pfas_df[(pfas_df["Year"] >= 2004) & (pfas_df["Year"] <= 2006)].reset_index(drop=True)

shares_pfas, counts_pfas = get_word_shares(pfas_df, "Abstract", "Date", "H2")
top_pfas, all_pfas = rank_weak_signals_pfas(shares_pfas, counts_pfas)

print("\nTop 10 PFAS Keywords as Weak Signals:")
print(top_pfas)

# Check target keywords position in the all list
targets_pfas = ["pfos", "pfoa", "perfluorooctane", "perfluorooctanesulfonic"]
print("\nTarget keywords stats in all PFAS keywords list:")
target_rows_pfas = all_pfas[all_pfas["Name"].isin(targets_pfas)].copy()
# Sort all_pfas by score descending to find raw rank before count bounds
all_pfas_sorted = all_pfas.sort_values("score", ascending=False).reset_index(drop=True)
for t in targets_pfas:
    match = all_pfas_sorted[all_pfas_sorted["Name"] == t]
    if not match.empty:
        rank = match.index[0]
        score = match["score"].values[0]
        cnt = match["Count"].values[0]
        print(f"Keyword '{t}': Raw Rank = {rank}, Count = {cnt}, Score = {score:.6f}")
    else:
        print(f"Keyword '{t}' not found in vocabulary.")

# --- RYR Execution ---
print("\n=== RYR Dataset ===")
ryr_df = pd.read_csv("c:\\Users\\axelm\\Documents\\Github\\WSM_paper\\data\\ryr.csv")
ryr_df["Date"] = pd.to_datetime(ryr_df["Date"], errors="coerce")
ryr_df = ryr_df.dropna(subset=["Abstract", "Date"])
ryr_df["Year"] = ryr_df["Date"].dt.year
ryr_df = ryr_df[(ryr_df["Year"] >= 2010) & (ryr_df["Year"] <= 2017)].reset_index(drop=True)

shares_ryr, counts_ryr = get_word_shares(ryr_df, "Abstract", "Date", "M")
top_ryr, all_ryr = rank_weak_signals_ryr(shares_ryr, counts_ryr)

print("\nTop 10 RYR Keywords as Weak Signals:")
print(top_ryr)

targets_ryr = ["citrinin", "ryr", "monacolin", "lovastatin"]
print("\nTarget keywords stats in all RYR keywords list:")
all_ryr_sorted = all_ryr.sort_values("score", ascending=False).reset_index(drop=True)
for t in targets_ryr:
    match = all_ryr_sorted[all_ryr_sorted["Name"] == t]
    if not match.empty:
        rank = match.index[0]
        score = match["score"].values[0]
        cnt = match["Count"].values[0]
        print(f"Keyword '{t}': Raw Rank = {rank}, Count = {cnt}, Score = {score:.6f}")
    else:
        print(f"Keyword '{t}' not found in vocabulary.")
