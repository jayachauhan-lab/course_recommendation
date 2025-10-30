# recommend_console_singlekeyword_immediate_metrics_with_university.py
# Console-only recommender using TF-IDF + sigmoid kernel.
# Prompts the user for CSV path if default not found, asks for a single keyword (first token only),
# returns top-K recommendations including Difficulty Level, Course Rating, and University,
# and immediately reports Hit@1, MAP, nDCG@K, and MRR computed on the generated output (single-query evaluation).
# Requirements: pandas numpy scikit-learn tqdm

import os
import math
import difflib
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score

# Configuration
RANDOM_SEED = None
K = 5

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def build_model(df, text_col='Skills', min_df=3, ngram_range=(1, 3)):
    df = df.copy()
    df['cleaned'] = df[text_col].fillna('')
    tfv = TfidfVectorizer(min_df=min_df,
                          strip_accents='unicode',
                          analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=ngram_range,
                          stop_words='english')
    tfv_matrix = tfv.fit_transform(df['cleaned'])
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    indices = pd.Series(df.index, index=df['Course Name']).drop_duplicates()
    return {
        'df': df,
        'tfv_matrix': tfv_matrix,
        'sig': sig,
        'indices': indices,
        'vectorizer': tfv
    }


def _ensure_single_index(raw_idx):
    if isinstance(raw_idx, (np.ndarray, pd.Series, list)):
        return int(raw_idx[0])
    return int(raw_idx)


def pearson_correlation(a, b):
    if a.size == 0 or b.size == 0:
        return 0.0
    a_mean = a.mean()
    b_mean = b.mean()
    a_dev = a - a_mean
    b_dev = b - b_mean
    num = np.dot(a_dev, b_dev)
    denom = math.sqrt(np.dot(a_dev, a_dev) * np.dot(b_dev, b_dev))
    if denom == 0:
        return 0.0
    return float(num / denom)


def _find_best_title_by_keyword(keyword, course_names):
    kw = keyword.lower().strip()
    if not kw:
        return None

    direct_matches = []
    for name in course_names:
        name_l = name.lower()
        words = name_l.split()
        if kw in words:
            direct_matches.append((name, words.count(kw), len(words)))

    if direct_matches:
        direct_matches.sort(key=lambda x: (-x[1], x[2]))
        return direct_matches[0][0]

    substr_matches = [name for name in course_names if kw in name.lower()]
    if substr_matches:
        substr_matches.sort(key=lambda x: len(x))
        return substr_matches[0]

    close = difflib.get_close_matches(kw, course_names, n=1)
    return close[0] if close else None


def recommend_for_keyword(keyword, model, df_org, topk=K):
    """
    Returns:
      recs: DataFrame of top-k recommended items
      matched_title: matched course title used for ranking
      matched_info: tuple (matched_index, full_scores) where full_scores is score vector across all candidates
      per_item_metrics: list of dicts for recommended items with score and pearson_r
    """
    indices = model['indices']
    sig = model['sig']
    tfv_matrix = model['tfv_matrix']
    course_names = indices.index.tolist()

    if not keyword or not str(keyword).strip():
        return pd.DataFrame(), None, None, []

    token = str(keyword).strip().split()[0]
    matched_title = _find_best_title_by_keyword(token, course_names)
    if not matched_title:
        return pd.DataFrame(), None, None, []

    raw_idx = indices[matched_title]
    idx = _ensure_single_index(raw_idx)
    sim_row = np.asarray(sig[idx]).ravel()
    full_scores = sim_row.astype(np.float32)

    sig_scores = list(enumerate(sim_row))
    sig_scores = sorted(sig_scores, key=lambda x: float(x[1]), reverse=True)
    sig_scores = [s for s in sig_scores if s[0] != idx][:topk]
    course_indices = [i[0] for i in sig_scores]
    recs = df_org.iloc[course_indices].reset_index(drop=True)

    q_vec = tfv_matrix[idx].toarray().ravel()
    per_item_metrics = []
    for (rec_idx, score) in sig_scores:
        rec_vec = tfv_matrix[rec_idx].toarray().ravel()
        corr = pearson_correlation(q_vec, rec_vec)
        per_item_metrics.append({
            'df_index': int(rec_idx),
            'score': float(score),
            'pearson_r': corr
        })

    return recs, matched_title, (idx, full_scores), per_item_metrics


def mean_reciprocal_rank_singlequery(rel, scores):
    ranking = np.argsort(-scores)
    ranks = np.where(rel[ranking] == 1)[0]
    return 0.0 if ranks.size == 0 else 1.0 / (ranks[0] + 1)


def compute_single_query_metrics(df_full, true_title, scores, k=K):
    """
    Compute Hit@1, MAP, nDCG@k, MRR for a single query using the provided full-score vector.
    true_title: the canonical true course title string to mark as relevant (binary relevance)
    """
    # true_title may be a single string or a list/iterable of strings.
    if isinstance(true_title, (list, tuple, set, np.ndarray)):
        matches = df_full['Course Name'].isin(list(true_title)).astype(int).values
    else:
        matches = (df_full['Course Name'] == true_title).astype(int).values

    if matches.sum() == 0:
        rel = np.zeros(len(df_full), dtype=int)
    else:
        rel = matches.astype(int)

    Y_true = rel.reshape(1, -1)
    Y_score = scores.reshape(1, -1)

    try:
        map_score = float(label_ranking_average_precision_score(Y_true, Y_score))
    except Exception:
        map_score = 0.0

    try:
        ndcg = float(ndcg_score(Y_true, Y_score, k=k))
    except Exception:
        ndcg = 0.0

    mrr = float(mean_reciprocal_rank_singlequery(rel, scores))

    top1_idx = int(np.argmax(scores))
    top1_title = df_full.iloc[top1_idx]['Course Name']
    hit1 = (top1_title == true_title)
    hit1_pct = 100.0 * int(hit1)

    metrics = {
        'hit1': bool(hit1),
        'hit1_pct': hit1_pct,
        'map': map_score,
        'ndcg': ndcg,
        'mrr': mrr
    }
    return metrics, rel


def print_recommendation_output(keyword, original_title, matched_title, rec_df, per_item_metrics, metrics):
    print(f"\nInput keyword (single token used): {keyword}")
    # Note: original dataset titles matching the keyword are available but not shown here.
    print(f"Matched title used for similarity: {matched_title}")
    print(f"\nTop-{K} recommendations:\n")
    if rec_df is None or rec_df.empty:
        print("  (no recommendations found)")
    else:
        for i, row in rec_df.head(K).iterrows():
            name = row.get('Course Name', '<no name>')
            skills = (row.get('Skills', '') or '')[:400].replace('\n', ' ')
            difficulty = row.get('Difficulty Level', '(unknown)')
            rating = row.get('Course Rating', '(unknown)')
            university = row.get('University', '(unknown)')
            try:
                item_metrics = per_item_metrics[i]
                score = item_metrics['score']
                pearson_r = item_metrics['pearson_r']
            except Exception:
                score = float('nan')
                pearson_r = float('nan')
            print(f" {i+1}. {name} | Sim: {score:.4f} | Pearson r: {pearson_r:.4f}")
            print(f"    University: {university} | Difficulty Level: {difficulty} | Course Rating: {rating}")
            print(f"    Skills excerpt: {skills}\n")

    print(f"Predicted (top-1): {rec_df.iloc[0]['Course Name'] if (rec_df is not None and not rec_df.empty) else '(no prediction)'}")
    print("\nPer-query metrics computed on generated output:")
    # Ranking / precision / evaluation metrics (accuracy fields omitted)
    # Ranking / precision metrics
    if 'precision_at_k' in metrics:
        print(f"  Precision@{K}: {metrics.get('precision_at_k', 0.0):.4f}")
        print(f"  Recall@{K}: {metrics.get('recall_at_k', 0.0):.4f}")
        print(f"  F1@{K}: {metrics.get('f1_at_k', 0.0):.4f}")
    print(f"  MAP: {metrics.get('map', 0.0):.4f}")
    print(f"  nDCG@{K}: {metrics.get('ndcg', 0.0):.4f}")
    print(f"  MRR: {metrics.get('mrr', 0.0):.4f}")
    print("\nComment: MAP, nDCG, and MRR were computed by building a full candidate score vector for the matched query\nand comparing it to a binary relevance vector where the chosen true course title is marked as relevant.")


def prompt_for_csv(default_name='Coursera.csv'):
    csv_path = default_name
    if os.path.exists(csv_path):
        return csv_path
    print(f"Default file '{default_name}' not found in current directory: {os.getcwd()}")
    user_path = input("Enter full path to Coursera.csv (or press Enter to cancel): ").strip()
    if not user_path:
        raise FileNotFoundError(f"{default_name} not found in current directory: {os.getcwd()}")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"CSV file not found: {user_path}")
    return user_path


def main():
    csv_path = prompt_for_csv('Coursera.csv')

    df_org_full = pd.read_csv(csv_path)
    df = df_org_full.copy()
    # Keep University, Difficulty Level and Course Rating for output; drop other optional columns if present
    drop_cols = ['Course URL', 'Course Description']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
    if 'Course Name' not in df.columns or 'Skills' not in df.columns:
        raise ValueError("Coursera.csv must contain 'Course Name' and 'Skills' columns.")

    print("Building model...")
    model = build_model(df, text_col='Skills')

    raw = input("Enter a single keyword to search (only the first token will be used): ").strip()
    if not raw:
        print("No input provided; exiting.")
        return
    keyword = raw.split()[0]

    course_names = model['indices'].index.tolist()
    # Collect all course names that contain the keyword as a separate word
    original_titles = []
    for name in course_names:
        words = name.lower().split()
        if keyword.lower() in words:
            original_titles.append(name)

    rec_df, matched_title, matched_info, per_item_metrics = recommend_for_keyword(keyword, model, df_org_full, topk=K)
    if matched_info is None:
        print("No matching title found for that keyword; no recommendations available.")
        return

    matched_idx, full_scores = matched_info
    # Use the list of original titles if any were found; otherwise fall back to the matched title
    true_label = original_titles if len(original_titles) > 0 else matched_title
    metrics, rel_vec = compute_single_query_metrics(df_org_full, true_label, full_scores, k=K)

    print_recommendation_output(keyword, original_titles, matched_title, rec_df, per_item_metrics, metrics)


if __name__ == '__main__':
    main()
