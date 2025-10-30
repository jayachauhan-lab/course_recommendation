
# Course Recommendation  Demo

This is a  course recommender system that uses **content-based filtering** with **TF-IDF vectorization** and **sigmoid kernel similarity** to suggest similar courses based on a single keyword. It ranks recommendations and immediately reports evaluation metrics like **MAP**, **nDCG@K**, and **MRR**.

---

##  Features

- Keyword-driven course recommendation
- Content-based filtering using course skill descriptions
- Top-K ranked suggestions with metadata:
  - Course Name
  - Difficulty Level
  - Course Rating
  - University
- Evaluation metrics computed on-the-fly:

  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (nDCG@K)
  - Mean Reciprocal Rank (MRR)

---

##  AI Method

- **Method:** Content-based recommendation
- **Vectorization:** TF-IDF (`scikit-learn`)
- **Similarity:** Sigmoid kernel
- **Evaluation:** Single-query ranking metrics

---

## Requirements

Install dependencies using pip:

```bash
pip install pandas numpy scikit-learn tqdm
```

---

## Dataset Format

The recommender expects a CSV file named `Coursera.csv` in the current directory with the following columns:

- `Course Name`
- `Skills`
- `Difficulty Level`
- `Course Rating`
- `University`

Optional columns like `Course URL` and `Course Description` will be dropped automatically.

---

## How to Run

```bash
python recommend_console_singlekeyword_immediate_metrics_with_university.py
```

- If `Coursera.csv` is not found, you’ll be prompted to enter its path.
- Enter a single keyword (e.g., `Python`) when prompted.
- The system will match the keyword to a course title and recommend similar courses.

---

## Example Output

```
Input keyword: Python
Matched title: Introduction to Python Programming

Top-5 recommendations:
  1. Python for Data Science | Sim: 0.8721 | Pearson r: 0.6543
     University: XYZ | Difficulty Level: Intermediate | Course Rating: 4.7
     Skills excerpt: data analysis, numpy, pandas...

Per-query metrics:
  MAP: 0.8123
  nDCG@5: 0.7945
  MRR: 1.0000
```

---

## valuation Notes

Metrics are computed using a binary relevance vector where the matched course title is marked as relevant. This enables single-query evaluation without needing user feedback.

---

## Notes

- Only the **first token** of the keyword is used for matching.
- Matching is done via exact word match, substring match, or fuzzy title similarity.
- Recommendations are based on skill similarity, not user behavior.

---

## Customization

You can adjust:
- `K` (number of recommendations)
- TF-IDF parameters: `min_df`, `ngram_range`
- Similarity metric (e.g., cosine similarity)

---


