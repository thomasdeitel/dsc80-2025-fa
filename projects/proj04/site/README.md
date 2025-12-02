# Cooking Time vs. Recipe Ratings

**Author:** Tommy Deitel  
**Course:** DSC 80 — Fall 2025

Food.com hosts 83,782 recipes and 731,927 user interactions spanning a decade. My project asks: **Do longer, more complex recipes actually earn higher ratings, or do home cooks reward speedy dishes?**

## Introduction
- Dataset: Recipes table with prep time, tags, nutrition stats + Interactions table with (optional) 1-5 ratings and text reviews.
- Question: Exploring whether time investment (total minutes) meaningfully changes user satisfaction.
- Key columns: `minutes`, `n_steps`, `n_ingredients`, `nutrition` vector, `rating`, and `review`.
- Each recipe row merges with aggregated review statistics so we can relate author-provided metadata with crowd responses.

## Data Cleaning and Exploratory Data Analysis
I created an analysis-ready table by
1. Expanding the stringified `nutrition` list into separate numeric columns (calories + macros) and parsing submission timestamps.
2. Treating `rating=0` as missing (Food.com writes zeros when users skip the stars) and counting review text length.
3. Aggregating interactions per recipe to compute average rating, number of ratings, median rating, share of missing ratings, and average review length.
4. Engineering helper features such as `minutes_total`, `log_minutes`, quick/slow flags, and bins for minutes and ingredient counts.

Head of the cleaned table:

| name |   minutes_total |   avg_rating |   review_count |   n_ingredients |   calories |   avg_review_length | is_quick   |
|:-------------------------------------|----------------:|-------------:|---------------:|----------------:|-----------:|--------------------:|:-----------|
| 1 brownies in the world    best ever |              40 |            4 |              1 |               9 |      138.4 |               254   | False      |
| 1 in canada chocolate chip cookies   |              45 |            5 |              1 |              11 |      595.1 |               336   | False      |
| 412 broccoli casserole               |              40 |            5 |              4 |               9 |      194.8 |               272   | False      |
| millionaire pound cake               |             120 |            5 |              1 |               7 |      878.3 |               196   | False      |
| 2000 meatloaf                        |              90 |            5 |              2 |              13 |      267   |               402.5 | False      |

### Univariate Analysis
The heavy right tail in prep minutes shows most dishes are quick weeknight meals; ratings, however, are tightly packed near 4.5-5.0.

<iframe src="assets/minutes_hist.html" width="800" height="500" frameborder="0"></iframe>

### Bivariate Analysis
A scatter with a trendline (restricted to recipes with ≥5 ratings) shows only a slight downward slope: longer recipes do *not* guarantee better ratings, though extremely long dishes appear more variable.

<iframe src="assets/minutes_vs_rating.html" width="800" height="500" frameborder="0"></iframe>

### Interesting Aggregates
Grouping by prep-time bins reveals quick dishes dominate the catalog and even have marginally *higher* mean ratings.

| minutes_bin   |   recipes |   avg_minutes |   avg_rating |   avg_reviews |
|:--------------|----------:|--------------:|-------------:|--------------:|
| <=30          |     37311 |       17.9    |      4.645   |       2.73    |
| 31-60         |     25415 |       45.7    |      4.607   |       2.59    |
| 61-120        |     12328 |       81.7    |      4.627   |       2.41    |
| 120+          |      8659 |      403.8    |      4.593   |       2.53    |

## Assessment of Missingness
- **NMAR reasoning:** The `rating` column is NMAR because reviewers can type a comment and submit without clicking any stars; failing to provide a rating depends on the unobserved intent of the reviewer, not just observed columns.
- Only 7.1% of interaction rows have blank ratings. Reviews with missing ratings are ~39 characters shorter on average (`p < 0.002`), suggesting quick comments omit the star widget. Prep minutes of the recipe do **not** explain missingness (`p = 0.116`).

<iframe src="assets/missingness_review_length.html" width="800" height="500" frameborder="0"></iframe>

## Hypothesis Testing
1. **Fast (≤30 min) vs. slow recipes.** Observed mean difference (fast - slow) = 0.040 stars; no permutation among 500 reshuffles matched this magnitude (`p < 0.002`). I reject H0 and conclude fast recipes receive slightly higher ratings.
2. **Simple (≤8 ingredients) vs. complex recipes.** Observed difference (simple - complex) = 0.020 stars with `p < 0.002`, pointing to a similar advantage for streamlined ingredient lists.

Both tests use the difference in mean rating as the statistic with `α = 0.05`.

## Framing a Prediction Problem
- **Task:** Binary classification — predict if a recipe’s eventual mean rating ≥ 4.5 using only metadata available when the author publishes the recipe.
- **Response:** `is_high_rating` (True iff avg rating ≥ 4.5) across 81,173 recipes; positives make up ~75%.
- **Features:** Minutes, steps, ingredient counts, and 7 nutrition metrics (10 quantitative features total).
- **Metric:** Prioritize F1 to balance precision and recall on the minority low-rated class while still reporting accuracy.

## Baseline Model
- **Model:** sklearn Pipeline → median-impute & standardize numeric features → LogisticRegression (`max_iter=1000`).
- **Performance:** `train_accuracy=0.751`, `test_accuracy=0.750`, `test_F1=0.857`, only slightly above the naïve baseline of always predicting "high rating".
- **Next steps:** Engineer domain-aware features (`log_minutes`, ingredient-per-step ratios, tag-derived flags) and run a grid search over class weights / `C`, while also trying tree-based models (e.g., RandomForest) that can capture non-linear interactions.

## Final Model
Planned upgrades (feature engineering + model selection) will be reported here for the final submission.

## Fairness Analysis
Coming soon: I plan to compare performance between quick weeknight dishes and multi-hour recipes using the final model’s F1 or recall.
