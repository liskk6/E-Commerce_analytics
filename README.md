# E-Commerce Customer Segmentation & Product Association Analysis

## About this project
This is a personal project built on a public e-commerce transactions dataset (~540,000 records). I wanted to practice two things that come up a lot in real analytics work: grouping customers by behavior (RFM segmentation) and finding which products tend to sell together (market basket analysis).

## What I was trying to answer
1. **Which customers matter most, and which are slipping away?** — group customers into segments (loyal, at-risk, new) so marketing effort can be targeted instead of blasted at everyone.
2. **What products go together?** — find item associations that could support a "customers also bought" type recommendation.

## Tech stack
- **Storage:** SQLite — loaded the raw CSV into a local database instead of keeping everything in memory, mainly to practice writing SQL against a larger dataset.
- **Data manipulation:** Python, Pandas, NumPy
- **Modeling:** scikit-learn (K-Means), mlxtend (FP-Growth / association rules)
- **Visualization:** Seaborn, Matplotlib

## What I did

### 1. ETL into SQLite
Loaded the ~500K raw rows into SQLite and did cleaning, type casting, and initial aggregation with SQL queries rather than pure Pandas — wanted the practice writing SQL against something bigger than a toy table.

### 2. RFM + K-Means segmentation
Calculated Recency, Frequency, and Monetary value per customer, scaled the features with `StandardScaler` (K-Means is distance-based, so unscaled monetary values would have dominated), and clustered customers into 4 groups.

A couple of the clusters were easy to interpret:
- One group had high frequency and recent activity — clear loyalty-program candidates.
- Another had high historical spend but long inactivity — worth a re-engagement campaign before they're gone for good.

### 3. Market basket analysis
Used FP-Growth to mine frequent itemsets and generate association rules, evaluated with lift and confidence, to see which products tend to be bought together.

## Visualizations
`figure_kmeans.png` — customer clusters plotted by frequency vs. monetary value
`figure_sells_clients.png` — monthly sales volume and active client growth over time

## What I'd do differently next time
Try DBSCAN or hierarchical clustering as a comparison to K-Means, since K-Means forces a fixed number of clusters and I picked 4 somewhat arbitrarily using the elbow method. Would also like to validate the association rules against a holdout period instead of the full dataset.
