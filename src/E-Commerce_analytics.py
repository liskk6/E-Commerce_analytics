import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# MLxtend imports
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# --- CONFIGURATION ---
DATA_PATH = '../data/ECommerce.csv'
DB_PATH = '../database/ECommerce.db'

# --- SQL QUERIES ---
QUERIES = {
    "useful_data": """
        SELECT *
        FROM Sells
        WHERE UnitPrice > 0 AND Quantity > 0 AND CustomerID IS NOT NULL
    """,
    "monthly_sales": """
        SELECT 
            SUM(UnitPrice * Quantity) as sells, 
            SUBSTR(InvoiceDate, 1 , INSTR(InvoiceDate, '/') -1) as month, 
            SUBSTR( SUBSTR( InvoiceDate, INSTR(InvoiceDate, '/') +1) , INSTR(SUBSTR(InvoiceDate ,INSTR(InvoiceDate, '/') + 1) , '/') +1  ,4  ) as year,
            COUNT(DISTINCT CustomerID) as clients
        FROM Sells
        WHERE UnitPrice > 0 AND Quantity > 0 AND CustomerID IS NOT NULL
        GROUP BY year, month
        ORDER BY year, month
    """,
    "client_activity": """
        SELECT CustomerID, COUNT(DISTINCT InvoiceNo) as frequency, SUM(Quantity * UnitPrice) as monetary
        FROM Sells
        WHERE UnitPrice > 0 AND Quantity > 0 AND CustomerID IS NOT NULL
        GROUP BY CustomerID 
        ORDER BY monetary, frequency
    """,
    "items_per_invoice": """
        SELECT InvoiceNo as InvNo, STRING_AGG(StockCode, ',') as items
        FROM Sells
        WHERE UnitPrice > 0 AND Quantity > 0 AND CustomerID IS NOT NULL
        GROUP BY InvoiceNo 
    """
}


# --- FUNCTIONS ---

def load_csv_to_db(csv_path, db_path):
    """
    Reads the CSV file and loads it into a SQLite database.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"Loading data from {csv_path}...")
    df_csv = pd.read_csv(csv_path, encoding="latin1")

    # Establish a temporary connection just to dump the data
    with sqlite3.connect(db_path) as conn:
        df_csv.to_sql('Sells', conn, if_exists='replace', index=False)
        print(f"Data successfully loaded into {db_path} (Table: Sells)")

    return df_csv


def analyze_monthly_sales(conn):
    """
    Retrieves monthly sales data, processes dates, and plots Sales vs Clients.
    """
    print("Analyzing monthly sales...")

    # Load data from DB
    df_sells = pd.read_sql(QUERIES["monthly_sales"], conn)

    # Construct a date string for plotting
    df_sells['date'] = df_sells['month'] + '/' + df_sells['year']

    # Convert types for sorting
    df_sells['year'] = df_sells['year'].astype('int')
    df_sells['month'] = df_sells['month'].astype('int')

    # Sort and remove the last entry (often incomplete data)
    df_sells = df_sells.sort_values(by=['year', 'month']).iloc[:-1]

    # --- Visualization ---
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))

    # Plot Sales (Left Axis)
    sns.lineplot(data=df_sells, x='date', y='sells', color='blue', ax=ax1, alpha=0.55, marker='o', legend=False)
    ax1.set_xlabel('Date (Month/Year)', color='black', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Sells', color='blue', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    ax1.set_title('Clients vs Sells per Month')

    # Plot Clients (Right Axis)
    ax2 = ax1.twinx()
    sns.lineplot(data=df_sells, x='date', y='clients', color='red', ax=ax2, label='Clients', alpha=0.55, legend=False, marker='s' )
    ax2.set_ylabel('Clients', color='red', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    # plt.show() # Commented out to prevent blocking execution if running all at once


def perform_customer_clustering(conn):
    """
    Performs RFM analysis and KMeans clustering on customers.
    """
    print("Performing customer clustering...")

    # 1. Get aggregated metrics (Frequency, Monetary)
    df_activities = pd.read_sql(QUERIES["client_activity"], conn)

    # 2. Get raw data to calculate Recency (Time since last purchase)
    df_raw = pd.read_sql(QUERIES["useful_data"], conn)
    df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])

    df_dates = df_raw.groupby('CustomerID', as_index=False).agg(
        first_sell=('InvoiceDate', 'min'),
        last_sell=('InvoiceDate', 'max')
    )

    # Calculate days active and days inactive
    last_date_activity = df_raw['InvoiceDate'].max()
    df_dates['days_activities'] = (df_dates['last_sell'] - df_dates['first_sell']).dt.days
    df_dates['days_inactive'] = (last_date_activity - df_dates['last_sell']).dt.days

    # Merge metrics
    df_combined = pd.merge(df_activities, df_dates, on='CustomerID', how='left')
    df_combined = df_combined.sort_values(by='days_activities')

    # Ensure types are correct
    df_combined['frequency'] = df_combined['frequency'].astype('int')
    df_combined['monetary'] = df_combined['monetary'].astype('int')

    # 3. Standardization
    scaler = StandardScaler()
    cols_to_normalize = ['days_activities', 'frequency', 'monetary', 'days_inactive']
    X_scaled = scaler.fit_transform(df_combined[cols_to_normalize])

    # 4. KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=15, n_init=10)
    df_combined['Cluster'] = kmeans.fit_predict(X_scaled)

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), dpi=100)

    # Cluster Plot: Frequency vs Monetary
    sns.scatterplot(data=df_combined, x='frequency', y='monetary', hue='Cluster', palette='plasma', ax=axes[0])
    axes[0].set_xlabel('Monetary', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=16, fontweight='bold')

    # Cluster Plot: Inactive vs Active Days
    sns.scatterplot(data=df_combined, x='days_inactive', y='days_activities', hue='Cluster', palette='rocket',
                    ax=axes[1])
    axes[1].set_xlabel('Days Inactive', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Days Active', fontsize=16, fontweight='bold')

    # Print mean stats per cluster
    print("\nCluster Averages:")
    print(df_combined.groupby("Cluster")[cols_to_normalize].mean())

    plt.tight_layout()
    plt.show()


def analyze_association_rules(conn):
    """
    Performs Market Basket Analysis using FP-Growth.
    """

    # Load transaction items
    # Note: Ensure your SQLite version supports STRING_AGG, otherwise use GROUP_CONCAT
    df_items = pd.read_sql(QUERIES["items_per_invoice"], conn)

    # Prepare data for MLxtend
    transactions = df_items['items'].apply(lambda x: x.split(',')).to_list()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # FP-Growth Algorithm
    frequent_itemsets = fpgrowth(df_encoded, min_support=0.01, use_colnames=True)

    # Generate Rules
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.01)
    rules = rules.sort_values('confidence', ascending=False)

    cols_to_show = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print("\nTop Association Rules:")
    print(rules[cols_to_show].head(10))  # Print top 10 for cleaner output


def main():
    # 1. Prepare Database
    try:
        load_csv_to_db(DATA_PATH, DB_PATH)
    except Exception as e:
        print(f"Data loading skipped or failed: {e}")
        # We continue because the DB might already exist

    # 2. Open Persistent Connection for Analysis
    with sqlite3.connect(DB_PATH) as conn:

        # 3. Run Analysis Modules
        analyze_monthly_sales(conn)

        perform_customer_clustering(conn)

        analyze_association_rules(conn)

        # Explicit commit if needed (usually only for writes)
        conn.commit()
        print("\nAnalysis complete.")


if __name__ == "__main__":
    main()