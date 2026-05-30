Customer Lifecycle Value Dashboard
A three-layer customer analytics system that segments customers by behaviour, clusters them with machine learning, and predicts how much each one will spend over the next 12 months.

What this does
Most customer analysis stops at "who bought what." This project goes further — it tries to answer: which customers are worth investing in, which ones are slipping away, and what is each customer likely to spend in the next year?
To answer that, three different approaches are stacked on top of each other, each one catching something the previous one misses:
Layer 1 — RFM Segmentation scores every customer on three dimensions: how recently they bought (Recency), how often they buy (Frequency), and how much they've spent in total (Monetary). These scores get combined into named segments like Champions, Loyal Customers, At Risk, and Lost. It's a heuristic — fast, interpretable, and useful for business teams who want plain English labels.
Layer 2 — KMeans Clustering ignores the hand-crafted rules and just finds natural groupings in the data. It uses an elbow curve to pick the optimal number of clusters, then assigns every customer to a cluster purely based on their RFM values. Sometimes the clusters line up with the heuristic segments, sometimes they don't — and the mismatches are often the most interesting customers.
Layer 3 — BG-NBD and Gamma-Gamma models are probabilistic. The BG-NBD model estimates the probability that a customer is still active (hasn't churned) and predicts how many purchases they'll make in the next 12 months. The Gamma-Gamma model then takes those predicted purchase counts and estimates the average spend per transaction. Multiply them together and you get a predicted 12-month CLV for every single customer.
All three layers feed into a single final_customer_table.csv, which powers a Streamlit dashboard and a Power BI report.

What the dashboard shows
There are four tabs:
Executive Overview — top-level KPIs (total customers, total historical revenue, projected 12-month CLV pipeline, average probability of being alive), segment distribution bar chart, CLV tier donut chart, and cluster distribution.
RFM & Cluster Analysis — segment-level aggregations for Recency, Frequency, and Monetary, plus a scatter plot of Recency vs Frequency with bubble size representing spend and colour representing cluster — the fastest way to visually spot your high-value outliers.
BG-NBD & Gamma-Gamma Insights — historical spend vs predicted 12-month CLV scatter, average predicted CLV by RFM segment, and distributions of P(Alive) and predicted purchases. This is where you see who's likely to churn and who's about to spend.
Customer Profile Lookup — enter any Customer ID and get their full profile: RFM segment, ML cluster, CLV tier, historical metrics, BG-NBD projections, and a side-by-side comparison against the store average.

How to run it
bash# Install dependencies
pip install -r requirements.txt

# Make sure final_customer_table.csv is in the same directory
# Then launch the dashboard
streamlit run clv.py
For the Power BI report, open finalcgt.pbix in Power BI Desktop. The data source points to final_customer_table.csv — update the path if needed.

Project structure
Customer-Lifecycle-Value-Dashboard-proj/
├── clv.py                        # Streamlit dashboard (all four tabs)
├── final_customer_table.csv      # Processed output: RFM + cluster + CLV per customer
├── finalcgt.pbix                 # Power BI dashboard
├── requirements.txt
├── optimal_k.png                 # Elbow curve used to select number of KMeans clusters
├── cluster_heatmap.png           # RFM heatmap by cluster (k=3)
├── cluster_heatmap_k4.png        # RFM heatmap by cluster (k=4)
├── cluster_scatter.png           # Cluster scatter plot (k=3)
├── cluster_scatter_k4.png        # Cluster scatter plot (k=4)
├── cluster_business_value.png    # Business value breakdown by cluster
└── distributions_comparison.png  # RFM distribution comparisons

Stack

Python — pandas, scikit-learn, lifetimes, plotly, streamlit
Power BI — executive dashboard (.pbix)
BG-NBD model — purchase frequency and churn probability (lifetimes library)
Gamma-Gamma model — average transaction value prediction (lifetimes library)


Background
The lifetimes library makes the BG-NBD and Gamma-Gamma models fairly straightforward to fit, but the interesting part was stacking all three approaches and seeing where they agree and disagree. A customer the RFM rules label as "At Risk" might land in a high-value KMeans cluster, and their BG-NBD P(Alive) might still be 85%. That tension between the three models is exactly the kind of thing a dashboard like this is built to surface.
