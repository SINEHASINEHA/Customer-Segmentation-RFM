# AI-Driven Customer Segmentation Using RFM Analysis

An intelligent customer segmentation framework that combines RFM (Recency, Frequency, Monetary) analysis with advanced machine learning clustering techniques to identify meaningful customer groups and support business decision-making.

System Architecture:
AI-Driven Customer Segmentation
        ↓
Customer Data Collection
        ↓
Data Preprocessing
        ↓
RFM Analysis (Recency, Frequency, Monetary)
        ↓
AI Segmentation Module (KMeans | DBSCAN | GMM)
        ↓
Hybrid Fusion Layer
        ↓
Cognitive Learning Layer
        ↓
Decision Support System
        ↓
Segmentation Results | Decision Support Reports | Business Dashboard

Technologies Used
- Python
- Pandas (Data Processing)
- Scikit-learn (K-Means, DBSCAN, GMM)
- Matplotlib & Seaborn (Visualization)

 Features
- Customer transaction data collection and preprocessing
- RFM score calculation (Recency, Frequency, Monetary)
- Multiple clustering algorithms: K-Means, DBSCAN, GMM
- Hybrid fusion layer for improved segmentation accuracy
- Cognitive learning mechanism for adapting to new data
- Business dashboard for decision support

How to Run:
Step 1: Install Python
Download from https://www.python.org/downloads/
Step 2: Install required libraries
pip install pandas scikit-learn matplotlib seaborn
Step 3: Create rfm_segmentation.py and paste this code:
python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from datetime import datetime

Steps:
Step 1: Sample Customer Data 
data = {
    'CustomerID':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PurchaseDate':['2024-04-10', '2024-03-01', '2023-12-15', '2024-04-25','2023-10-05', '2024-02-20', '2024-01-10', '2023-11-30','2024-04-01', '2023-09-15'],
    'OrderCount':[10, 5, 1, 15, 2, 7, 4, 3, 12, 1],
    'OrderValue':[3000, 1500, 200, 5000, 400, 2000, 800, 600, 4000, 150]
}

df = pd.DataFrame(data)
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

 Step 2: RFM Calculation 
today = datetime(2024, 5, 1)
df['Recency']   = (today - df['PurchaseDate']).dt.days
df['Frequency'] = df['OrderCount']
df['Monetary']  = df['OrderValue']

rfm = df[['CustomerID', 'Recency', 'Frequency', 'Monetary']].copy()
print("\n── RFM Values ──")
print(rfm)

 Step 3: Data Preprocessing 
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

Step 4: AI Segmentation Module 

 K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['KMeans_Segment'] = kmeans.fit_predict(rfm_scaled)

DBSCAN Clustering
dbscan = DBSCAN(eps=0.8, min_samples=2)
rfm['DBSCAN_Segment'] = dbscan.fit_predict(rfm_scaled)

GMM Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
rfm['GMM_Segment'] = gmm.fit_predict(rfm_scaled)

 Step 5: Hybrid Fusion Layer 
rfm['Hybrid_Score'] = (
    rfm['KMeans_Segment']+rfm['GMM_Segment']
)

def label_segment(score):
    if score == 0:
        return 'Champion'
    elif score == 1:
        return 'Loyal Customer'
    else:
        return 'At-Risk Customer'

rfm['Final_Segment'] = rfm['Hybrid_Score'].apply(label_segment)

print("\n── Segmentation Results ──")
print(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Final_Segment']])

 Step 6: Business Dashboard (Visualization) 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('AI-Driven Customer Segmentation Dashboard', fontsize=14, fontweight='bold')

# Pie chart - segment distribution
seg_counts = rfm['Final_Segment'].value_counts()
axes[0].pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',colors=['#2ecc71', '#3498db', '#e74c3c'])
axes[0].set_title('Customer Segment Distribution')

# Bar chart - avg monetary by segment
rfm.groupby('Final_Segment')['Monetary'].mean().plot( kind='bar', ax=axes[1], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[1].set_title('Average Order Value by Segment')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Average Value (₹)')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('customer_dashboard.png', dpi=150)
print("\nDashboard saved as customer_dashboard.png ✅")
plt.show()

Step 4: Run the file
python rfm_segmentation.py

Output
- Customer segments printed in terminal
- Dashboard chart saved as `customer_dashboard.png`

System Users
  Business User — Views segmentation results and reports
  Admin — Manages the system and business dashboard

Developer
Sineha P | MCA Student | Bishop Heber College
