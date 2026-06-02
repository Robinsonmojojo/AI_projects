import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# ── Load & Prepare Data ──
df = pd.read_csv('dataset.csv')
le_g = LabelEncoder()
le_c = LabelEncoder()
df['gender_enc']   = le_g.fit_transform(df['gender'])
df['category_enc'] = le_c.fit_transform(df['preferred_category'])

features = ['age','gender_enc','total_spend',
            'purchases_electronics','purchases_clothing',
            'purchases_groceries','purchases_books','purchases_sports']

X = df[features]
y = df['category_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

sc        = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

# ── Train Models ──
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
nb  = GaussianNB().fit(X_train_s, y_train)

# ── K-Means Clustering ──
X_scaled     = sc.transform(X)
km           = KMeans(n_clusters=3, random_state=42)
df['cluster'] = km.fit_predict(X_scaled)

# ── PCA for Scatter Plot ──
pca  = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# ── Build Dashboard ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Project 7 — AI Recommendation System Dashboard',
             fontsize=16, fontweight='bold')

# Plot 1: Preferred Category Distribution
ax1 = axes[0, 0]
df['preferred_category'].value_counts().plot(
    kind='bar', ax=ax1, color='steelblue', edgecolor='black')
ax1.set_title('Preferred Category Distribution')
ax1.set_xlabel('Category')
ax1.set_ylabel('Number of Customers')
ax1.tick_params(axis='x', rotation=30)

# Plot 2: Model Accuracy Comparison
ax2   = axes[0, 1]
from sklearn.metrics import accuracy_score
accs  = [accuracy_score(y_test, knn.predict(X_test_s)),
         accuracy_score(y_test, nb.predict(X_test_s))]
bars  = ax2.bar(['KNN', 'Naïve Bayes'], accs,
                color=['steelblue', 'coral'], edgecolor='black')
ax2.set_title('Model Accuracy Comparison')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
for bar, val in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + 0.01, f"{val:.1%}", ha='center', fontweight='bold')

# Plot 3: Customer Cluster Scatter Plot
ax3    = axes[1, 0]
colors = ['red', 'green', 'blue']
for i in range(3):
    mask = df['cluster'] == i
    ax3.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=colors[i], label=f'Cluster {i}', alpha=0.7)
ax3.set_title('Customer Segments (K-Means)')
ax3.set_xlabel('PCA Component 1')
ax3.set_ylabel('PCA Component 2')
ax3.legend()

# Plot 4: Spend Distribution by Cluster
ax4 = axes[1, 1]
for c in range(3):
    ax4.hist(df[df['cluster'] == c]['total_spend'],
             alpha=0.6, label=f'Cluster {c}', bins=15)
ax4.set_title('Spend Distribution by Cluster')
ax4.set_xlabel('Total Spend (KSh)')
ax4.set_ylabel('Frequency')
ax4.legend()

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/dashboard.png', dpi=150)
plt.show()
print("✅ Dashboard saved to outputs/dashboard.png")