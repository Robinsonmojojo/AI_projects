import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
from recommendation_engine import RecommendationEngine

# ── Page Config ──
st.set_page_config(
    page_title="ShopAI Recommender",
    page_icon="🛍️",
    layout="centered"
)

# ── Load Model Once ──
@st.cache_resource
def load_engine():
    engine = RecommendationEngine(model_type='knn')
    if os.path.exists('models/knn_model.pkl'):
        engine.load('knn')
    else:
        engine.fit('dataset.csv')
    return engine

engine = load_engine()

# ── Sidebar Navigation ──
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-bag.png", width=80)
st.sidebar.title("ShopAI")
st.sidebar.markdown("Customer Purchase Recommendation System")
page = st.sidebar.radio("Navigate", [
    "🎯 Get Recommendations",
    "📋 Customer Records",
    "📊 Dashboard",
    "ℹ️ About"
])

# ════════════════════════════════════════
# PAGE 1 — GET RECOMMENDATIONS
# ════════════════════════════════════════
if page == "🎯 Get Recommendations":
    st.title("🛍️ ShopAI Recommender")
    st.markdown("Fill in the customer details below to get personalised product recommendations.")

    with st.form("customer_form"):
        st.subheader("👤 Customer Details")

        col1, col2 = st.columns(2)
        with col1:
            name   = st.text_input("Customer Name", placeholder="e.g. Jane Wanjiku")
            age    = st.number_input("Age", min_value=18, max_value=80, value=25)
        with col2:
            gender = st.selectbox("Gender", ["M", "F"],
                                  format_func=lambda x: "Male" if x == "M" else "Female")
            spend  = st.number_input("Total Spend (KSh)", min_value=0, value=15000, step=500)

        st.subheader("🛒 Purchase History")
        st.caption("How many times has this customer bought from each category? (0–20)")

        col3, col4, col5 = st.columns(3)
        with col3:
            elec  = st.slider("⚡ Electronics", 0, 20, 5)
            cloth = st.slider("👕 Clothing",    0, 20, 3)
        with col4:
            groc  = st.slider("🛒 Groceries",   0, 20, 4)
            books = st.slider("📚 Books",        0, 20, 2)
        with col5:
            sports = st.slider("🏋️ Sports",     0, 20, 6)

        submitted = st.form_submit_button("🎯 Get Recommendations", use_container_width=True)

    if submitted:
        if not name:
            st.warning("⚠️ Please enter a customer name.")
        else:
            with st.spinner("Analysing customer profile..."):
                recs = engine.recommend(age, gender, spend,
                                        elec, cloth, groc, books, sports)

            st.success("✅ Recommendations Ready!")

            # Show recommendations
            st.subheader("🎯 Top 3 Recommended Categories")
            prices = {
                'electronics': 4500, 'clothing': 1800,
                'groceries': 950, 'books': 650, 'sports': 2200
            }
            icons = {
                'electronics': '⚡', 'clothing': '👕',
                'groceries': '🛒', 'books': '📚', 'sports': '🏋️'
            }
            labels = ["🥇 TOP PICK", "🥈 RECOMMENDED", "🥉 SUGGESTED"]

            subtotal = 0
            for i, rec in enumerate(recs):
                price     = prices.get(rec, 0)
                subtotal += price
                with st.container():
                    col_a, col_b, col_c = st.columns([1, 3, 1])
                    with col_a:
                        st.markdown(f"### {icons.get(rec,'•')}")
                    with col_b:
                        st.markdown(f"**{rec.capitalize()}**")
                        st.caption(labels[i])
                    with col_c:
                        st.markdown(f"**KSh {price:,}**")
                st.divider()

            vat   = subtotal * 0.16
            total = subtotal + vat

            # Receipt
            st.subheader("🧾 Receipt")
            now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            receipt_no = f"RCT-{datetime.now().strftime('%H%M%S')}"

            receipt = f"""
| Field        | Details                          |
|-------------|----------------------------------|
| Receipt No  | {receipt_no}                     |
| Date        | {now}                            |
| Customer    | {name}                           |
| Age         | {age} yrs                        |
| Gender      | {'Male' if gender == 'M' else 'Female'} |
| Spend Hist  | KSh {spend:,}                    |
| #1 Pick     | {recs[0].capitalize()} — KSh {prices.get(recs[0],0):,} |
| #2 Pick     | {recs[1].capitalize()} — KSh {prices.get(recs[1],0):,} |
| #3 Pick     | {recs[2].capitalize()} — KSh {prices.get(recs[2],0):,} |
| Subtotal    | KSh {subtotal:,}                 |
| VAT (16%)   | KSh {vat:,.2f}                   |
| **TOTAL**   | **KSh {total:,.2f}**             |
| AI Model    | KNN Classifier (K=5)             |
| Accuracy    | 82%                              |
            """
            st.markdown(receipt)
            st.download_button(
                label="⬇️ Download Receipt as Text",
                data=receipt,
                file_name=f"receipt_{receipt_no}.txt",
                mime="text/plain",
                use_container_width=True
            )

            # Save to database
            conn = sqlite3.connect('project7.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customer_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, age INTEGER, gender TEXT,
                    total_spend REAL,
                    recommendation1 TEXT,
                    recommendation2 TEXT,
                    recommendation3 TEXT,
                    query_date TEXT
                )
            ''')
            cursor.execute('''
                INSERT INTO customer_queries (
                    name, age, gender, total_spend,
                    recommendation1, recommendation2,
                    recommendation3, query_date
                ) VALUES (?,?,?,?,?,?,?,?)
            ''', (name, age, gender, spend,
                  recs[0], recs[1], recs[2], now))
            conn.commit()
            conn.close()
            st.info("💾 Customer record saved to database.")


# ════════════════════════════════════════
# PAGE 2 — CUSTOMER RECORDS
# ════════════════════════════════════════
elif page == "📋 Customer Records":
    st.title("📋 Customer Records")
    st.markdown("All customers who have used the recommendation system.")

    if not os.path.exists('project7.db'):
        st.warning("No records yet. Go to Get Recommendations first.")
    else:
        conn = sqlite3.connect('project7.db')
        df   = pd.read_sql('SELECT * FROM customer_queries', conn)
        conn.close()

        if df.empty:
            st.warning("No records found in the database.")
        else:
            st.success(f"✅ {len(df)} customer records found.")
            st.dataframe(df, use_container_width=True)

            # Download as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Records as CSV",
                data=csv,
                file_name="customer_records.csv",
                mime="text/csv",
                use_container_width=True
            )


# ════════════════════════════════════════
# PAGE 3 — DASHBOARD
# ════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 AI Model Dashboard")

    df = pd.read_csv('dataset.csv')

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", "100")
    col2.metric("KNN Accuracy",    "82%")
    col3.metric("Categories",      "5")

    st.subheader("Preferred Category Distribution")
    counts = df['preferred_category'].value_counts()
    st.bar_chart(counts)

    st.subheader("Age Distribution of Customers")
    fig, ax = plt.subplots()
    ax.hist(df['age'], bins=15, color='steelblue', edgecolor='black')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader("Spend Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['total_spend'], bins=15, color='coral', edgecolor='black')
    ax2.set_xlabel('Total Spend (KSh)')
    ax2.set_ylabel('Count')
    st.pyplot(fig2)

    if os.path.exists('outputs/dashboard.png'):
        st.subheader("Full ML Dashboard")
        st.image('outputs/dashboard.png', use_column_width=True)


# ════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About ShopAI")
    st.markdown("""
    ### 🛍️ ShopAI Recommendation System
    
    **Built for:** TVET Level 6 AI Capstone Project 7  
    **Developer:** Arnold  
    **Institution:** Kenya — CDACC/KNEC Curriculum  

    ---

    ### 🤖 How It Works
    1. Customer details are entered into the system
    2. KNN Classifier predicts the top 3 product categories
    3. K-Means Clustering segments the customer into a group
    4. A receipt is generated and the record is saved

    ---

    ### 📊 Models Used
    | Model | Accuracy |
    |-------|----------|
    | KNN Classifier (K=5) | 82% |
    | Naïve Bayes | 76% |
    | K-Means Clustering | 3 Segments |

    ---

    ### 💼 Business Value
    - Increases sales through personalised recommendations
    - Reduces marketing costs by targeting the right customers
    - Improves customer satisfaction and retention
    - Helps Kenyan e-commerce businesses compete globally
    """)