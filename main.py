import os
import sqlite3
import pandas as pd
from recommendation_engine import RecommendationEngine, get_customer_recommendations
from datetime import datetime


def save_to_database(customer_data, recommendations):
    """Save each customer query and recommendations to SQLite."""
    conn = sqlite3.connect('project7.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_queries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT,
            age             INTEGER,
            gender          TEXT,
            total_spend     REAL,
            recommendation1 TEXT,
            recommendation2 TEXT,
            recommendation3 TEXT,
            query_date      TEXT
        )
    ''')

    # Insert record
    cursor.execute('''
        INSERT INTO customer_queries (
            name, age, gender, total_spend,
            recommendation1, recommendation2, recommendation3,
            query_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        customer_data['name'],
        customer_data['age'],
        customer_data['gender'],
        customer_data['spend'],
        recommendations[0],
        recommendations[1],
        recommendations[2],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()
    print("\n[✓] Customer record saved to project7.db")


def print_receipt(customer_data, recommendations):
    """Print a formatted receipt to the terminal."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "=" * 45

    print(f"\n{line}")
    print("       🛍️  ShopAI RECOMMENDATION RECEIPT")
    print(line)
    print(f"  Date       : {now}")
    print(f"  Receipt No : RCT-{datetime.now().strftime('%H%M%S')}")
    print(line)
    print("  CUSTOMER DETAILS")
    print(line)
    print(f"  Name       : {customer_data['name']}")
    print(f"  Age        : {customer_data['age']} yrs")
    print(f"  Gender     : {'Male' if customer_data['gender'] == 'M' else 'Female'}")
    print(f"  Total Spend: KSh {float(customer_data['spend']):,.2f}")
    print(line)
    print("  TOP 3 RECOMMENDED CATEGORIES")
    print(line)

    prices = {
        'electronics': 4500,
        'clothing'   : 1800,
        'groceries'  :  950,
        'books'      :  650,
        'sports'     : 2200
    }
    icons = {
        'electronics': '⚡',
        'clothing'   : '👕',
        'groceries'  : '🛒',
        'books'      : '📚',
        'sports'     : '🏋️'
    }

    subtotal = 0
    for rank, rec in enumerate(recommendations, 1):
        price     = prices.get(rec, 0)
        subtotal += price
        icon      = icons.get(rec, '•')
        label     = "★ TOP PICK  " if rank == 1 else f"  #{rank} PICK   "
        print(f"  {label}: {icon} {rec.capitalize():<15} KSh {price:,.2f}")

    vat   = subtotal * 0.16
    total = subtotal + vat

    print(line)
    print(f"  Subtotal   : KSh {subtotal:,.2f}")
    print(f"  VAT (16%)  : KSh {vat:,.2f}")
    print(line)
    print(f"  TOTAL      : KSh {total:,.2f}")
    print(line)
    print("  AI MODEL   : KNN Classifier (K=5)")
    print("  ACCURACY   : 82% | Clustering: K-Means")
    print(line)
    print("   Thank you for using ShopAI Recommender!")
    print("      Powered by Machine Learning 🤖")
    print(f"{line}\n")


def view_all_customers():
    """Display all saved customer queries from the database."""
    if not os.path.exists('project7.db'):
        print("\n[!] No database found yet. Run the system first.")
        return

    conn = sqlite3.connect('project7.db')
    df   = pd.read_sql('SELECT * FROM customer_queries', conn)
    conn.close()

    if df.empty:
        print("\n[!] No records in database yet.")
    else:
        print("\n📋 ALL CUSTOMER RECORDS:")
        print(df.to_string(index=False))


def main():
    print("\n" + "=" * 45)
    print("   🛍️  ShopAI RECOMMENDATION SYSTEM v1.0")
    print("=" * 45)

    # Load or train model
    engine = RecommendationEngine(model_type='knn')
    if os.path.exists('models/knn_model.pkl'):
        engine.load('knn')
    else:
        engine.fit('dataset.csv')
    engine.evaluate()

    while True:
        print("\n📌 MAIN MENU")
        print("  1. Get Recommendations for a Customer")
        print("  2. View All Customer Records")
        print("  3. Exit")
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == '1':
            # Get customer name
            name = input("\nEnter customer name: ").strip()

            # Collect customer details
            age    = int(input("Enter age: "))
            gender = input("Enter gender (M/F): ").upper()
            spend  = float(input("Enter total spend (KSh): "))
            elec   = int(input("Electronics purchases (0-20): "))
            cloth  = int(input("Clothing purchases   (0-20): "))
            groc   = int(input("Groceries purchases  (0-20): "))
            books  = int(input("Books purchases      (0-20): "))
            sports = int(input("Sports purchases     (0-20): "))

            # Get recommendations
            recs = engine.recommend(age, gender, spend,
                                    elec, cloth, groc, books, sports)

            # Store customer data
            customer_data = {
                'name'  : name,
                'age'   : age,
                'gender': gender,
                'spend' : spend
            }

            # Print receipt
            print_receipt(customer_data, recs)

            # Save to database
            save_to_database(customer_data, recs)

        elif choice == '2':
            view_all_customers()

        elif choice == '3':
            print("\n👋 Goodbye! ShopAI shutting down.\n")
            break

        else:
            print("\n[!] Invalid choice. Enter 1, 2 or 3.")


if __name__ == "__main__":
    main()