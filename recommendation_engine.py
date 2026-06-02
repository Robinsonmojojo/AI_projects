import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RecommendationEngine:
    """OOP-based Customer Purchase Recommendation Engine."""

    def __init__(self, model_type='knn'):
        self.model_type = model_type
        self.model      = None
        self.le_gender  = LabelEncoder()
        self.le_cat     = LabelEncoder()
        self.scaler     = StandardScaler()
        self.features   = [
            'age', 'gender_enc', 'total_spend',
            'purchases_electronics', 'purchases_clothing',
            'purchases_groceries', 'purchases_books',
            'purchases_sports'
        ]
        self.df = None

    def fit(self, filepath='dataset.csv'):
        """Load data, train model and save it."""
        self.df = pd.read_csv(filepath)
        self.df['gender_enc']   = self.le_gender.fit_transform(self.df['gender'])
        self.df['category_enc'] = self.le_cat.fit_transform(
            self.df['preferred_category'])

        X = self.df[self.features]
        y = self.df['category_enc']

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        X_train_s     = self.scaler.fit_transform(X_train)
        self.X_test_s = self.scaler.transform(self.X_test)

        if self.model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            self.model = GaussianNB()

        self.model.fit(X_train_s, y_train)
        print(f"[✓] Model '{self.model_type}' trained successfully.")

        # Save model
        os.makedirs('models', exist_ok=True)
        bundle = {
            'model'    : self.model,
            'scaler'   : self.scaler,
            'le_gender': self.le_gender,
            'le_cat'   : self.le_cat,
        }
        with open(f'models/{self.model_type}_model.pkl', 'wb') as f:
            pickle.dump(bundle, f)
        print(f"[✓] Model saved to models/{self.model_type}_model.pkl")

    def load(self, model_type='knn'):
        """Load a previously saved model and prepare test data."""
        path = f'models/{model_type}_model.pkl'
        if not os.path.exists(path):
            print(f"[✗] No saved model at {path}. Run fit() first.")
            return
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        self.model     = bundle['model']
        self.scaler    = bundle['scaler']
        self.le_gender = bundle['le_gender']
        self.le_cat    = bundle['le_cat']
        print(f"[✓] Model loaded from {path}")

        # Re-prepare test data for evaluation
        self.df = pd.read_csv('dataset.csv')
        self.df['gender_enc']   = self.le_gender.transform(
            self.df['gender'])
        self.df['category_enc'] = self.le_cat.transform(
            self.df['preferred_category'])

        X = self.df[self.features]
        y = self.df['category_enc']

        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.X_test_s = self.scaler.transform(self.X_test)

    def recommend(self, age, gender, total_spend,
                  elec, cloth, groc, books, sports):
        """Return top 3 product recommendations."""
        gender_enc = self.le_gender.transform([gender])[0]
        customer   = pd.DataFrame(
            [[age, gender_enc, total_spend,
              elec, cloth, groc, books, sports]],
            columns=self.features
        )
        customer_s = self.scaler.transform(customer)

        proba    = self.model.predict_proba(customer_s)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3     = self.le_cat.inverse_transform(top3_idx)

        print("\n🛒 Top 3 Recommendations:")
        for rank, cat in enumerate(top3, 1):
            print(f"  {rank}. {cat.capitalize()} "
                  f"(confidence: {proba[top3_idx[rank-1]]:.1%})")
        return list(top3)

    def evaluate(self):
        """Evaluate and print model accuracy."""
        y_pred = self.model.predict(self.X_test_s)
        acc    = accuracy_score(self.y_test, y_pred)
        print(f"\n📊 Model Accuracy ({self.model_type.upper()}): {acc:.2%}")
        return acc


def get_customer_recommendations(engine):
    """Takes customer input and returns top 3 recommendations."""
    print("\n=== Customer Purchase Recommendation System ===")
    age         = int(input("Enter customer age: "))
    gender      = input("Enter gender (M/F): ").upper()
    total_spend = float(input("Enter total spend (KSh): "))
    elec        = int(input("Electronics purchases (0-20): "))
    cloth       = int(input("Clothing purchases   (0-20): "))
    groc        = int(input("Groceries purchases  (0-20): "))
    books_p     = int(input("Books purchases      (0-20): "))
    sports_p    = int(input("Sports purchases     (0-20): "))

    return engine.recommend(age, gender, total_spend,
                            elec, cloth, groc, books_p, sports_p)


# ── Main Program ──
if __name__ == "__main__":
    engine = RecommendationEngine(model_type='knn')

    if os.path.exists('models/knn_model.pkl'):
        engine.load('knn')
    else:
        engine.fit('dataset.csv')

    engine.evaluate()
    get_customer_recommendations(engine)