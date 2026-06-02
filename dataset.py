import numpy as np
import pandas as pd

np.random.seed(42)
n = 100

ages        = np.random.randint(18, 65, n)
genders     = np.random.choice(['M', 'F'], n)
total_spend = np.random.randint(500, 50000, n)

electronics = np.random.randint(0, 20, n)
clothing    = np.random.randint(0, 20, n)
groceries   = np.random.randint(0, 20, n)
books       = np.random.randint(0, 20, n)
sports      = np.random.randint(0, 20, n)

categories = ['electronics','clothing','groceries','books','sports']
purchases  = np.column_stack([electronics, clothing,
                               groceries, books, sports])
preferred  = [categories[i] for i in np.argmax(purchases, axis=1)]

df = pd.DataFrame({
    'customer_id'           : range(1, n+1),
    'age'                   : ages,
    'gender'                : genders,
    'total_spend'           : total_spend,
    'purchases_electronics' : electronics,
    'purchases_clothing'    : clothing,
    'purchases_groceries'   : groceries,
    'purchases_books'       : books,
    'purchases_sports'      : sports,
    'preferred_category'    : preferred
})

df.to_csv('dataset.csv', index=False)
print("✅ Dataset created! Shape:", df.shape)
print(df.head())