import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv('data\\train.csv')
test = pd.read_csv('data\\test_x.csv')

train = train[train['Basvuru Yili'] >= 2018]

# Process 'Dogum Tarihi' to extract 'Age'
def parse_date(date_str):
    try:
        return parser.parse(str(date_str), dayfirst=True)
    except:
        return np.nan

train['Dogum Tarihi'] = train['Dogum Tarihi'].apply(parse_date)
test['Dogum Tarihi'] = test['Dogum Tarihi'].apply(parse_date)

def calculate_age(born):
    today = datetime.now()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

train['Age'] = train['Dogum Tarihi'].apply(lambda x: calculate_age(x) if pd.notnull(x) else np.nan)
test['Age'] = test['Dogum Tarihi'].apply(lambda x: calculate_age(x) if pd.notnull(x) else np.nan)

# Drop 'Dogum Tarihi' after extracting 'Age'
train.drop('Dogum Tarihi', axis=1, inplace=True)
test.drop('Dogum Tarihi', axis=1, inplace=True)

# Combine train and test data for consistent processing
data = pd.concat([train, test], sort=False)

# Process 'Cinsiyet' (Gender)
data['Cinsiyet'] = data['Cinsiyet'].str.lower().str.strip()
data['Cinsiyet'] = data['Cinsiyet'].map({'erkek': 0, 'kadın': 1, 'kadin': 1})
data['Cinsiyet'] = data['Cinsiyet'].fillna(-1)

# Process 'Universite Not Ortalamasi' and 'Lise Mezuniyet Notu'
def process_grade(grade_str):
    if pd.isnull(grade_str):
        return np.nan
    grade_str = str(grade_str).replace(',', '.').replace(' ', '')
    if '-' in grade_str:
        parts = grade_str.split('-')
        try:
            parts = [float(p) for p in parts if p != '']
            return sum(parts) / len(parts)
        except:
            return np.nan
    elif 'vealtı' in grade_str.lower():
        try:
            return float(grade_str.split('ve')[0])
        except:
            return np.nan
    else:
        try:
            return float(grade_str)
        except:
            return np.nan

data['Universite Not Ortalamasi'] = data['Universite Not Ortalamasi'].apply(process_grade)
data['Lise Mezuniyet Notu'] = data['Lise Mezuniyet Notu'].apply(process_grade)

# Process binary columns with 'Evet'/'Hayır' values
yes_no_cols = [
    'Burs Aliyor mu?', 'Daha Once Baska Bir Universiteden Mezun Olmus',
    'Baska Bir Kurumdan Burs Aliyor mu?', 'Anne Calisma Durumu',
    'Baba Calisma Durumu', 'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
    'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
    'Aktif olarak bir STK üyesi misiniz?', 'Stk Projesine Katildiniz Mi?',
    'Girisimcilikle Ilgili Deneyiminiz Var Mi?', 'Ingilizce Biliyor musunuz?'
]

for col in yes_no_cols:
    data[col] = data[col].astype(str).str.lower().str.strip()
    data[col] = data[col].map({'evet': 1, 'hayır': 0, 'hayir': 0})
    data[col] = data[col].fillna(0)

# Handle missing numerical values
num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [col for col in num_cols if col not in ['Degerlendirme Puani', 'id', 'Basvuru Yili']]
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

# Fill categorical missing values with 'Unknown'
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'id']
data[cat_cols] = data[cat_cols].fillna('Unknown')

# Convert all categorical columns to strings to avoid CatBoost errors
for col in cat_cols:
    data[col] = data[col].astype(str)

# Split data back into train and test sets
train_processed = data[data['Degerlendirme Puani'].notnull()]
test_processed = data[data['Degerlendirme Puani'].isnull()]

# Define features and target variable
X_train = train_processed.drop(['Degerlendirme Puani', 'id', 'Basvuru Yili'], axis=1)
y_train = train_processed['Degerlendirme Puani']
X_test = test_processed.drop(['Degerlendirme Puani', 'id', 'Basvuru Yili'], axis=1)

# Identify categorical features (CatBoost handles these automatically)
categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Set up the parameter grid for RandomizedSearchCV
param_grid = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'iterations': [200, 500, 1000],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 50, 100]
}

# Initialize the CatBoost model
model = CatBoostRegressor(loss_function='RMSE', cat_features=categorical_features, random_state=42, verbose=False)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                   n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=-1)

# Fit the random search model
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", random_search.best_params_)

# Predict on test data with best model
y_pred = random_search.best_estimator_.predict(X_test)

# Prepare submission file
submission = test_processed[['id']].copy()
submission['Degerlendirme Puani'] = y_pred
submission.to_csv('submission_catboost_optimized_ date.csv', index=False)
#Best Parameters: {'learning_rate': 0.1, 'l2_leaf_reg': 5, 'iterations': 500, 'depth': 8, 'border_count': 50}