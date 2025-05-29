import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import statsmodels.api as sm

# ---------- Custom Styling ----------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #C8A2D4;  /* Lilac color */
        }
        .stSidebar {
            background-color: #D1A7D5;  /* A lighter lilac color for sidebar */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Header ----------
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/8/83/TUMS_Signature_Variation_1_BLUE.png' width='200' style='margin-bottom: 10px;'/>
    </div>
    """,
    unsafe_allow_html=True
)

st.title('🤖🤰 Machine Learning Models APP for Advanced Predicting Infertility Risk in Women')
st.info('Predict the **Infertility** based on health data using NNet and Logistic Regression.')

# ---------- Load Data ----------
@st.cache_data
def load_data():
    url = "https://github.com/Bahsobi/WWI_project/raw/refs/heads/main/encodeddata.xlsx"
    return pd.read_excel(url)

df = load_data()

# ---------- Rename Columns ----------
df.rename(columns={
    'AGE': 'age',
    'Race': 'race',
    'BMI': 'BMI',
    'Hyperlipidemia': 'hyperlipidemia',
    'diabetes': 'diabetes',
    'Female infertility': 'infertility',
    'WWI': 'WWI',
    'HOMA-IR': 'HOMA_IR'
}, inplace=True)

# ---------- Features & Target ----------
features = ['WWI', 'age', 'BMI', 'HOMA_IR', 'race', 'hyperlipidemia', 'diabetes']
target = 'infertility'
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# ---------- Preprocessing ----------
categorical_features = ['race', 'hyperlipidemia', 'diabetes']
numerical_features = ['WWI', 'age', 'BMI', 'HOMA_IR']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# ---------- XGBoost Pipeline ----------
model = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# ---------- Feature Importance ----------
xgb_model = model.named_steps['xgb']
encoder = model.named_steps['prep'].named_transformers_['cat']
feature_names = encoder.get_feature_names_out(categorical_features).tolist() + numerical_features
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# ---------- Logistic Regression for Odds Ratio ----------
odds_pipeline = Pipeline([
    ('prep', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])
odds_pipeline.fit(X_train, y_train)
log_model = odds_pipeline.named_steps['logreg']
odds_ratios = np.exp(log_model.coef_[0])

odds_df = pd.DataFrame({
    'Feature': feature_names,
    'Odds Ratio': odds_ratios
}).sort_values(by='Odds Ratio', ascending=False)

filtered_odds_df = odds_df[~odds_df['Feature'].str.contains("race")]

# ---------- Sidebar User Input ----------
st.sidebar.header("📝 Input Individual Data")
race_options = [
    "Mexican American", "Other Hispanic", "Non-Hispanic White",
    "Non-Hispanic Black", "Non-Hispanic Asian", "Other Race - Including Multi-Racial"
]

WWI = st.sidebar.number_input("WWI (8.04 - 14.14)", min_value=8.04, max_value=14.14, value=10.0)
age = st.sidebar.number_input("Age (18 - 59)", min_value=18, max_value=59, value=30)
bmi = st.sidebar.number_input("BMI (14.6 - 82.0)", min_value=14.6, max_value=82.0, value=25.0)
HOMA_IR = st.sidebar.number_input("HOMA-IR (0.22 - 34.1)", min_value=0.22, max_value=34.1, value=2.0)
race = st.sidebar.selectbox("Race", race_options)
hyperlipidemia = st.sidebar.selectbox("Hyperlipidemia", ['Yes', 'No'])
diabetes = st.sidebar.selectbox("Diabetes", ['Yes', 'No'])

# ---------- Prediction ----------
user_input = pd.DataFrame([{
    'WWI': WWI,
    'age': age,
    'BMI': bmi,
    'HOMA_IR': HOMA_IR,
    'race': race,
    'hyperlipidemia': hyperlipidemia,
    'diabetes': diabetes
}])

prediction = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0][1]
odds_value = probability / (1 - probability)

# ---------- Display Result ----------
if prediction == 1:
    st.error(f"""
        ⚠️ **Prediction: Infertile**

        🧮 **Probability of Infertility:** {probability:.2%}  
        🎲 **Odds of Infertility:** {odds_value:.2f}
    """)
else:
    st.success(f"""
        ✅ **Prediction: Not Infertile**

        🧮 **Probability of Infertility:** {probability:.2%}  
        🎲 **Odds of Infertility:** {odds_value:.2f}
    """)

# ---------- Show Tables ----------
st.subheader("📊 Odds Ratios for Infertility (Logistic Regression) (Excluding Race)")
st.dataframe(filtered_odds_df)

st.subheader("💡 Feature Importances (XGBoost)")
st.dataframe(importance_df)

# ---------- Plot Feature Importances ----------
st.subheader("📈 Bar Chart: Feature Importances")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
st.pyplot(fig)

# ---------- Quartile Odds Ratio for WWI ----------
st.subheader("📉 Odds Ratios for Infertility by WWI Quartiles")
df_wwi = df[['WWI', 'infertility']].copy()
df_wwi['WWI_quartile'] = pd.qcut(df_wwi['WWI'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

X
