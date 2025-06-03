import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
import altair as alt
import warnings
warnings.filterwarnings('ignore')

# Инициализация сессии
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# Ожидаемые порядки признаков
FEATURES_ORDER_LASSO = [
    'year_pro_diff', 'best_of', 'year', 'player_1_hand', 'player_2_hand', 'Surface', 'Series', 'Round', 'player_1_flag', 'player_2_flag', 'Court',
    'height_weight_interaction', 'rank_surface_Carpet_interaction', 'rank_surface_Clay_interaction', 'rank_surface_Grass_interaction', 'rank_surface_Hard_interaction',
    'rank_court_Indoor_interaction', 'rank_court_Outdoor_interaction', 'poly_height_diff', 'poly_weight_diff', 'poly_rank_diff', 'poly_height_diff^2',
    'poly_height_diff_weight_diff', 'poly_height_diff_rank_diff', 'poly_weight_diff^2', 'poly_weight_diff_rank_diff', 'poly_rank_diff^2'
]

FEATURES_ORDER_XGBOOST = [
    'year_pro_diff', 'best_of', 'year', 'player_1_hand', 'player_2_hand', 'Surface', 'Series', 'Round', 'player_1_flag', 'player_2_flag', 'Court',
    'height_weight_interaction', 'rank_surface_Carpet_interaction', 'rank_surface_Clay_interaction', 'rank_surface_Grass_interaction', 'rank_surface_Hard_interaction',
    'rank_court_Indoor_interaction', 'rank_court_Outdoor_interaction', 'poly_height_diff', 'poly_weight_diff', 'poly_rank_diff', 'poly_height_diff^2',
    'poly_height_diff_weight_diff', 'poly_height_diff_rank_diff', 'poly_weight_diff^2', 'poly_weight_diff_rank_diff', 'poly_rank_diff^2'
]

FEATURES_ORDER_CATBOOST = [
    'year_pro_diff', 'best_of', 'year', 'player_1_hand', 'player_2_hand', 'Surface', 'Series', 'Round', 'player_1_flag', 'player_2_flag', 'Court',
    'height_weight_interaction', 'custom_log_rank_diff', 'custom_rank_year_pro_interaction', 'custom_rank_surface_Carpet_interaction',
    'custom_rank_surface_Clay_interaction', 'custom_rank_surface_Grass_interaction', 'custom_rank_surface_Hard_interaction',
    'custom_rank_court_Indoor_interaction', 'custom_rank_court_Outdoor_interaction'
]

FEATURES_ORDER_CATBOOST_EXTENDED = [
    'year_pro_diff', 'best_of', 'player_1_hand', 'player_2_hand', 'Surface', 'Series', 'Round', 'player_1_flag', 'player_2_flag', 'Court',
    'height_weight_interaction', 'custom_log_rank_diff', 'custom_rank_year_pro_interaction', 'player_1_B365', 'player_2_B365', 'player_1_Avg', 'player_2_Avg',
    'bet_diff_B365', 'bet_diff_Avg', 'log_B365_ratio', 'log_Avg_ratio', 'bet_rank_interaction_B365', 'bet_rank_interaction_Avg',
    'player_1_years_since_pro', 'player_2_years_since_pro', 'custom_rank_surface_Carpet_interaction', 'custom_rank_surface_Clay_interaction',
    'custom_rank_surface_Grass_interaction', 'custom_rank_surface_Hard_interaction', 'custom_rank_court_Indoor_interaction', 'custom_rank_court_Outdoor_interaction'
]

# Заголовок приложения
st.title("Предсказание исходов теннисных матчей")

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("tennis_data.csv")
        unique_flags = pd.concat([data['pl1_flag'], data['pl2_flag']]).dropna().unique()
        return data, unique_flags
    except FileNotFoundError:
        st.error("Датасет 'tennis_data.csv' не найден. Пожалуйста, загрузите файл.")
        return None, None

# Функции для загрузки моделей
@st.cache_resource
def load_lasso_model():
    try:
        return joblib.load("lasso_model.pkl")
    except FileNotFoundError:
        st.error("Модель Lasso не найдена. Сохраните её как 'lasso_model.pkl'.")
        return None

@st.cache_resource
def load_xgboost_model():
    try:
        return joblib.load("xgboost_model.pkl")
    except FileNotFoundError:
        st.error("Модель XGBoost не найдена. Сохраните её как 'xgboost_model.pkl'.")
        return None

@st.cache_resource
def load_catboost_model():
    try:
        return CatBoostClassifier().load_model("catboost_model.cbm")
    except FileNotFoundError:
        st.error("Модель CatBoost не найдена. Сохраните её как 'catboost_model.cbm'.")
        return None

@st.cache_resource
def load_catboost_extended_model():
    try:
        return CatBoostClassifier().load_model("catboost_extended_model.cbm")
    except FileNotFoundError:
        st.error("Модель CatBoost Extended не найдена. Сохраните её как 'catboost_extended_model.cbm'.")
        return None

# Загрузка моделей и данных
lasso_model = load_lasso_model()
xgboost_model = load_xgboost_model()
catboost_model = load_catboost_model()
catboost_extended_model = load_catboost_extended_model()
data, unique_flags = load_data()

if all(model is None for model in [lasso_model, xgboost_model, catboost_model, catboost_extended_model]) or data is None:
    st.stop()

# Инициализация LabelEncoder для стран
flag_encoder = LabelEncoder()
flag_encoder.fit(unique_flags)

# Боковая панель для ввода параметров
st.sidebar.header("Параметры матча")
player_1_rank = st.sidebar.number_input("Ранг игрока 1", min_value=1, max_value=1000, value=50)
player_2_rank = st.sidebar.number_input("Ранг игрока 2", min_value=1, max_value=1000, value=60)
player_1_height = st.sidebar.number_input("Рост игрока 1 (см)", min_value=150, max_value=220, value=185)
player_2_height = st.sidebar.number_input("Рост игрока 2 (см)", min_value=150, max_value=220, value=180)
player_1_weight = st.sidebar.number_input("Вес игрока 1 (кг)", min_value=50, max_value=150, value=80)
player_2_weight = st.sidebar.number_input("Вес игрока 2 (кг)", min_value=50, max_value=150, value=75)
player_1_year_pro = st.sidebar.number_input("Год начала карьеры игрока 1", min_value=1980, max_value=2025, value=2015)
player_2_year_pro = st.sidebar.number_input("Год начала карьеры игрока 2", min_value=1980, max_value=2025, value=2015)
player_1_hand = st.sidebar.selectbox("Рука игрока 1", ["Right", "Left"], index=0)
player_2_hand = st.sidebar.selectbox("Рука игрока 2", ["Right", "Left"], index=0)
surface = st.sidebar.selectbox("Покрытие", ["Hard", "Clay", "Grass", "Carpet"], index=0)
court = st.sidebar.selectbox("Тип корта", ["Indoor", "Outdoor"], index=1)
series = st.sidebar.selectbox("Серия турнира", ["ATP250", "ATP500", "Masters", "Grand Slam"], index=0)
round = st.sidebar.selectbox("Раунд", ["1st Round", "2nd Round", "3rd Round", "Quarterfinals", "Semifinals", "Final"], index=0)
best_of = st.sidebar.number_input("Best of (3 или 5)", min_value=3, max_value=5, value=3, step=2)
player_1_flag = st.sidebar.selectbox("Страна игрока 1", unique_flags, index=list(unique_flags).index("USA") if "USA" in unique_flags else 0)
player_2_flag = st.sidebar.selectbox("Страна игрока 2", unique_flags, index=list(unique_flags).index("ESP") if "ESP" in unique_flags else 0)
year = st.sidebar.number_input("Год матча", min_value=2000, max_value=2025, value=2023)

# Ввод букмекерских коэффициентов для CatBoost Extended
st.sidebar.header("Букмекерские коэффициенты (для CatBoost Extended)")
player_1_b365 = st.sidebar.number_input("Коэффициент B365 игрока 1", min_value=1.01, max_value=100.0, value=1.80)
player_2_b365 = st.sidebar.number_input("Коэффициент B365 игрока 2", min_value=1.01, max_value=100.0, value=2.00)
player_1_avg = st.sidebar.number_input("Средний коэффициент игрока 1", min_value=1.01, max_value=100.0, value=1.85)
player_2_avg = st.sidebar.number_input("Средний коэффициент игрока 2", min_value=1.01, max_value=100.0, value=1.95)

st.header("Букмекерская информация")
if st.button("Показать букмекерские коэффициенты и вероятности"):
    with st.expander("Обзор букмекерских коэффициентов"):
        st.write("Средние коэффициенты для матчей с выбранным покрытием и типом корта:")
        filtered_data = data[(data['Surface'] == surface) & (data['Court'] == court)]
        if not filtered_data.empty:
            odds_columns = {
                'Bet365': ['B365W', 'B365L'],
                'Pinnacle': ['PSW', 'PSL'],
                'Expekt': ['EXW', 'EXL'],
                'Ladbrokes': ['LBW', 'LBL'],
                'Sportingbet': ['SBW', 'SBL'],
                'Unibet': ['UBW', 'UBL'],
                'Average': ['AvgW', 'AvgL'],
                'Maximum': ['MaxW', 'MaxL']
            }
            
            odds_data = {'Bookmaker': [], 'Odds for Player 1': [], 'Odds for Player 2': []}
            for bookmaker, cols in odds_columns.items():
                if all(col in filtered_data.columns for col in cols):
                    filtered_data[cols] = filtered_data[cols].apply(pd.to_numeric, errors='coerce')
                    odds_p1 = filtered_data[cols[0]].mean()
                    odds_p2 = filtered_data[cols[1]].mean()
                    if not pd.isna(odds_p1) and not pd.isna(odds_p2):
                        odds_data['Bookmaker'].append(bookmaker)
                        odds_data['Odds for Player 1'].append(odds_p1)
                        odds_data['Odds for Player 2'].append(odds_p2)
            
            if odds_data['Bookmaker']:
                odds_df = pd.DataFrame(odds_data)
                st.table(odds_df.style.format({"Odds for Player 1": "{:.2f}", "Odds for Player 2": "{:.2f}"}))
            else:
                st.write("Данные о коэффициентах недоступны для выбранных параметров.")
        else:
            st.write("Данные о коэффициентах недоступны для выбранных параметров.")
    
    with st.expander("Вероятность победы (букмекеры)"):
        if not filtered_data.empty and 'AvgW' in filtered_data.columns and 'AvgL' in filtered_data.columns:
            avg_w = filtered_data['AvgW'].mean()
            avg_l = filtered_data['AvgL'].mean()
            
            if not pd.isna(avg_w) and not pd.isna(avg_l) and avg_w > 0 and avg_l > 0:
                implied_prob_p1 = (1 / avg_w) / (1 / avg_w + 1 / avg_l)
                implied_prob_p2 = 1 - implied_prob_p1
                
                st.write(f"Вероятность победы игрока 1: **{implied_prob_p1:.2f}**")
                st.write(f"Вероятность победы игрока 2: **{implied_prob_p2:.2f}**")
                
                chart_data = pd.DataFrame({
                    "Player": ["Player 1", "Player 2"],
                    "Вероятность": [implied_prob_p1, implied_prob_p2]
                })
                
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Player", title=""),
                    y=alt.Y("Вероятность", title="Вероятность"),
                    color=alt.Color("Player", scale=alt.Scale(domain=["Player 1", "Player 2"], range=["#1f77b4", "#ff7f0e"]))
                ).properties(width="container")
                text = chart.mark_text(align="center", baseline="bottom", dy=-5).encode(
                    text=alt.Text("Вероятность", format=".2f")
                )
                st.altair_chart(chart + text, use_container_width=True)
            else:
                st.write("Невозможно вычислить вероятность: данные о коэффициентах отсутствуют.")
        else:
            st.write("Данные о средних коэффициентах недоступны.")

# Выбор модели
st.header("Выберите модель для предсказания победы первого игрока.")
model_choice = st.selectbox("Выберите модель", ["Lasso", "XGBoost", "CatBoost", "CatBoost Extended"])

# Валидация порядка признаков
def validate_features(input_df, expected_features):
    missing = [f for f in expected_features if f not in input_df.columns]
    if missing:
        st.error(f"Отсутствуют признаки: {', '.join(missing)}")
        return False
    return True

# Подготовка данных для предсказания
rank_diff = player_1_rank - player_2_rank
height_diff = player_1_height - player_2_height
weight_diff = player_1_weight - player_2_weight
year_pro_diff = player_1_year_pro - player_2_year_pro
height_weight_interaction = height_diff * weight_diff
custom_log_rank_diff = np.log1p(abs(rank_diff)) * np.sign(rank_diff)
custom_rank_year_pro_interaction = rank_diff * year_pro_diff
player_1_years_since_pro = year - player_1_year_pro
player_2_years_since_pro = year - player_2_year_pro
bet_diff_b365 = player_1_b365 - player_2_b365
bet_diff_avg = player_1_avg - player_2_avg
log_b365_ratio = np.log1p(player_1_b365 / player_2_b365)
log_avg_ratio = np.log1p(player_1_avg / player_2_avg)
bet_rank_interaction_b365 = bet_diff_b365 * rank_diff
bet_rank_interaction_avg = bet_diff_avg * rank_diff

# Dummy-переменные для Surface и Court
surface_dict = {
    "Hard": [1, 0, 0, 0],
    "Clay": [0, 1, 0, 0],
    "Grass": [0, 0, 1, 0],
    "Carpet": [0, 0, 0, 1]
}
court_dict = {"Indoor": [1, 0], "Outdoor": [0, 1]}
surface_dummies = pd.DataFrame([surface_dict.get(surface, [0, 0, 0, 0])], 
                              columns=["surface_Hard", "surface_Clay", "surface_Grass", "surface_Carpet"])
court_dummies = pd.DataFrame([court_dict[court]], columns=["court_Indoor", "court_Outdoor"])

# Взаимодействия
interaction_features = {}
for col in ["surface_Hard", "surface_Clay", "surface_Grass", "surface_Carpet"]:
    interaction_features[f"rank_{col}_interaction"] = surface_dummies[col].iloc[0] * rank_diff
    interaction_features[f"custom_rank_{col}_interaction"] = surface_dummies[col].iloc[0] * rank_diff
for col in ["court_Indoor", "court_Outdoor"]:
    interaction_features[f"rank_{col}_interaction"] = court_dummies[col].iloc[0] * rank_diff
    interaction_features[f"custom_rank_{col}_interaction"] = court_dummies[col].iloc[0] * rank_diff

# Полиномиальные признаки
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_features = poly.fit_transform(pd.DataFrame({
    "height_diff": [height_diff],
    "weight_diff": [weight_diff],
    "rank_diff": [rank_diff]
}))
poly_feature_names = poly.get_feature_names_out(["height_diff", "weight_diff", "rank_diff"])
poly_columns = [f"poly_{name.replace(' ', '_')}" for name in poly_feature_names]
poly_df = pd.DataFrame(poly_features, columns=poly_columns)

# Кодирование категориальных признаков
hand_mapping = {"Right": 0, "Left": 1}
series_mapping = {"ATP250": 0, "ATP500": 1, "Masters": 2, "Grand Slam": 3}
round_mapping = {"1st Round": 0, "2nd Round": 1, "3rd Round": 2, "Quarterfinals": 3, "Semifinals": 4, "Final": 5}
surface_mapping = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
court_mapping = {"Indoor": 0, "Outdoor": 1}

player_1_flag_encoded = flag_encoder.transform([player_1_flag])[0]
player_2_flag_encoded = flag_encoder.transform([player_2_flag])[0]

# Формируем входные данные для моделей
input_data_lasso = pd.DataFrame([{
    "year_pro_diff": year_pro_diff,
    "best_of": best_of,
    "year": year,
    "player_1_hand": hand_mapping[player_1_hand],
    "player_2_hand": hand_mapping[player_2_hand],
    "Surface": surface_mapping[surface],
    "Series": series_mapping[series],
    "Round": round_mapping[round],
    "player_1_flag": player_1_flag_encoded,
    "player_2_flag": player_2_flag_encoded,
    "Court": court_mapping[court],
    "height_weight_interaction": height_weight_interaction,
    "rank_surface_Carpet_interaction": interaction_features["rank_surface_Carpet_interaction"],
    "rank_surface_Clay_interaction": interaction_features["rank_surface_Clay_interaction"],
    "rank_surface_Grass_interaction": interaction_features["rank_surface_Grass_interaction"],
    "rank_surface_Hard_interaction": interaction_features["rank_surface_Hard_interaction"],
    "rank_court_Indoor_interaction": interaction_features["rank_court_Indoor_interaction"],
    "rank_court_Outdoor_interaction": interaction_features["rank_court_Outdoor_interaction"],
    "poly_height_diff": poly_df["poly_height_diff"].iloc[0],
    "poly_weight_diff": poly_df["poly_weight_diff"].iloc[0],
    "poly_rank_diff": poly_df["poly_rank_diff"].iloc[0],
    "poly_height_diff^2": poly_df["poly_height_diff^2"].iloc[0],
    "poly_height_diff_weight_diff": poly_df["poly_height_diff_weight_diff"].iloc[0],
    "poly_height_diff_rank_diff": poly_df["poly_height_diff_rank_diff"].iloc[0],
    "poly_weight_diff^2": poly_df["poly_weight_diff^2"].iloc[0],
    "poly_weight_diff_rank_diff": poly_df["poly_weight_diff_rank_diff"].iloc[0],
    "poly_rank_diff^2": poly_df["poly_rank_diff^2"].iloc[0]
}])[FEATURES_ORDER_LASSO]

input_data_xgboost = pd.DataFrame([{
    "year_pro_diff": year_pro_diff,
    "best_of": best_of,
    "year": year,
    "player_1_hand": hand_mapping[player_1_hand],
    "player_2_hand": hand_mapping[player_2_hand],
    "Surface": surface_mapping[surface],
    "Series": series_mapping[series],
    "Round": round_mapping[round],
    "player_1_flag": player_1_flag_encoded,
    "player_2_flag": player_2_flag_encoded,
    "Court": court_mapping[court],
    "height_weight_interaction": height_weight_interaction,
    "rank_surface_Carpet_interaction": interaction_features["rank_surface_Carpet_interaction"],
    "rank_surface_Clay_interaction": interaction_features["rank_surface_Clay_interaction"],
    "rank_surface_Grass_interaction": interaction_features["rank_surface_Grass_interaction"],
    "rank_surface_Hard_interaction": interaction_features["rank_surface_Hard_interaction"],
    "rank_court_Indoor_interaction": interaction_features["rank_court_Indoor_interaction"],
    "rank_court_Outdoor_interaction": interaction_features["rank_court_Outdoor_interaction"],
    "poly_height_diff": poly_df["poly_height_diff"].iloc[0],
    "poly_weight_diff": poly_df["poly_weight_diff"].iloc[0],
    "poly_rank_diff": poly_df["poly_rank_diff"].iloc[0],
    "poly_height_diff^2": poly_df["poly_height_diff^2"].iloc[0],
    "poly_height_diff_weight_diff": poly_df["poly_height_diff_weight_diff"].iloc[0],
    "poly_height_diff_rank_diff": poly_df["poly_height_diff_rank_diff"].iloc[0],
    "poly_weight_diff^2": poly_df["poly_weight_diff^2"].iloc[0],
    "poly_weight_diff_rank_diff": poly_df["poly_weight_diff_rank_diff"].iloc[0],
    "poly_rank_diff^2": poly_df["poly_rank_diff^2"].iloc[0]
}])[FEATURES_ORDER_XGBOOST]

input_data_catboost = pd.DataFrame([{
    "year_pro_diff": year_pro_diff,
    "best_of": best_of,
    "year": year,
    "player_1_hand": hand_mapping[player_1_hand],
    "player_2_hand": hand_mapping[player_2_hand],
    "Surface": surface_mapping[surface],
    "Series": series_mapping[series],
    "Round": round_mapping[round],
    "player_1_flag": player_1_flag_encoded,
    "player_2_flag": player_2_flag_encoded,
    "Court": court_mapping[court],
    "height_weight_interaction": height_weight_interaction,
    "custom_log_rank_diff": custom_log_rank_diff,
    "custom_rank_year_pro_interaction": custom_rank_year_pro_interaction,
    "custom_rank_surface_Carpet_interaction": interaction_features["custom_rank_surface_Carpet_interaction"],
    "custom_rank_surface_Clay_interaction": interaction_features["custom_rank_surface_Clay_interaction"],
    "custom_rank_surface_Grass_interaction": interaction_features["custom_rank_surface_Grass_interaction"],
    "custom_rank_surface_Hard_interaction": interaction_features["custom_rank_surface_Hard_interaction"],
    "custom_rank_court_Indoor_interaction": interaction_features["custom_rank_court_Indoor_interaction"],
    "custom_rank_court_Outdoor_interaction": interaction_features["custom_rank_court_Outdoor_interaction"]
}])[FEATURES_ORDER_CATBOOST]

input_data_catboost_extended = pd.DataFrame([{
    "year_pro_diff": year_pro_diff,
    "best_of": best_of,
    "player_1_hand": hand_mapping[player_1_hand],
    "player_2_hand": hand_mapping[player_2_hand],
    "Surface": surface_mapping[surface],
    "Series": series_mapping[series],
    "Round": round_mapping[round],
    "player_1_flag": player_1_flag_encoded,
    "player_2_flag": player_2_flag_encoded,
    "Court": court_mapping[court],
    "height_weight_interaction": height_weight_interaction,
    "custom_log_rank_diff": custom_log_rank_diff,
    "custom_rank_year_pro_interaction": custom_rank_year_pro_interaction,
    "player_1_B365": player_1_b365,
    "player_2_B365": player_2_b365,
    "player_1_Avg": player_1_avg,
    "player_2_Avg": player_2_avg,
    "bet_diff_B365": bet_diff_b365,
    "bet_diff_Avg": bet_diff_avg,
    "log_B365_ratio": log_b365_ratio,
    "log_Avg_ratio": log_avg_ratio,
    "bet_rank_interaction_B365": bet_rank_interaction_b365,
    "bet_rank_interaction_Avg": bet_rank_interaction_avg,
    "player_1_years_since_pro": player_1_years_since_pro,
    "player_2_years_since_pro": player_2_years_since_pro,
    "custom_rank_surface_Carpet_interaction": interaction_features["custom_rank_surface_Carpet_interaction"],
    "custom_rank_surface_Clay_interaction": interaction_features["custom_rank_surface_Clay_interaction"],
    "custom_rank_surface_Grass_interaction": interaction_features["custom_rank_surface_Grass_interaction"],
    "custom_rank_surface_Hard_interaction": interaction_features["custom_rank_surface_Hard_interaction"],
    "custom_rank_court_Indoor_interaction": interaction_features["custom_rank_court_Indoor_interaction"],
    "custom_rank_court_Outdoor_interaction": interaction_features["custom_rank_court_Outdoor_interaction"]
}])[FEATURES_ORDER_CATBOOST_EXTENDED]

# Предсказание
if st.button("Предсказать"):
    prediction = None
    
    if model_choice == "Lasso" and lasso_model:
        prediction = lasso_model.predict_proba(input_data_lasso)[:, 1][0]
    elif model_choice == "XGBoost" and xgboost_model:
        prediction = xgboost_model.predict_proba(input_data_xgboost)[:, 1][0]
    elif model_choice == "CatBoost" and catboost_model:
        prediction = catboost_model.predict_proba(input_data_catboost)[:, 1][0]
    elif model_choice == "CatBoost Extended" and catboost_extended_model:
        prediction = catboost_extended_model.predict_proba(input_data_catboost_extended)[:, 1][0]
    
    if prediction is not None:
        st.session_state.predictions.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": model_choice,
            "player_1_rank": player_1_rank,
            "player_2_rank": player_2_rank,
            "probability": prediction
        })
        st.write(f"Вероятность победы игрока 1 ({model_choice}): **{prediction:.2f}**")
        
        chart_data = pd.DataFrame({
            "Player": ["Player 1", "Player 2"],
            "Вероятность": [prediction, 1 - prediction]
        })
        
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Player", title=""),
            y=alt.Y("Вероятность", title="Вероятность"),
            color=alt.Color("Player", scale=alt.Scale(domain=["Player 1", "Player 2"], range=["#1f77b4", "#ff7f0e"]))
        ).properties(width="container")
        text = chart.mark_text(align="center", baseline="bottom", dy=-5).encode(
            text=alt.Text("Вероятность", format=".2f")
        )
        st.altair_chart(chart + text, use_container_width=True)
    else:
        st.error("Выбранная модель недоступна.")

# Сравнение всех моделей
if st.checkbox("Сравнить все модели"):
    predictions = {}
    
    if lasso_model and validate_features(input_data_lasso, FEATURES_ORDER_LASSO):
        predictions["Lasso"] = lasso_model.predict_proba(input_data_lasso)[:, 1][0]
    
    if xgboost_model and validate_features(input_data_xgboost, FEATURES_ORDER_XGBOOST):
        predictions["XGBoost"] = xgboost_model.predict_proba(input_data_xgboost)[:, 1][0]
    
    if catboost_model and validate_features(input_data_catboost, FEATURES_ORDER_CATBOOST):
        predictions["CatBoost"] = catboost_model.predict_proba(input_data_catboost)[:, 1][0]
    
    if catboost_extended_model and validate_features(input_data_catboost_extended, FEATURES_ORDER_CATBOOST_EXTENDED):
        predictions["CatBoost Extended"] = catboost_extended_model.predict_proba(input_data_catboost_extended)[:, 1][0]
    
    if predictions:
        metrics = {
            "Lasso": {"Accuracy": 0.65, "F1": 0.66, "AUC": 0.69},
            "XGBoost": {"Accuracy": 0.66, "F1": 0.65, "AUC": 0.72},
            "CatBoost": {"Accuracy": 0.66, "F1": 0.66, "AUC": 0.72},
            "CatBoost Extended": {"Accuracy": 0.75, "F1": 0.75, "AUC": 0.85}
        }
        
        comparison_df = pd.DataFrame({
            "Model": list(predictions.keys()),
            "Probability": list(predictions.values()),
            "Accuracy": [metrics[model]["Accuracy"] for model in predictions.keys()],
            "F1": [metrics[model]["F1"] for model in predictions.keys()],
            "AUC": [metrics[model]["AUC"] for model in predictions.keys()]
        })
        st.write("Сравнение моделей:")
        st.table(comparison_df.style.format({
            "Probability": "{:.2f}",
            "Accuracy": "{:.2f}",
            "F1": "{:.2f}",
            "AUC": "{:.2f}"
        }))
        
        chart = alt.Chart(comparison_df).mark_bar().encode(
            x=alt.X("Model", title="Model", sort=["Lasso", "XGBoost", "CatBoost", "CatBoost Extended"]),
            y=alt.Y("Probability", title="Probability"),
            color=alt.Color("Model", scale=alt.Scale(scheme="category10"))
        ).properties(width="container")
        text = chart.mark_text(align="center", baseline="bottom", dy=-5).encode(
            text=alt.Text("Probability", format=".2f")
        )
        st.altair_chart(chart + text, use_container_width=True)

# История предсказаний
st.header("История предсказаний")
if st.session_state.predictions:
    history_df = pd.DataFrame(st.session_state.predictions)
    st.table(history_df)
else:
    st.write("Нет истории предсказаний.")

# Массовое предсказание
st.header("Массовое предсказание")
uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Переименовываем колонки
    df = df.rename(columns={
        "WRank": "player_1_rank",
        "LRank": "player_2_rank",
        "pl1_hand": "player_1_hand",
        "pl2_hand": "player_2_hand",
        "Surface": "surface",
        "pl1_height": "player_1_height",
        "pl2_height": "player_2_height",
        "pl1_weight": "player_1_weight",
        "pl2_weight": "player_2_weight",
        "pl1_year_pro": "player_1_year_pro",
        "pl2_year_pro": "player_2_year_pro",
        "pl1_flag": "player_1_flag",
        "pl2_flag": "player_2_flag",
        "Best of": "best_of",
        "B365W": "player_1_B365",
        "B365L": "player_2_B365",
        "AvgW": "player_1_Avg",
        "AvgL": "player_2_Avg"
    })
    
    # Извлекаем год
    df['year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year.fillna(year)
    
    # Вычисляем разницы и дополнительные признаки
    df["height_diff"] = df["player_1_height"] - df["player_2_height"]
    df["weight_diff"] = df["player_1_weight"] - df["player_2_weight"]
    df["rank_diff"] = df["player_1_rank"] - df["player_2_rank"]
    df["year_pro_diff"] = df["player_1_year_pro"] - df["player_2_year_pro"]
    df["height_weight_interaction"] = df["height_diff"] * df["weight_diff"]
    df["custom_log_rank_diff"] = np.log1p(abs(df["rank_diff"].fillna(0))) * np.sign(df["rank_diff"].fillna(0))
    df["custom_rank_year_pro_interaction"] = df["rank_diff"].fillna(0) * df["year_pro_diff"].fillna(0)
    df["player_1_years_since_pro"] = df["year"] - df["player_1_year_pro"]
    df["player_2_years_since_pro"] = df["year"] - df["player_2_year_pro"]
    df["bet_diff_B365"] = df["player_1_B365"] - df["player_2_B365"]
    df["bet_diff_Avg"] = df["player_1_Avg"] - df["player_2_Avg"]
    df["log_B365_ratio"] = np.log1p(df["player_1_B365"] / df["player_2_B365"])
    df["log_Avg_ratio"] = np.log1p(df["player_1_Avg"] / df["player_2_Avg"])
    df["bet_rank_interaction_B365"] = df["bet_diff_B365"] * df["rank_diff"]
    df["bet_rank_interaction_Avg"] = df["bet_diff_Avg"] * df["rank_diff"]
    
    # Dummy-переменные
    surface_dummies = pd.get_dummies(df["surface"].astype(str).fillna("Hard"), prefix="surface")
    court_dummies = pd.get_dummies(df["Court"].astype(str).fillna("Outdoor"), prefix="court")
    df = pd.concat([df, surface_dummies, court_dummies], axis=1)
    
    # Взаимодействия
    for col in surface_dummies.columns:
        df[f"rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
        df[f"custom_rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
    for col in court_dummies.columns:
        df[f"rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
        df[f"custom_rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
    
    # Полиномиальные признаки
    df[["height_diff", "weight_diff", "rank_diff"]] = df[["height_diff", "weight_diff", "rank_diff"]].fillna(df[["height_diff", "weight_diff", "rank_diff"]].median())
    poly_features = poly.fit_transform(df[["height_diff", "weight_diff", "rank_diff"]])
    poly_df = pd.DataFrame(poly_features, columns=poly_columns, index=df.index)
    df = pd.concat([df, poly_df], axis=1)
    
    # Удаляем строки с пропущенными значениями
    required_columns = [
        "player_1_rank", "player_2_rank", "player_1_height", "player_2_height", 
        "player_1_weight", "player_2_weight", "player_1_year_pro", "player_2_year_pro", 
        "player_1_hand", "player_2_hand", "surface", "Court", "Series", "Round", 
        "best_of", "player_1_flag", "player_2_flag", "year"
    ]
    if model_choice == "CatBoost Extended":
        required_columns.extend(["player_1_B365", "player_2_B365", "player_1_Avg", "player_2_Avg"])
    df = df.dropna(subset=required_columns)
    
    # Кодирование категориальных признаков
    df["player_1_hand"] = df["player_1_hand"].replace({np.nan: "Right", "Unknown": "Right"}).map(hand_mapping).fillna(0).astype(int)
    df["player_2_hand"] = df["player_2_hand"].replace({np.nan: "Right", "Unknown": "Right"}).map(hand_mapping).fillna(0).astype(int)
    df["Surface"] = df["surface"].replace({np.nan: "Hard", "Unknown": "Hard"}).map(surface_mapping).fillna(0).astype(int)
    df["Court"] = df["Court"].replace({np.nan: "Outdoor", "Unknown": "Outdoor"}).map(court_mapping).fillna(1).astype(int)
    df["Series"] = df["Series"].replace({np.nan: "ATP250", "Unknown": "ATP250"}).map(series_mapping).fillna(0).astype(int)
    df["Round"] = df["Round"].replace({np.nan: "1st Round", "Unknown": "1st Round"}).map(round_mapping).fillna(0).astype(int)
    df["player_1_flag"] = flag_encoder.transform(df["player_1_flag"].fillna("Unknown")).astype(int)
    df["player_2_flag"] = flag_encoder.transform(df["player_2_flag"].fillna("Unknown")).astype(int)
    
    # Формируем входные данные
    df_input_lasso = pd.DataFrame({
        "year_pro_diff": df["year_pro_diff"],
        "best_of": df["best_of"],
        "year": df["year"],
        "player_1_hand": df["player_1_hand"],
        "player_2_hand": df["player_2_hand"],
        "Surface": df["Surface"],
        "Series": df["Series"],
        "Round": df["Round"],
        "player_1_flag": df["player_1_flag"],
        "player_2_flag": df["player_2_flag"],
        "Court": df["Court"],
        "height_weight_interaction": df["height_weight_interaction"],
        "rank_surface_Carpet_interaction": df.get("rank_surface_Carpet_interaction", 0),
        "rank_surface_Clay_interaction": df.get("rank_surface_Clay_interaction", 0),
        "rank_surface_Grass_interaction": df.get("rank_surface_Grass_interaction", 0),
        "rank_surface_Hard_interaction": df.get("rank_surface_Hard_interaction", 0),
        "rank_court_Indoor_interaction": df.get("rank_court_Indoor_interaction", 0),
        "rank_court_Outdoor_interaction": df.get("rank_court_Outdoor_interaction", 0),
        "poly_height_diff": df["poly_height_diff"],
        "poly_weight_diff": df["poly_weight_diff"],
        "poly_rank_diff": df["poly_rank_diff"],
        "poly_height_diff^2": df["poly_height_diff^2"],
        "poly_height_diff_weight_diff": df["poly_height_diff_weight_diff"],
        "poly_height_diff_rank_diff": df["poly_height_diff_rank_diff"],
        "poly_weight_diff^2": df["poly_weight_diff^2"],
        "poly_weight_diff_rank_diff": df["poly_weight_diff_rank_diff"],
        "poly_rank_diff^2": df["poly_rank_diff^2"]
    })[FEATURES_ORDER_LASSO]

    df_input_xgboost = pd.DataFrame({
        "year_pro_diff": df["year_pro_diff"],
        "best_of": df["best_of"],
        "year": df["year"],
        "player_1_hand": df["player_1_hand"],
        "player_2_hand": df["player_2_hand"],
        "Surface": df["Surface"],
        "Series": df["Series"],
        "Round": df["Round"],
        "player_1_flag": df["player_1_flag"],
        "player_2_flag": df["player_2_flag"],
        "Court": df["Court"],
        "height_weight_interaction": df["height_weight_interaction"],
        "rank_surface_Carpet_interaction": df.get("rank_surface_Carpet_interaction", 0),
        "rank_surface_Clay_interaction": df.get("rank_surface_Clay_interaction", 0),
        "rank_surface_Grass_interaction": df.get("rank_surface_Grass_interaction", 0),
        "rank_surface_Hard_interaction": df.get("rank_surface_Hard_interaction", 0),
        "rank_court_Indoor_interaction": df.get("rank_court_Indoor_interaction", 0),
        "rank_court_Outdoor_interaction": df.get("rank_court_Outdoor_interaction", 0),
        "poly_height_diff": df["poly_height_diff"],
        "poly_weight_diff": df["poly_weight_diff"],
        "poly_rank_diff": df["poly_rank_diff"],
        "poly_height_diff^2": df["poly_height_diff^2"],
        "poly_height_diff_weight_diff": df["poly_height_diff_weight_diff"],
        "poly_height_diff_rank_diff": df["poly_height_diff_rank_diff"],
        "poly_weight_diff^2": df["poly_weight_diff^2"],
        "poly_weight_diff_rank_diff": df["poly_weight_diff_rank_diff"],
        "poly_rank_diff^2": df["poly_rank_diff^2"]
    })[FEATURES_ORDER_XGBOOST]

    df_input_catboost = pd.DataFrame({
        "year_pro_diff": df["year_pro_diff"],
        "best_of": df["best_of"],
        "year": df["year"],
        "player_1_hand": df["player_1_hand"],
        "player_2_hand": df["player_2_hand"],
        "Surface": df["Surface"],
        "Series": df["Series"],
        "Round": df["Round"],
        "player_1_flag": df["player_1_flag"],
        "player_2_flag": df["player_2_flag"],
        "Court": df["Court"],
        "height_weight_interaction": df["height_weight_interaction"],
        "custom_log_rank_diff": df["custom_log_rank_diff"],
        "custom_rank_year_pro_interaction": df["custom_rank_year_pro_interaction"],
        "custom_rank_surface_Carpet_interaction": df.get("custom_rank_surface_Carpet_interaction", 0),
        "custom_rank_surface_Clay_interaction": df.get("custom_rank_surface_Clay_interaction", 0),
        "custom_rank_surface_Grass_interaction": df.get("custom_rank_surface_Grass_interaction", 0),
        "custom_rank_surface_Hard_interaction": df.get("custom_rank_surface_Hard_interaction", 0),
        "custom_rank_court_Indoor_interaction": df.get("custom_rank_court_Indoor_interaction", 0),
        "custom_rank_court_Outdoor_interaction": df.get("custom_rank_court_Outdoor_interaction", 0)
    })[FEATURES_ORDER_CATBOOST]

    df_input_catboost_extended = pd.DataFrame({
        "year_pro_diff": df["year_pro_diff"],
        "best_of": df["best_of"],
        "player_1_hand": df["player_1_hand"],
        "player_2_hand": df["player_2_hand"],
        "Surface": df["Surface"],
        "Series": df["Series"],
        "Round": df["Round"],
        "player_1_flag": df["player_1_flag"],
        "player_2_flag": df["player_2_flag"],
        "Court": df["Court"],
        "height_weight_interaction": df["height_weight_interaction"],
        "custom_log_rank_diff": df["custom_log_rank_diff"],
        "custom_rank_year_pro_interaction": df["custom_rank_year_pro_interaction"],
        "player_1_B365": df["player_1_B365"],
        "player_2_B365": df["player_2_B365"],
        "player_1_Avg": df["player_1_Avg"],
        "player_2_Avg": df["player_2_Avg"],
        "bet_diff_B365": df["bet_diff_B365"],
        "bet_diff_Avg": df["bet_diff_Avg"],
        "log_B365_ratio": df["log_B365_ratio"],
        "log_Avg_ratio": df["log_Avg_ratio"],
        "bet_rank_interaction_B365": df["bet_rank_interaction_B365"],
        "bet_rank_interaction_Avg": df["bet_rank_interaction_Avg"],
        "player_1_years_since_pro": df["player_1_years_since_pro"],
        "player_2_years_since_pro": df["player_2_years_since_pro"],
        "custom_rank_surface_Carpet_interaction": df.get("custom_rank_surface_Carpet_interaction", 0),
        "custom_rank_surface_Clay_interaction": df.get("custom_rank_surface_Clay_interaction", 0),
        "custom_rank_surface_Grass_interaction": df.get("custom_rank_surface_Grass_interaction", 0),
        "custom_rank_surface_Hard_interaction": df.get("custom_rank_surface_Hard_interaction", 0),
        "custom_rank_court_Indoor_interaction": df.get("custom_rank_court_Indoor_interaction", 0),
        "custom_rank_court_Outdoor_interaction": df.get("custom_rank_court_Outdoor_interaction", 0)
    })[FEATURES_ORDER_CATBOOST_EXTENDED]

    prediction = None
    if model_choice == "Lasso" and lasso_model:
        prediction = lasso_model.predict_proba(df_input_lasso)[:, 1]
    elif model_choice == "XGBoost" and xgboost_model:
        prediction = xgboost_model.predict_proba(df_input_xgboost)[:, 1]
    elif model_choice == "CatBoost" and catboost_model:
        prediction = catboost_model.predict_proba(df_input_catboost)[:, 1]
    elif model_choice == "CatBoost Extended" and catboost_extended_model:
        prediction = catboost_extended_model.predict_proba(df_input_catboost_extended)[:, 1]
    
    if prediction is not None:
        df["probability"] = prediction
        st.write("Результаты предсказаний:")
        st.table(df[["player_1_rank", "player_2_rank", "probability"]])

# Важность признаков
st.header("Важность признаков")
if model_choice == "Lasso" and lasso_model:
    # Проверяем, что модель поддерживает coef_
    try:
        coef = lasso_model.best_estimator_.coef_
        if coef.ndim == 1:  # Убеждаемся, что коэффициенты одномерные
            feature_importance = pd.DataFrame({
                "Feature": FEATURES_ORDER_LASSO,
                "Importance": np.abs(coef)
            })
        else:
            feature_importance = pd.DataFrame({
                "Feature": FEATURES_ORDER_LASSO,
                "Importance": np.abs(coef[0])  # Берем первый набор коэффициентов, если их несколько
            })
        st.bar_chart(feature_importance.set_index("Feature"))
    except AttributeError:
        st.write("Модель Lasso не поддерживает отображение важности признаков. Используйте другую модель.")
elif model_choice == "XGBoost" and xgboost_model:
    feature_importance = pd.DataFrame({
        "Feature": FEATURES_ORDER_XGBOOST,
        "Importance": xgboost_model.best_estimator_.feature_importances_
    })
    st.bar_chart(feature_importance.set_index("Feature"))
elif model_choice == "CatBoost" and catboost_model:
    feature_importance = pd.DataFrame({
        "Feature": FEATURES_ORDER_CATBOOST,
        "Importance": catboost_model.get_feature_importance()
    })
    st.bar_chart(feature_importance.set_index("Feature"))
elif model_choice == "CatBoost Extended" and catboost_extended_model:
    try:
        feature_importance = pd.DataFrame({
            "Feature": FEATURES_ORDER_CATBOOST_EXTENDED,
            "Importance": catboost_extended_model.get_feature_importance()
        })
        feature_importance = feature_importance.sort_values("Importance", ascending=False)
        st.bar_chart(feature_importance.set_index("Feature"))
    except Exception as e:
        st.error(f"Ошибка при получении важности признаков для CatBoost Extended: {str(e)}")
else:
    st.write("Важность признаков недоступна для выбранной модели.")
