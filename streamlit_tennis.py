import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
import altair as alt

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

FEATURES_ORDER_CATBOOST = [
    'year_pro_diff', 'best_of', 'year', 'player_1_hand', 'player_2_hand', 'Surface', 'Series', 'Round', 'player_1_flag', 'player_2_flag', 'Court',
    'height_weight_interaction', 'custom_log_rank_diff', 'custom_rank_year_pro_interaction', 'custom_rank_surface_Carpet_interaction',
    'custom_rank_surface_Clay_interaction', 'custom_rank_surface_Grass_interaction', 'custom_rank_surface_Hard_interaction',
    'custom_rank_court_Indoor_interaction', 'custom_rank_court_Outdoor_interaction'
]

FEATURES_ORDER_XGBOOST = [
    'year_pro_diff', 'best_of', 'year', 'player_1_hand', 'player_2_hand', 'Surface', 'Series', 'Round', 'player_1_flag', 'player_2_flag', 'Court',
    'height_weight_interaction', 'rank_surface_Carpet_interaction', 'rank_surface_Clay_interaction', 'rank_surface_Grass_interaction', 'rank_surface_Hard_interaction',
    'rank_court_Indoor_interaction', 'rank_court_Outdoor_interaction', 'poly_height_diff', 'poly_weight_diff', 'poly_rank_diff', 'poly_height_diff^2',
    'poly_height_diff_weight_diff', 'poly_height_diff_rank_diff', 'poly_weight_diff^2', 'poly_weight_diff_rank_diff', 'poly_rank_diff^2'
]

# Заголовок приложения
st.title("Теннисный предсказатель победителя (Lasso, XGBoost, CatBoost)")

# Функция для загрузки датасета
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("tennis_data.csv")
        return data
    except FileNotFoundError:
        st.error("Датасет 'tennis_data.csv' не найден. Пожалуйста, загрузите файл.")
        return None

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

# Загрузка моделей и данных
lasso_model = load_lasso_model()
xgboost_model = load_xgboost_model()
catboost_model = load_catboost_model()
data = load_data()

if all(model is None for model in [lasso_model, xgboost_model, catboost_model]) or data is None:
    st.stop()

# Сайдбар для ввода данных
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
player_1_flag = st.sidebar.text_input("Страна игрока 1 (код, например, USA)", value="USA")
player_2_flag = st.sidebar.text_input("Страна игрока 2 (код, например, ESP)", value="ESP")
year = st.sidebar.number_input("Год матча", min_value=2000, max_value=2025, value=2023)

# Раздел с коэффициентами букмекеров
st.header("Bookmaker Odds Overview")
st.write("Средние коэффициенты для матчей с выбранным покрытием и типом корта:")
filtered_data = data[(data['Surface'] == surface) & (data['Court'] == court)]
if not filtered_data.empty:
    # Список всех возможных коэффициентов
    odds_columns = {
        'Bet365': ['B365W', 'B365L'],
        'Pinnacle': ['PSW', 'PSL'],
        'Expekt': ['EXW', 'EXL'],
        'Ladbrokes': ['LBW', 'LBL'],
        'Sportingbet': ['SBW', 'SBL'],
        'Unibet': ['UBW', 'UBL'],
        'Average': ['AvgW', 'AvgL'],
        'Maximum': ['MaxW', 'MaxL'],
        'Minimum': ['MinW', 'MinL']
    }
    
    odds_data = {'Bookmaker': [], 'Odds for Player 1': [], 'Odds for Player 2': []}
    for bookmaker, cols in odds_columns.items():
        if all(col in filtered_data.columns for col in cols):
            # Преобразование строк в числа, игнорируя ошибки
            for col in cols:
                filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
            # Вычисление среднего только для числовых значений
            odds_p1 = filtered_data[cols[0]].mean()
            odds_p2 = filtered_data[cols[1]].mean()
            if not pd.isna(odds_p1) and not pd.isna(odds_p2):  # Проверка на наличие данных
                odds_data['Bookmaker'].append(bookmaker)
                odds_data['Odds for Player 1'].append(odds_p1)
                odds_data['Odds for Player 2'].append(odds_p2)
    
    if odds_data['Bookmaker']:
        odds_df = pd.DataFrame(odds_data)
        st.table(odds_df.style.format({"Odds for Player 1": "{:.2f}", "Odds for Player 2": "{:.2f}"}))
    else:
        st.write("Данные о коэффициентах недоступны для выбранных параметров или содержат ошибки.")
else:
    st.write("Данные о коэффициентах недоступны для выбранных параметров.")

# Раздел с вероятностью победы игрока (по данным букмекеров)
st.header("Вероятность победы игрока (по данным букмекеров)")
if not filtered_data.empty and 'AvgW' in filtered_data.columns and 'AvgL' in filtered_data.columns:
    # Преобразование коэффициентов в числа
    filtered_data['AvgW'] = pd.to_numeric(filtered_data['AvgW'], errors='coerce')
    filtered_data['AvgL'] = pd.to_numeric(filtered_data['AvgL'], errors='coerce')
    
    # Вычисление средних коэффициентов
    avg_w = filtered_data['AvgW'].mean()
    avg_l = filtered_data['AvgL'].mean()
    
    if not pd.isna(avg_w) and not pd.isna(avg_l) and avg_w > 0 and avg_l > 0:
        # Вычисление вероятности победы игрока 1
        implied_prob_p1 = (1 / avg_w) / (1 / avg_w + 1 / avg_l)
        implied_prob_p2 = 1 - implied_prob_p1
        
        st.write(f"Вероятность победы игрока 1 (по средним коэффициентам букмекеров): **{implied_prob_p1:.2f}**")
        st.write(f"Вероятность победы игрока 2 (по средним коэффициентам букмекеров): **{implied_prob_p2:.2f}**")
        
        # Создаем DataFrame для Altair
        chart_data = pd.DataFrame({
            "Player": ["Player 1", "Player 2"],
            "Вероятность": [implied_prob_p1, implied_prob_p2]
        })
        
        # Создаем график с Altair
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Player", title=""),
            y=alt.Y("Вероятность", title="Вероятность"),
            color=alt.Color("Player", legend=None)
        ).properties(
            width="container"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("Невозможно вычислить вероятность: данные о средних коэффициентах отсутствуют или некорректны.")
else:
    st.write("Данные о средних коэффициентах недоступны для выбранных параметров.")

st.write("Выберите модель для предсказания победы первого игрока.")
# Выбор модели
model_choice = st.selectbox("Выберите модель", ["Lasso", "XGBoost", "CatBoost"])

# Подготовка данных для предсказания
rank_diff = player_1_rank - player_2_rank
height_diff = player_1_height - player_2_height
weight_diff = player_1_weight - player_2_weight
year_pro_diff = player_1_year_pro - player_2_year_pro
height_weight_interaction = height_diff * weight_diff
custom_log_rank_diff = np.log1p(abs(rank_diff)) * np.sign(rank_diff)
custom_rank_year_pro_interaction = rank_diff * year_pro_diff

# Создаём dummy-переменные для Surface и Court
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

# Полиномиальные признаки (для Lasso и XGBoost)
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
hand_mapping = {"Right": 0, "Left": 1, "Right-Handed": 0, "Left-Handed": 1}
series_mapping = {"ATP250": 0, "ATP500": 1, "Masters": 2, "Grand Slam": 3}
round_mapping = {"1st Round": 0, "2nd Round": 1, "3rd Round": 2, "Quarterfinals": 3, "Semifinals": 4, "Final": 5}
flag_mapping = {"USA": 0, "ESP": 1}
surface_mapping = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
court_mapping = {"Indoor": 0, "Outdoor": 1}

# Формируем входные данные для каждой модели
input_data_lasso = pd.DataFrame([{
    "year_pro_diff": year_pro_diff,
    "best_of": best_of,
    "year": year,
    "player_1_hand": hand_mapping[player_1_hand],
    "player_2_hand": hand_mapping[player_2_hand],
    "Surface": surface_mapping[surface],
    "Series": series_mapping[series],
    "Round": round_mapping[round],
    "player_1_flag": flag_mapping.get(player_1_flag, 0),
    "player_2_flag": flag_mapping.get(player_2_flag, 1),
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

input_data_catboost = pd.DataFrame([{
    "year_pro_diff": year_pro_diff,
    "best_of": best_of,
    "year": year,
    "player_1_hand": hand_mapping[player_1_hand],
    "player_2_hand": hand_mapping[player_2_hand],
    "Surface": surface_mapping[surface],
    "Series": series_mapping[series],
    "Round": round_mapping[round],
    "player_1_flag": flag_mapping.get(player_1_flag, 0),
    "player_2_flag": flag_mapping.get(player_2_flag, 1),
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

input_data_xgboost = pd.DataFrame([{
    "year_pro_diff": year_pro_diff,
    "best_of": best_of,
    "year": year,
    "player_1_hand": hand_mapping[player_1_hand],
    "player_2_hand": hand_mapping[player_2_hand],
    "Surface": surface_mapping[surface],
    "Series": series_mapping[series],
    "Round": round_mapping[round],
    "player_1_flag": flag_mapping.get(player_1_flag, 0),
    "player_2_flag": flag_mapping.get(player_2_flag, 1),
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

# Предсказание
if st.button("Предсказать"):
    prediction = None
    
    if model_choice == "Lasso" and lasso_model:
        prediction = lasso_model.predict_proba(input_data_lasso)[:, 1][0]
    elif model_choice == "XGBoost" and xgboost_model:
        prediction = xgboost_model.predict_proba(input_data_xgboost)[:, 1][0]
    elif model_choice == "CatBoost" and catboost_model:
        prediction = catboost_model.predict_proba(input_data_catboost)[:, 1][0]
    
    if prediction is not None:
        st.session_state.predictions.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model": model_choice,
            "player_1_rank": player_1_rank,
            "player_2_rank": player_2_rank,
            "probability": prediction
        })
        st.write(f"Вероятность победы игрока 1 ({model_choice}): **{prediction:.2f}**")
        
        # Создаем DataFrame для Altair
        chart_data = pd.DataFrame({
            "Player": ["Player 1", "Player 2"],
            "Вероятность": [prediction, 1 - prediction]
        })
        
        # Создаем график с Altair
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Player", title=""),
            y=alt.Y("Вероятность", title="Вероятность"),
            color=alt.Color("Player", legend=None)
        ).properties(
            width="container"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("Выбранная модель недоступна.")

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
    })
    
    # Извлекаем год
    df['year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
    
    # Вычисляем разницы
    df["height_diff"] = df["player_1_height"] - df["player_2_height"]
    df["weight_diff"] = df["player_1_weight"] - df["player_2_weight"]
    df["rank_diff"] = df["player_1_rank"] - df["player_2_rank"]
    df["year_pro_diff"] = df["player_1_year_pro"] - df["player_2_year_pro"]
    df["height_weight_interaction"] = df["height_diff"] * df["weight_diff"]
    df["custom_log_rank_diff"] = np.log1p(abs(df["rank_diff"].fillna(0))) * np.sign(df["rank_diff"].fillna(0))
    df["custom_rank_year_pro_interaction"] = df["rank_diff"].fillna(0) * df["year_pro_diff"].fillna(0)
    
    # Dummy-переменные
    surface_dummies = pd.get_dummies(df["surface"].astype(str).fillna("Unknown"), prefix="surface")
    court_dummies = pd.get_dummies(df["Court"].astype(str).fillna("Outdoor"), prefix="court")
    df = pd.concat([df, surface_dummies, court_dummies], axis=1)
    
    # Взаимодействия
    for col in surface_dummies.columns:
        df[f"rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
        df[f"custom_rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
    for col in court_dummies.columns:
        df[f"rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
        df[f"custom_rank_{col}_interaction"] = df["rank_diff"].fillna(0) * df[col]
    
    # Полиномиальные признаки (для Lasso и XGBoost)
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
    df = df.dropna(subset=required_columns)
    
    # Кодирование категориальных признаков
    df["player_1_hand"] = df["player_1_hand"].map(hand_mapping).fillna(0)
    df["player_2_hand"] = df["player_2_hand"].map(hand_mapping).fillna(0)
    df["Surface"] = df["surface"].map(surface_mapping).fillna(0)
    df["Court"] = df["Court"].map(court_mapping).fillna(1)
    df["Series"] = df["Series"].map(series_mapping).fillna(0)
    df["Round"] = df["Round"].map(round_mapping).fillna(0)
    df["player_1_flag"] = df["player_1_flag"].map(flag_mapping).fillna(0)
    df["player_2_flag"] = df["player_2_flag"].map(flag_mapping).fillna(1)
    
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
        "rank_surface_Carpet_interaction": df["rank_surface_Carpet_interaction"],
        "rank_surface_Clay_interaction": df["rank_surface_Clay_interaction"],
        "rank_surface_Grass_interaction": df["rank_surface_Grass_interaction"],
        "rank_surface_Hard_interaction": df["rank_surface_Hard_interaction"],
        "rank_court_Indoor_interaction": df["rank_court_Indoor_interaction"],
        "rank_court_Outdoor_interaction": df["rank_court_Outdoor_interaction"],
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
        "custom_rank_surface_Carpet_interaction": df["custom_rank_surface_Carpet_interaction"],
        "custom_rank_surface_Clay_interaction": df["custom_rank_surface_Clay_interaction"],
        "custom_rank_surface_Grass_interaction": df["custom_rank_surface_Grass_interaction"],
        "custom_rank_surface_Hard_interaction": df["custom_rank_surface_Hard_interaction"],
        "custom_rank_court_Indoor_interaction": df["custom_rank_court_Indoor_interaction"],
        "custom_rank_court_Outdoor_interaction": df["custom_rank_court_Outdoor_interaction"]
    })[FEATURES_ORDER_CATBOOST]

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
        "rank_surface_Carpet_interaction": df["rank_surface_Carpet_interaction"],
        "rank_surface_Clay_interaction": df["rank_surface_Clay_interaction"],
        "rank_surface_Grass_interaction": df["rank_surface_Grass_interaction"],
        "rank_surface_Hard_interaction": df["rank_surface_Hard_interaction"],
        "rank_court_Indoor_interaction": df["rank_court_Indoor_interaction"],
        "rank_court_Outdoor_interaction": df["rank_court_Outdoor_interaction"],
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

    prediction = None
    if model_choice == "Lasso" and lasso_model:
        prediction = lasso_model.predict_proba(df_input_lasso)[:, 1]
    elif model_choice == "XGBoost" and xgboost_model:
        prediction = xgboost_model.predict_proba(df_input_xgboost)[:, 1]
    elif model_choice == "CatBoost" and catboost_model:
        prediction = catboost_model.predict_proba(df_input_catboost)[:, 1]
    
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
else:
    st.write("Важность признаков недоступна для выбранной модели.")