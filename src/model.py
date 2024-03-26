import os
import pandas as pd
import streamlit as st
from typing import List, Tuple, Optional
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@st.cache_resource
def fit_and_save_model(
) -> None:
    try:
        model = __init_model()
        __save_model(model)
    except Exception as e:
        raise e

@st.cache_resource
def load_model_and_predict(
        df: pd.DataFrame,
        path: Optional[str] = None
) -> Tuple[str, pd.DataFrame]:
    model = __load_model(path)
    prediction, prediction_df = __get_prediction(model, df)

    return prediction, prediction_df

@st.cache_resource
def __init_model(
        data_path: Optional[str] = None
) -> CatBoostClassifier:
    data = __load_data(data_path)
    target_col, num_cols, cat_cols = __get_columns(data)
    X_train, X_test, y_train, y_test = __split_data(data, target_col)
    model = CatBoostClassifier(n_estimators=200, cat_features=cat_cols)
    model.fit(X_train, y_train)
    test_prediction = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_prediction)
    print(f"Model accuracy is {test_accuracy}")

    return model

def __save_model(
        model: CatBoostClassifier
) -> None:
    path = os.environ['MODEL_WEIGHTS_PATH']
    model.save_model(path)
    print(f"Model was saved to {path}")

@st.cache_data
def __load_data(
        path: Optional[str] = None
) -> pd.DataFrame:
    data = pd.read_csv(path if path is not None else os.environ['DATA_PATH'], sep=',')
    data = data.dropna(inplace=False)

    return data

@st.cache_data
def __get_columns(
        data: pd.DataFrame
) -> Tuple[str, List[str], List[str]]:
    # Целевой признак
    target_col = os.environ['DATA_TARGET_FEATURE']
    # Числовые признаки
    num_cols = list(filter(lambda item: item != target_col, sorted(data.select_dtypes(include=['float64']).columns.to_list())))
    # Категориальные признаки
    cat_cols = list(filter(lambda item: item != target_col, sorted(data.select_dtypes(include=['object']).columns.to_list())))

    return target_col, num_cols, cat_cols

@st.cache_data
def __split_data(
        data: pd.DataFrame,
        target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = data.loc[:, data.columns != target_col]
    y = data.loc[:, data.columns == target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=42)

    return X_train, X_test, y_train, y_test

@st.cache_resource
def __load_model(
        model_weights: Optional[str] = None
) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(model_weights if model_weights is not None else os.environ['MODEL_WEIGHTS_PATH'])

    return model

def __get_prediction(
        model: CatBoostClassifier,
        row: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    prediction = model.predict(row)
    prediction_proba = model.predict_proba(row)

    encode_prediction_proba = {
        0: "1",
        1: "2",
        2: "3",
        3: "4",
        4: "5",
        5: "6",
        6: "7",
        7: "8",
        8: "9",
        9: "10",
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[0][key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = prediction[0][0]

    return prediction, prediction_df


if __name__ == "__main__":
    fit_and_save_model()