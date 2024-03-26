import pandas as pd
import streamlit as st
from PIL import Image
from src.constants import set_constants
from src.model import load_model_and_predict, fit_and_save_model


def process_main_page(
) -> None:
    image = Image.open('src/img/wine.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Wine Quality",
        page_icon=image,

    )

    st.write(
        """
        # Классификация качества вина
        Определяем идеальное вино по его характеристикам.
        """
    )

    st.image(image)

def write_user_data(
        df: pd.DataFrame
) -> None:
    st.write("## Ваши данные")
    st.write(df)

def write_prediction(
        prediction: str,
        prediction_probas: pd.DataFrame
) -> None:
    images = {
        1: 'src/img/1.jpg',
        2: 'src/img/2.jpg',
        3: 'src/img/3.jpg',
        4: 'src/img/4.jpg',
        5: 'src/img/5.jpg',
        6: 'src/img/6.jpg',
        7: 'src/img/7.jpg',
        8: 'src/img/8.jpg',
        9: 'src/img/9.jpg',
        10: 'src/img/10.jpg'
    }

    st.write("## Предсказание")

    st.image(Image.open(images[prediction]).resize((350, 350)), width=350)
    st.write(f"Оценка вина: {prediction}")

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)

def process_side_bar_inputs(
) -> None:
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    user_X_df = user_input_df
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)

def sidebar_input_features(
) -> None:
    # Wine type
    type = st.sidebar.selectbox("Цвет", ("Белое", "Красное"))
    # Fixed Acidity
    fixed_acidity = st.sidebar.slider(
        "Фиксированная кислотность",
        min_value=0.1, max_value=20.0, value=7.4, step=0.1)
    # Volatile Acidity
    volatile_acidity = st.sidebar.slider(
        "Летучая кислотность",
        min_value=0.1, max_value=2.0, value=0.36, step=0.01)
    # Citric Acid
    citric_acid = st.sidebar.slider(
        "Лимонная кислота",
        min_value=0.1, max_value=3.0, value=0.3, step=0.01)
    # Residual Sugar
    residual_sugar = st.sidebar.slider(
        "Остаточный сахар",
        min_value=0.1, max_value=70.0, value=1.8, step=0.1)
    # Chlorides
    chlorides = st.sidebar.slider(
        "Хлориды",
        min_value=0.1, max_value=0.7, value=0.074, step=0.001)
    # Free Sulfur Dioxide
    free_sulfur_dioxide = st.sidebar.slider(
        "Свободный диоксид серы",
        min_value=1.0, max_value=300.0, value=17.0, step=1.0)
    # Total Sulfur Dioxide
    total_sulfur_dioxide = st.sidebar.slider(
        "Общий диоксид серы",
        min_value=5.0, max_value=450.0, value=24.0, step=1.0)
    # Density
    density = st.sidebar.slider(
        "Плотность",
        min_value=0.7, max_value=1.5, value=0.99419, step=0.01)
    # pH
    pH = st.sidebar.slider(
        "pH",
        min_value=0.1, max_value=5.0, value=3.24, step=0.01)
    # Sulphates
    sulphates = st.sidebar.slider(
        "Сульфаты",
        min_value=0.1, max_value=2.0, value=0.7, step=0.01)
    # Alcohol
    alcohol = st.sidebar.slider(
        "Процент спирта",
        min_value=0.1, max_value=35.0, value=11.4, step=0.1)

    translation = {
        "Красное": "red",
        "Белое": "white"
    }

    data = {
        "type": translation[type],
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
    set_constants()
    fit_and_save_model()
    process_side_bar_inputs()