import os
import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
import pickle
import time


def title_header():
    st.title("Tropical Cyclone Best Track Prediction")
    st.header(
        "Test the track prediction by using the Typhoon data from 2021-2023"
    )

    st.write("🟢Green circles are tail tracks")
    st.write("🔵Blue circles are head tracks")
    st.write("🔴Red circles are predicted tracks")


def get_dataset_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'resource', 'typhoon_21_23.csv')
    return file_path


def get_data_frame():
    df = pd.read_csv(get_dataset_path())

    return df


def split_X_y(df):
    suffixes = [f"_{i}" for i in range(15, 20)]
    columns_to_extract = [col for col in df.columns if any(col.endswith(suffix) for suffix in suffixes)]

    # 创建新的DataFrame y，仅包含这些列
    y = df[columns_to_extract]

    # 创建新的DataFrame X，不包含这些列
    X = df.drop(columns=columns_to_extract)
    X = X.drop(['ID'], axis=1)

    columns_to_keep = [
    'Latitude of the center_15', 'Latitude of the center_16',
    'Latitude of the center_17', 'Latitude of the center_18',
    'Latitude of the center_19', 'Longitude of the center_15',
    'Longitude of the center_16', 'Longitude of the center_17',
    'Longitude of the center_18', 'Longitude of the center_19',
    'Central pressure_15', 'Central pressure_16', 'Central pressure_17',
    'Central pressure_18', 'Central pressure_19',
    'Maximum sustained wind speed_15', 'Maximum sustained wind speed_16',
    'Maximum sustained wind speed_17', 'Maximum sustained wind speed_18',
    'Maximum sustained wind speed_19']
    y = y[columns_to_keep]


    return X, y


# 这样就可以预测其他数值来预测大小了
def split_X_y_for_other(df):
    suffixes = [f"_{i}" for i in range(15, 20)]
    columns_to_extract = [col for col in df.columns if any(col.endswith(suffix) for suffix in suffixes)]

    # 创建新的DataFrame y，仅包含这些列
    y = df[columns_to_extract]

    # 创建新的DataFrame X，不包含这些列
    X = df.drop(columns=columns_to_extract)
    X = X.drop(['ID'], axis=1)

    columns_to_keep = [
    'Latitude of the center_15', 'Latitude of the center_16',
    'Latitude of the center_17', 'Latitude of the center_18',
    'Latitude of the center_19', 'Longitude of the center_15',
    'Longitude of the center_16', 'Longitude of the center_17',
    'Longitude of the center_18', 'Longitude of the center_19',
    'Central pressure_15', 'Central pressure_16', 'Central pressure_17',
    'Central pressure_18', 'Central pressure_19',
    'Maximum sustained wind speed_15', 'Maximum sustained wind speed_16',
    'Maximum sustained wind speed_17', 'Maximum sustained wind speed_18',
    'Maximum sustained wind speed_19']
    y = y[columns_to_keep]


    return X, y



def get_track_model():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'model', 'ridge_regression.pkl')
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def read_size_model():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'model', 'stacked_gbr_rf_lr.pkl')
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def read_other_model():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'model', 'random_forest_for_other.pkl')
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def size_prediction(input_data):
    model = read_size_model()
    input_data = input_data.fillna(150)

    latitude = input_data['Latitude']
    longitude = input_data['Longitude']
    pressure = input_data['Pressure']
    wind_speed = input_data['Wind Speed']

    pre_data = [[latitude, longitude, pressure, wind_speed]]
    prediction = model.predict(pre_data)
    return float(prediction)


def get_selected_row(df, id_no):
    selected_row = df[df['ID'] == id_no].copy()
    return selected_row


def get_initial_coordinates(X):
    initial_latitude = X.iloc[0, 22:23]
    initial_longitude = X.iloc[0, 37:38]
    # initial_coordinates = list(zip(initial_latitude, initial_longitude))
    return initial_longitude, initial_latitude


def get_input_coordinates(X):
    latitude_input = X.iloc[0, 15:30]
    longitude_input = X.iloc[0, 30:45]
    coordinates_input = list(zip(latitude_input, longitude_input))
    return coordinates_input


def get_actual_coordinates(y):
    actual_latitudes = y.iloc[0, 0:5]
    actual_longitudes = y.iloc[0, 5:10]

    coordinates_actual = list(zip(actual_latitudes, actual_longitudes))
    return coordinates_actual


def add_circle_on_map(m, coordinates, color):
    for crd in coordinates:
        folium.Circle(
            location=crd,
            radius=2000,
            color=color,
            fill=True,
            fill_opacity=0.2
        ).add_to(m)


def add_circle_on_map_v2(m, coordinates, df, color):
    for index, row in df.iterrows():
        # 添加圆圈
        folium.Circle(
            location=coordinates[index],
            radius=size_prediction(row) * 1000,
            color=color,
            fill=True,
            fill_opacity=0.2
        ).add_to(m)


def get_y_df(y):
    actual_latitudes = y.iloc[0, 0:5]
    actual_longitudes = y.iloc[0, 5:10]

    coordinates_actual = list(zip(actual_latitudes, actual_longitudes))
    y_pressure = y.iloc[0, 10:15]
    y_wind_speed = y.iloc[0, 15:20]
    y_pressure_wind_speed = list(zip(y_pressure, y_wind_speed))
    # 合并两个列表
    combined_list = [(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in zip(coordinates_actual, y_pressure_wind_speed)]

    # 创建DataFrame
    y_df = pd.DataFrame(combined_list, columns=['Longitude', 'Latitude', 'Pressure', 'Wind Speed'])

    return y_df


def get_X_df(X):
    # 提取包含 'pressure' 的列
    pressure_columns = [col for col in X.columns if 'pressure' in col]
    X_pressure = X[pressure_columns]
    X_pressure = X_pressure.iloc[0]
    # 提取包含 'wind speed' 的列
    wind_speed_columns = [col for col in X.columns if 'wind speed' in col]
    X_wind_speed = X[wind_speed_columns]
    X_wind_speed = X_wind_speed.iloc[0]
    X_pressure_wind_speed = list(zip(X_pressure, X_wind_speed))
    latitude_input = X.iloc[0, 15:30]
    longitude_input = X.iloc[0, 30:45]
    coordinates_input = list(zip(latitude_input, longitude_input))
    combined_list = [(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in zip(coordinates_input, X_pressure_wind_speed)]

    # 创建DataFrame
    X_df = pd.DataFrame(combined_list, columns=['Longitude', 'Latitude', 'Pressure', 'Wind Speed'])
    return X_df


def get_pred_df(X, pred):
    other_model = read_other_model()
    other_pred = other_model.predict(X)
    predicted_latitudes = pred[0, :5]
    predicted_longitudes = pred[0, 5:10]

    coordinates_predicted = list(zip(predicted_latitudes, predicted_longitudes))
    pred_pressure = other_pred[0, :5]
    pred_wind_speed = other_pred[0, 5:10]
    pred_pressure_wind_speed = list(zip(pred_pressure, pred_wind_speed))
    combined_list = [(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in zip(coordinates_predicted, pred_pressure_wind_speed)]

    # 创建DataFrame
    pred_df = pd.DataFrame(combined_list, columns=['Longitude', 'Latitude', 'Pressure', 'Wind Speed'])
    return pred_df


def main():
    st.image(
        'https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')
    df = get_data_frame()



    title_header()


    id_no = st.selectbox(
        'Select the International ID number of the Tropical Cyclone:',
        df['ID'].unique()
    )

    selected_row = get_selected_row(df, id_no)

    X, y = split_X_y(selected_row)
    X_other, y_other = split_X_y(selected_row)

    y_df = get_y_df(y_other)
    X_df = get_X_df(X_other)





    model = get_track_model()

    pred = model.predict(X)

    # pred
    predicted_latitudes = pred[0, :5]
    predicted_longitudes = pred[0, 5:10]
    coordinates_predicted = list(zip(predicted_latitudes, predicted_longitudes))

    pred_df = get_pred_df(X_other, pred)

    # From X
    input_coordinates = get_input_coordinates(X)
    # From y
    actual_coordinates = get_actual_coordinates(y)

    m = folium.Map((input_coordinates[10]), zoom_start=4)



    on = st.toggle('Size Prediction')




    if on:
        add_circle_on_map_v2(m, input_coordinates, X_df, 'blue')
        add_circle_on_map_v2(m, actual_coordinates, y_df, 'green')
        add_circle_on_map_v2(m, coordinates_predicted, pred_df, 'red')
    else:
        add_circle_on_map(m, input_coordinates, 'blue')
        add_circle_on_map(m, actual_coordinates, 'green')
        add_circle_on_map(m, coordinates_predicted, 'red')




    st_data = st_folium(m)


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Track Prediction",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()