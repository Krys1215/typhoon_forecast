import os
import folium
import keras
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Attention
from keras.layers import Lambda

from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler


@keras.saving.register_keras_serializable()
def haversine_loss(y_true, y_pred):
    lat_true, lon_true = y_true[:, 1], y_true[:, 2]
    lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]

    # 将度转换为弧度
    pi = tf.constant(np.pi)
    lat_true, lon_true, lat_pred, lon_pred = [x * (pi / 180) for x in [lat_true, lon_true, lat_pred, lon_pred]]

    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true

    a = tf.sin(dlat / 2)**2 + tf.cos(lat_true) * tf.cos(lat_pred) * tf.sin(dlon / 2)**2
    c = 2 * tf.asin(tf.sqrt(a))

    # 地球平均半径，单位公里
    R = 6371.0
    return R * c

# get the file path of the dataset
# for this one is the processed & min max scaled one
def get_dataset_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ''))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, ''))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'resource', 'pages/min_max_with_id_name.csv')
    return file_path


def get_data_frame():
    df = pd.read_csv(get_dataset_path())

    return df


def get_ts_name(df, id_no):
    return df.loc[df['International number ID'] == id_no, 'Name of the storm'].iloc[0]


def min_max_denormalize(normalized_data, min_value, max_value):
    denormalized_data = normalized_data * (max_value - min_value) + min_value
    return denormalized_data


def get_id_no_row(df, year):
    df1 = df[df["year"] == year]
    df2 = df1["International number ID"]
    return df2


# 最好还是有这个，要不还是会因为CSV的bug导致id的前几位的0不见
def df_adjustment(df):
    # df['Time of analysis'] = pd.to_datetime(df['Time of analysis'])

    # 创建新列
    # 只要一个year，因为要用year来选取ID
    # df['year'] = df['Time of analysis'].dt.year

    df['International number ID'] = df['International number ID'].astype(str)
    df['International number ID'] = df['International number ID'].apply(lambda x: x.zfill(4))
    df['International number ID'].sample(10)

    return df.loc[:, ~df.columns.str.contains('^Unnamed')]


@st.cache_data
def get_10_rows(df, selected_id):
    df = df.sort_values(by='International number ID')

    # 找到所有可能的'International number ID
    unique_ids = df['International number ID'].unique()

    # 从中随机选择一个ID
    # selected_id = np.random.choice(unique_ids, size=1)[0]

    # 获取该ID对应的所有行
    selected_rows = df[df['International number ID'] == selected_id]

    # 随机选择10行
    if len(selected_rows) >= 10:
        random_10_rows = selected_rows.sample(n=10)
    else:
        random_10_rows = selected_rows

    return random_10_rows


def get_first_half(df: pd.DataFrame):
    half_length = len(df) // 2
    return df.head(half_length)


def get_tail_half(df: pd.DataFrame):
    half_length = len(df) // 2
    return df.tail(half_length)


def get_info_by_id(df, id_no):
    lat = df.loc[df['International number ID'] == id_no, 'Latitude of the center']
    longi = df.loc[df['International number ID'] == id_no, 'Longitude of the center']
    pressure = df.loc[df['International number ID'] == id_no, 'Central pressure']
    wind_speed = df.loc[df['International number ID'] == id_no, 'Maximum sustained wind speed']
    info = pd.concat([lat, longi, pressure, wind_speed], axis=1)

    max_lat = 69.0
    min_lat = 1.4
    max_long = 188.0
    min_long = 95.0
    max_pressure = 1022
    min_pressure = 870
    max_wind_speed = 140.0
    min_wind_speed = 0.0

    info['Latitude of the center'] = info['Latitude of the center'].apply(min_max_denormalize, args=(min_lat, max_lat))
    info['Longitude of the center'] = info['Longitude of the center'].apply(min_max_denormalize,
                                                                            args=(min_long, max_long))
    info['Central pressure'] = info['Central pressure'].apply(min_max_denormalize, args=(min_pressure, max_pressure))
    info['Maximum sustained wind speed'] = info['Maximum sustained wind speed'].apply(min_max_denormalize,
                                                                                      args=(min_wind_speed, max_wind_speed))

    return info


# use the id to get the long and lat
def get_the_long_lat(df, id_no):
    lat = df.loc[df['International number ID'] == id_no, 'Latitude of the center']
    longi = df.loc[df['International number ID'] == id_no, 'Longitude of the center']
    lat = min_max_denormalize(lat, 1.4, 69.0)
    longi = min_max_denormalize(longi, 95.0, 188.0)
    coor = pd.concat([lat, longi], axis=1)
    arr = coor.values.tolist()
    return arr


def add_circle_on_map_v2(map, df, id_no, color):
    info = get_info_by_id(df, id_no)
    locations = []  # 用于存储所有圆圈的中心点坐标

    for index, row in info.iterrows():
        # 获取经纬度坐标
        location = (row['Latitude of the center'], row['Longitude of the center'])
        locations.append(location)

        # 添加圆圈
        folium.Circle(
            location=location,
            radius=size_prediction(row) * 1000,
            color=color,
            fill=True,
            fill_opacity=0.2
        ).add_to(map)

    # 在所有圆圈的中心点之间添加线
    if len(locations) > 1:
        folium.PolyLine(locations, color='red', weight=2.5, opacity=1).add_to(map)


def size_prediction(input_data):
    model = read_size_model()
    input_data = input_data.fillna(150)

    latitude = input_data['Latitude of the center']
    longitude = input_data['Longitude of the center']
    pressure = input_data['Central pressure']
    wind_speed = input_data['Maximum sustained wind speed']

    pre_data = [[latitude, longitude, pressure, wind_speed]]
    prediction = model.predict(pre_data)
    return float(prediction)
    # add float


def add_circle_on_map(map, df, id_no, color):
    coor = get_the_long_lat(df, id_no)
    for crd in coor:
        folium.Circle(
            location=crd,
            radius=200000,
            color=color,
            fill=True,
            fill_opacity=0.2
        ).add_to(map)


def adjust_head_half(df_head):
    df_head = df_head.drop(
        ['International number ID',
         'Name of the storm', 'Latitude of the center',
         'Longitude of the center'], axis=1)
    return df_head


def read_size_model():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ''))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, ''))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'model', 'stacked_gbr_rf_lr.pkl')
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def build_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(24, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(3))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='haversine_loss', optimizer='adam')  # 确保这里的loss函数是可用的
    return model


def read_model():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ''))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, ''))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'model', 'CNN-LSTM-EXP10-KAGGLE-TF-2-13.keras')
    # WORKAROUND!!!!
    new_model = build_model()

    loaded_model = keras.models.load_model(file_path)

    weights = loaded_model.get_weights()

    new_model.set_weights(weights)

    return loaded_model


def title_header():
    st.title("Tropical Cyclone Best Track Prediction")
    st.header(
        "Test the track prediction"
    )

    st.write("🟢Green circles are tail tracks")
    st.write("🔵Blue circles are head tracks")
    st.write("🔴Red circles are predicted tracks")


def main():
    st.image(
        'https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')
    df = get_data_frame()
    df = df_adjustment(df)

    title_header()

    id_no = st.selectbox(
        'Select the International ID number of the Tropical Cyclone:',
        df['International number ID'].unique()
    )

    storm_name = get_ts_name(df, id_no)

    # TODO: Dataset used: min_max_with_id_name

    # TODO get first half and second half
    # get 10 rows and ready to show
    df_show = get_10_rows(df, id_no)

    # sort it by the time.
    df_show = df_show.sort_values(by='Time of analysis')

    # we split 2 halves, head is for training, tail is for testing.
    # all split from the 10 rows we got
    df_head = get_first_half(df_show)
    df_tail = get_tail_half(df_show)

    # but anyway, we need an initial coordinates for our map.
    initial_coordinates = get_the_long_lat(df_show, id_no)

    # initialize our map
    m = folium.Map(location=initial_coordinates[0], zoom_start=4)

    # head
    add_circle_on_map_v2(m, df_head, id_no, 'blue')
    # tail
    add_circle_on_map_v2(m, df_tail, id_no, 'green')

    # -------- MAKE TRACK PREDICTION ------------

    # 1. adjust head half
    df_for_model = adjust_head_half(df_head)

    # 2. load the model
    model = read_model()

    # 3. do the prediction
    pred = model.predict(df_for_model)

    # 4. formatting the prediction
    result = pred[:, 1:, :]
    result = [i.flatten().tolist() for i in result]

    # 5. put the result on the map
    for crd in result:
        lat = crd[0]
        longi = crd[1]
        lat = min_max_denormalize(lat, 1.4, 69.0)
        longi = min_max_denormalize(longi, 95.0, 188.0)
        folium.Circle(
            location=(lat, longi),
            radius=200000,
            color='red',
            fill=True,
            fill_opacity=0.2
        ).add_to(m)

    # st_data = st_folium(m)
    map_width, map_height = 1000, 500
    m = m._repr_html_()  # 转换为 HTML 表示

    # 使用 Streamlit 的 HTML 组件来显示地图
    components.html(m, width=map_width, height=map_height)


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Track Prediction",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()



