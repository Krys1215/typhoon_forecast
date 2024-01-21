import os
import streamlit as st
import pandas as pd
import folium
import pickle
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import time

def read_model():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'model', 'stacked_gbr_rf_lr.pkl')
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def get_dataset_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'resource', 'RSMC_Best_Track_Data.csv')
    return file_path


def df_adjustment():
    df = pd.read_csv(get_dataset_path())

    df['Time of analysis'] = pd.to_datetime(df['Time of analysis'])

    # 创建新列
    df['year'] = df['Time of analysis'].dt.year
    df['month'] = df['Time of analysis'].dt.month
    df['day'] = df['Time of analysis'].dt.day
    df['hour'] = df['Time of analysis'].dt.hour

    df['International number ID'] = df['International number ID'].astype(str)
    df['International number ID'] = df['International number ID'].apply(lambda x: x.zfill(4))
    df['International number ID'].sample(10)

    return df


def get_id_no_row(df, year):
    df1 = df[df["year"] == year]
    df2 = df1["International number ID"]
    return df2


def get_ts_name(df, id_no):
    return df.loc[df['International number ID'] == id_no, 'Name of the storm'].iloc[0]


def get_the_long_lat(df, id_no):
    lat = df.loc[df['International number ID'] == id_no, 'Latitude of the center']
    longi = df.loc[df['International number ID'] == id_no, 'Longitude of the center']
    coor = pd.concat([lat, longi], axis=1)
    arr = coor.values.tolist()
    return arr


def get_info_by_id(df, id_no):
    lat = df.loc[df['International number ID'] == id_no, 'Latitude of the center']
    longi = df.loc[df['International number ID'] == id_no, 'Longitude of the center']
    pressure = df.loc[df['International number ID'] == id_no, 'Central pressure']
    wind_speed = df.loc[df['International number ID'] == id_no, 'Maximum sustained wind speed']
    info = pd.concat([lat, longi, pressure, wind_speed], axis=1)
    return info


def get_gradient_color(index, total_rows):
    # 将index映射到0-1的范围
    normalized_index = index / max(total_rows - 1, 1)

    # 生成渐变颜色，从红色渐变到绿色
    red = int(255 * (1 - normalized_index))
    blue = int(255 * normalized_index)
    return f'#{red:02x}00{blue:02x}'


def add_circle_on_map(map, df, id_no):
    info = get_info_by_id(df, id_no)
    total_rows = len(info)
    for index, row in info.iterrows():
        gradient_color = get_gradient_color(index, total_rows)

        folium.Circle(
            location=(row['Latitude of the center'], row['Longitude of the center']),
            radius=size_prediction(row) * 1000,
            color=gradient_color,
            fill=True,
            fill_opacity=0.2
        ).add_to(map)


def add_circle_on_map_no_size(map, df, id_no):
    info = get_info_by_id(df, id_no)
    for index, row in info.iterrows():
        folium.Circle(
            location=(row['Latitude of the center'], row['Longitude of the center']),
            radius=10,
            color='red',
            fill=True,
            fill_opacity=0.2
        ).add_to(map)


def size_prediction(input_data):
    model = read_model()
    input_data = input_data.fillna(150)

    latitude = input_data['Latitude of the center']
    longitude = input_data['Longitude of the center']
    pressure = input_data['Central pressure']
    wind_speed = input_data['Maximum sustained wind speed']

    pre_data = [[latitude, longitude, pressure, wind_speed]]
    prediction = model.predict(pre_data)
    return float(prediction)
    # add float


def main():
    st.image(
        'https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')
    st.title("Tropical Cyclone Best Track Visualization")
    st.write(
        "Including historical track data from 1951-2023"
    )

    # get the data
    df = df_adjustment()

    year = st.selectbox(
        "Select a year:",
        df['year'].unique()
    )

    id_no = st.selectbox(
        'Select the International ID number of the Tropical Cyclone:',
        get_id_no_row(df, year).unique()
    )

    on = st.toggle('Size Prediction')

    name = get_ts_name(df, id_no)

    st.info(f"The best track of {name} :")

    # ------------------------------------------------------------------
    initial_coordinates = get_the_long_lat(df, id_no)

    m = folium.Map(location=initial_coordinates[0], zoom_start=3)
    if on:
        progress_text = "Predicting.."
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()
        add_circle_on_map(m, df, id_no)
    else:
        add_circle_on_map_no_size(m, df, id_no)
    # st_data = st_folium(m)
    map_width, map_height = 1000, 500
    m = m._repr_html_()  # 转换为 HTML 表示

    # 使用 Streamlit 的 HTML 组件来显示地图
    components.html(m, width=map_width, height=map_height)


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Track Data",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    main()
