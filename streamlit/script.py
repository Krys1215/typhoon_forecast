import os
import pandas as pd
import pickle
import folium


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


# adjust the dataframe for visualization
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


def add_circle_on_map(map, df, id_no):
    info = get_info_by_id(df, id_no)
    for index, row in info.iterrows():
        folium.Circle(
            location=(row['Latitude of the center'], row['Longitude of the center']),
            radius=size_prediction(row) * 1000,
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


def print_sth():
    print("yes!")


def main():
    ...


if __name__ == '__main__':
    main()
