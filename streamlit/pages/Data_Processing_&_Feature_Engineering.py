import os
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split


def get_size_data():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'tropical_cyclone_size.csv')

    df = pd.read_csv(file_path)

    return df

def get_track_data():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'RSMC_Best_Track_Data_no_null.csv')

    df = pd.read_csv(file_path)

    return df


def get_normalized_data():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'min_max_with_id_name.csv')

    df = pd.read_csv(file_path)

    return df


df_size = get_size_data()
df_track = get_track_data()
df_normalize = get_normalized_data()

def cma_size_data():
    st.subheader("For CMA Storm Size Dataset", divider='blue')
    st.caption("Preview of the data:")
    st.dataframe(df_size.sample(1), hide_index=True)

    X = df_size[['Latitude', 'Longitude', 'Pressure', 'Wind Speed']]
    y = df_size[['SiR34']]

    "1. Feature selection"
    st.code('''
    input = ['Latitude', 'Longitude', 'Pressure', 'Wind Speed']
    output = ['SiR34']
    ''')
    "2. Train Test Split"

    col1, col2 = st.columns(2)
    with col1:
        st.caption('X')
        st.dataframe(X, hide_index=True)
    with col2:
        st.caption('y')
        st.dataframe(y, hide_index=True)


def track_data():
    st.subheader("For Storm Track Dataset", divider='blue')
    st.caption("Preview of the data:")
    st.dataframe(get_track_data().drop(['Unnamed: 0'], axis=1).sample(1))

    "1. Feature Engineering"

    st.caption("1. Direction of the longest radius of 30kt winds or greater")
    st.code('''
    direction_30_mapping = {'(symmetric circle)': 1, 'Northeast (NE)': 6, 'South (S)': 3,
                            'East (E)': 5, 'Southeast (SE)': 9, 'West (W)': 4, 'North (N)': 2,
                            'Northwest (NW)': 6, 'Southwest (SW)': 8}
    ''')

    st.caption("2. direction_50_mapping")
    st.code('''
    direction_50_mapping = {'(symmetric circle)': 1, 'Northeast (NE)': 6, 'South (S)': 3,
                            'East (E)': 5, 'Southeast (SE)': 9, 'West (W)': 4, 'North (N)': 2,
                            'Northwest (NW)': 6, 'Southwest (SW)': 8,
                            'No direction (Longest radius of 50kt winds is 0)': 9}
    ''')

    st.caption("3. Indicator of landfall or passage")
    st.code('''
    indicator_mapping = {'#': 1, ' ': 0}
    ''')

    st.caption("4. Dropping unrelated attributes")
    st.code('''
    data.drop(['International number ID', 'Name of the storm', 'Grade'], axis=1)
    ''')

    st.caption("5. One-Hot Encoding")
    st.code('''
    data['Direction of the longest radius of 30kt winds or greater']
    data['Direction of the longest radius of 50kt winds or greater']
    ''')

    st.caption("6. Format time attribute to Datetime datatype")
    st.code('''
    Pandas.to_datetime(data['Time of analysis'])
    ''')

    st.caption("7. Data Normalization")
    st.code('''
    scaler = MinMaxScaler()

    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    ''')
    '  '
    "Data set after feature engineering:"
    st.dataframe(df_normalize.drop(['Unnamed: 0'], axis=1).sample(2), hide_index=True)

    "2. Train Test Split"
    X = df_normalize.drop(["Latitude of the center", "Longitude of the center", "Unnamed: 0", "International number ID",
                       "Name of the storm"], axis=1)

    col1, col2 = st.columns(2)
    with col1:
        st.caption('X')
        st.dataframe(X, hide_index=True)
    with col2:
        st.caption('y')
        st.dataframe(df_normalize.loc[:, ["Latitude of the center", "Longitude of the center"]], hide_index=True)





def main():
    st.image('https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
             '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Data Processing for Model Training")
    st.caption("Each model has it's own way to accept the format of data.")

    cma_size_data()

    track_data()


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Data Preparation",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()