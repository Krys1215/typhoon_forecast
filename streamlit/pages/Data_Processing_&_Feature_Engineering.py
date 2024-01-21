import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
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

    file_path = os.path.join(parent1_dir, 'resource', 'RSMC_Best_Track_Data.csv')

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


def data_distribution(data = get_track_data()):
    typhoon_counts = data['International number ID'].value_counts().sort_index()
    ' '
    st.caption('Distribution of Record Counts per Typhoon:')
    fig = px.histogram(typhoon_counts, nbins=50)

    # 更新布局
    fig.update_layout(
        xaxis_title='Number of Records',
        yaxis_title='Frequency',
        showlegend=False
    )

    # 使用 Streamlit 显示图表
    st.plotly_chart(fig)

def get_train_set():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'typhoon_51_20.csv')

    df = pd.read_csv(file_path)

    return df


def get_test_set():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'typhoon_21_23.csv')

    df = pd.read_csv(file_path)

    return df


def get_y():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'y.csv')

    df = pd.read_csv(file_path)

    return df

def track_data():
    st.subheader("For Storm Track Dataset", divider='blue')
    st.caption("Preview of the data:")
    # st.dataframe(get_track_data().drop(['Unnamed: 0'], axis=1).sample(1))
    st.dataframe(get_track_data().sample(1))
    st.caption("Total Lines: 68,750")
    st.caption("Total Storms recorded: 1,884")

    "1. Drop Typhoon data lines which are less than 20 rows"
    st.caption('A total of 3,754 rows of data will be '
               'removed after deleting typhoons '
               'with less than 20 rows of records. '
               'These rows represent approximately '
               '5.46 per cent of the data set.')
    st.caption('Unify the number of rows for each typhoon to 20, and '
               'discard anything less than 20.Split the extra 20 rows; '
               'if the typhoon is 2020 with 20 extra rows, and if it is 41 rows, '
               'then the first 20 rows are 2020, the last 20 rows are '
               '2020_1, and the remaining 1 row is discarded.')
    data_distribution()
    ' '
    st.caption('Sample Format, note the ID:')
    st.dataframe(get_test_set().tail(5))

    ' '
    "2. Data Mapping"
    st.caption("Map the data from string to numerical format")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
    | Grade                                        | int |
    | ------------------------------------------- | -------- |
    | Extra-tropical Cyclone (L)                   | 0        |
    | Just entering into the responsible area of RSMC Tokyo-Typhoon Center | 1        |
    | Severe Tropical Storm (STS)                  | 2        |
    | Tropical Cyclone of TS intensity or higher  | 3        |
    | Tropical Depression (TD)                    | 4        |
    | Tropical Storm (TS)                         | 5        |
    | Typhoon (TY)                                | 6        |
    ''')
    ' '
    ' '
    with col2:
        st.markdown('''
    | Direction of the longest radius of 50kt winds or greater | int |
    | ------------------------------------------------------- | -------- |
    | Symmetric circle                                        | 1        |
    | 0                                                       | 0        |
    | East (E)                                               | 2        |
    | No direction (Longest radius of 50kt winds is 0)       | 3        |
    | North (N)                                              | 4        |
    | Northeast (NE)                                         | 5        |
    | Northwest (NW)                                         | 6        |
    | South (S)                                              | 7        |
    | Southeast (SE)                                         | 8        |
    |   Southwest (SW)                                         | 9        |
    | West (W)                                               | 10       |
    ''')

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('''
    | Direction of the longest radius of 30kt winds or greater | int |
    | ------------------------------------------------------- | -------- |
    | Symmetric circle                                        | 1        |
    | 0                                                       | 0        |
    | East (E)                                               | 2        |
    | North (N)                                              | 3        |
    | Northeast (NE)                                         | 4        |
    | Northwest (NW)                                         | 5        |
    | South (S)                                              | 6        |
    | Southeast (SE)                                         | 7        |
    | Southwest (SW)                                         | 8        |
    | West (W)                                               | 9        |
    ''')
    with col4:
        st.markdown('''
    | Indicator of landfall or passage | int |
    | ---------------------------------- | -------- |
    |                                        | 0        |
    | #                                      | 1        |
    
    ''')



    ' '
    ' '
    "2. Shaping Each Typhoon Data in 1 row"
    st.dataframe(get_data().sample(5))


    "3. Split dataset"
    st.caption("Train set: From 1951 - 2020")
    st.dataframe(get_train_set().sample(3))
    st.caption("Test set: From 2021 - 2023")
    st.dataframe(get_test_set().tail(3))


    "4. Split input & output for model training"
    st.caption('X: input will have the first 15 "rows" of every data')
    st.dataframe(get_test_set().drop(['ID'], axis=1).sample(3))
    st.caption('y: output will have the last 5 "rows" of the latitude and longitude')
    st.dataframe(get_y().tail(3))

def get_data():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'updated_reshaped_typhoon_data.csv')

    df = pd.read_csv(file_path)
    return df


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