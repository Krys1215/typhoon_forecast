import os

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go


def get_dataset_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'resource', 'RSMC_Best_Track_Data.csv')
    return file_path


data = pd.read_csv(get_dataset_path())


def number_of_storms():
    st.subheader('Number of Storms Per Year', divider='blue')
    # Convert 'Time of analysis' to datetime and extract the year
    data['Time of analysis'] = pd.to_datetime(data['Time of analysis'])
    data['Year'] = data['Time of analysis'].dt.year

    # Count the number of storms per year
    storms_per_year = data.groupby('Year').size()

    # Plotting the number of storms per year
    # plt.figure(figsize=(12, 6))
    storms_per_year.plot(kind='line', color='blue', marker='o')
    plt.title('Number of Storms Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Storms')
    plt.grid(True)
    # plt.show()
    st.line_chart(storms_per_year)
    st.caption('The line chart above shows the number of storms recorded each year. This visualization helps in '
               'understanding the frequency of storms over the years covered in the dataset.')


def storm_grades_distribution():
    st.subheader('Distribution of Storm Grades', divider='blue')
    # Counting the number of occurrences of each storm grade
    storm_grades_count = data['Grade'].value_counts()

    # Plotting the distribution of storm grades
    plt.figure(figsize=(12, 6))
    sns.barplot(x=storm_grades_count.index, y=storm_grades_count.values, palette="viridis")
    plt.title('Distribution of Storm Grades')
    plt.xlabel('Storm Grade')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    # plt.show()
    st.bar_chart(storm_grades_count)
    st.caption('The bar chart above illustrates the distribution of different storm grades in the dataset. Each bar '
               'represents a specific grade of storm, such as "Tropical Depression (TD)" or "Tropical Cyclone of TS '
               'intensity or higher", and the height of the bar indicates the frequency of each grade.')


def storm_distribution():
    st.subheader('Geographic Distribution of Storms (Color-coded by Central Pressure)', divider='blue')
    # Scatter plot of storm centers, color-coded by central pressure
    plt.figure(figsize=(12, 8))
    plt.scatter(data['Longitude of the center'], data['Latitude of the center'],
                c=data['Central pressure'], alpha=0.5, cmap='viridis')
    plt.colorbar(label='Central Pressure (hPa)')
    plt.title('Geographic Distribution of Storms (Color-coded by Central Pressure)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    # plt.show()

    # st.scatter_chart(data, x='Longitude of the center', y='Latitude of the center')
    fig = px.scatter(data, x='Longitude of the center', y='Latitude of the center',
                     color='Central pressure', color_continuous_scale='Viridis',
                     labels={'color': 'Central Pressure (hPa)'})

    st.plotly_chart(fig)
    st.caption("The scatter plot above displays the geographic distribution of storms, color-coded by central "
               "pressure. In this visualization, each point represents a storm's location, with the color indicating "
               "its central pressure. Lower central pressures, often indicative of more intense storms, "
               "are represented by colors towards the darker end of the spectrum.")
    st.caption("This analysis helps to identify if certain geographic regions are associated with more intense storms "
               "based on central pressure. Darker areas on the map suggest regions where storms with lower central "
               "pressures, and potentially higher intensities, are more common.")


def central_pressure_wind_speed():
    st.subheader('Central Pressure vs. Maximum Sustained Wind Speed', divider='blue')
    # Scatter plot with regression line for Central Pressure vs. Maximum Sustained Wind Speed
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Central pressure', y='Maximum sustained wind speed', data=data, scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'})
    plt.title('Central Pressure vs. Maximum Sustained Wind Speed')
    plt.xlabel('Central Pressure (hPa)')
    plt.ylabel('Maximum Sustained Wind Speed (knots)')
    plt.grid(True)
    # plt.show()
    fig = px.scatter(data, x='Central pressure', y='Maximum sustained wind speed',
                     trendline='ols',  # OLS 回归线
                     trendline_color_override='red',  # 设置回归线颜色
                     labels={
                         'Central pressure': 'Central Pressure (hPa)',
                         'Maximum sustained wind speed': 'Maximum Sustained Wind Speed (knots)'
                     })

    # 更新图表标题
    fig.update_layout(
                      title=' ',  # 居中标题
                      xaxis_title='Central Pressure (hPa)',
                      yaxis_title='Maximum Sustained Wind Speed (knots)')

    # 在Streamlit中显示图表
    st.plotly_chart(fig)
    st.caption("The scatter plot with a regression line illustrates the relationship between central pressure and "
               "maximum sustained wind speed. Each point represents a storm, with its central pressure on the x-axis "
               "and wind speed on the y-axis. The regression line helps to visualize the overall trend and potential "
               "correlation between these two variables.")


def landfall_number():
    st.subheader('Number of Landfall Storms Per Year', divider='blue')
    # Assuming '#' indicates landfall
    landfall_storms = data[data['Indicator of landfall or passage'] == '#']

    # Counting the number of landfall storms per year
    landfall_storms_per_year = landfall_storms.groupby('Year').size()

    # Plotting the number of landfall storms per year
    plt.figure(figsize=(12, 6))
    landfall_storms_per_year.plot(kind='line', color='darkred', marker='o')
    plt.title('Number of Landfall Storms Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Landfall Storms')
    plt.grid(True)
    # plt.show()
    fig = px.line(landfall_storms_per_year, x=landfall_storms_per_year.index, y=landfall_storms_per_year,
                  labels={'y': 'Number of Landfall Storms', 'x': 'Year'},
                  markers=True,  # 添加点标记
                  line_shape='linear')  # 线形

    # 更新图表布局
    fig.update_layout(title='',
                      title_x=0.5,  # 标题居中
                      xaxis_title='Year',
                      yaxis_title='Number of Landfall Storms')  # 背景色

    # 更新线条样式
    fig.update_traces(line=dict(color='blue', width=2))

    # 在Streamlit中显示图表
    st.plotly_chart(fig)
    st.caption("The line chart above shows the number of storms that made landfall each year. This visualization "
               "helps in understanding how the frequency of landfall storms has changed over the years.")


def storm_distribution_by_grades():
    st.subheader('Geographic Distribution of Storms by Grade', divider='blue')
    key_grades = ['Tropical Depression (TD)', 'Tropical Cyclone of TS intensity or higher']
    selected_storms = data[data['Grade'].isin(key_grades)].copy()  # 创建副本

    # 映射风暴等级到数值
    grade_mapping = {'Tropical Depression (TD)': 1,
                     'Tropical Cyclone of TS intensity or higher': 2}
    selected_storms['Grade Numerical'] = selected_storms['Grade'].map(grade_mapping)

    # 创建散点图
    fig = px.scatter(selected_storms, x='Longitude of the center', y='Latitude of the center',
                     color='Grade Numerical',
                     color_continuous_scale='viridis',
                     labels={'Grade Numerical': 'Storm Grade (1: TD, 2: TS or higher)'})

    # 更新图表布局
    fig.update_layout(title='',
                      title_x=0.5,
                      xaxis_title='Longitude',
                      yaxis_title='Latitude')

    # 在 Streamlit 中显示图表
    st.plotly_chart(fig)

    st.caption("The scatter plot above displays the geographic distribution of storms, color-coded by their grade. In "
               "this visualization, each point represents a storm's location, with the color indicating its grade. "
               "I've focused on two key grades for clarity: Tropical Depression (TD) and Tropical Cyclone of TS "
               "intensity or higher.")
    st.caption("The color coding (1: TD, 2: TS or higher) helps to identify if certain geographic regions are more "
               "associated with intense storms (TS or higher) as opposed to less intense ones (TD).")


def storm_category_frequency():
    st.subheader('Frequency of Storm Categories Over Time', divider='blue')
    storm_categories = data.groupby(['Year', 'Grade']).size().unstack().fillna(0)

    # 假设 storm_categories 是你按年份和风暴等级分组的数据
    # storm_categories = ...

    # 创建一个堆积面积图
    fig = go.Figure()
    for grade in storm_categories.columns:
        fig.add_trace(go.Scatter(
            x=storm_categories.index,
            y=storm_categories[grade],
            name=grade,
            stackgroup='one',  # 堆积面积图
            mode='none'  # 无线条和标记
        ))

    fig.update_traces(line=dict(width=10))
    # 更新图表布局
    fig.update_layout(title='',
                      xaxis_title='Year',
                      yaxis_title='Number of Storms',
                      legend_title='Storm Grade')

    # 在 Streamlit 中显示图表
    st.plotly_chart(fig)
    st.caption("This visualization helps in understanding the temporal distribution and trends of various types of "
               "storms. For example, we can observe if certain storm categories became more or less frequent over the"
               " years.")


def main():
    st.image(
        'https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
        '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Tropical Cyclone Size Data Analysis")

    storm_distribution()

    storm_distribution_by_grades()

    number_of_storms()

    storm_grades_distribution()

    central_pressure_wind_speed()

    landfall_number()

    storm_category_frequency()


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Track Data Analysis",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
