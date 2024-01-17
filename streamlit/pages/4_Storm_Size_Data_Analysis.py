import os

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
# import geopandas as gpd
# from shapely.geometry import Point


def get_dataset_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    # 构建正确的文件路径
    file_path = os.path.join(parent1_dir, 'resource', 'tropical_cyclone_size.csv')
    return file_path


data = pd.read_csv(get_dataset_path())


def size_by_year():
    st.subheader('Average Tropical Cyclone Size (SiR34) Over Years', divider='blue')
    data['Time'] = pd.to_datetime(data['Time'])

    # Grouping data by year to see the yearly trend of SiR34
    data['Year'] = data['Time'].dt.year
    yearly_sir34 = data.groupby('Year')['SiR34'].mean()

    # Plotting the yearly trend of SiR34
    plt.figure(figsize=(12, 6))
    yearly_sir34.plot(kind='line', marker='o')
    plt.title('Yearly Average SiR34 Trend')
    plt.xlabel('Year')
    plt.ylabel('Average SiR34 (km)')
    plt.grid(True)
    # st.pyplot(plt)
    st.line_chart(yearly_sir34)
    st.caption('The line graph above illustrates the average tropical cyclone size (SiR34, measured in kilometers) '
               'over the years. This visualization helps in identifying any long-term trends in the average size of '
               'tropical cyclones.')


def pressure_wind_speed():
    st.subheader('Relationship between Atmospheric Pressure and Wind Speed in Tropical Cyclones', divider='blue')
    # Scatter plot of Pressure vs Wind Speed
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Pressure', y='Wind Speed')
    plt.title('Relationship between Atmospheric Pressure and Wind Speed in Tropical Cyclones')
    plt.xlabel('Atmospheric Pressure (hPa)')
    plt.ylabel('Wind Speed (knots)')
    plt.grid(True)
    plt.show()
    # st.pyplot(plt)
    st.scatter_chart(data, x='Pressure', y='Wind Speed')
    st.caption('The scatter plot above displays the relationship between atmospheric pressure (in hPa) and wind speed '
               '(in knots) for the tropical cyclones in the dataset. In general, a trend where lower pressures '
               'correspond to higher wind speeds is typical in cyclones, indicating more intense storm conditions.')


def wind_speed_size():
    st.subheader('Relationship between Cyclone Size (SiR34) and Wind Speed', divider='blue')
    # Scatter plot of SiR34 (Cyclone Size) vs Wind Speed
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='SiR34', y='Wind Speed')
    plt.title('Relationship between Cyclone Size (SiR34) and Wind Speed')
    plt.xlabel('Cyclone Size (SiR34 in km)')
    plt.ylabel('Wind Speed (knots)')
    plt.grid(True)
    # plt.show()
    st.scatter_chart(data, x='SiR34', y='Wind Speed')
    st.caption('The scatter plot above illustrates the relationship between cyclone size (SiR34, measured in '
               'kilometers) and wind speed (in knots). This visualization helps in understanding whether there is a '
               'correlation between the size of the cyclone and the intensity of its wind speed.')


def frequency_year():
    st.subheader('Frequency of Tropical Cyclones by Year', divider='blue')
    # Count the number of cyclone occurrences each year
    cyclone_frequency_by_year = data.groupby('Year').size().reset_index(name='Count')

    # Plotting the frequency of cyclones over years
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cyclone_frequency_by_year, x='Year', y='Count', color='skyblue')
    plt.title('Frequency of Tropical Cyclones by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Cyclone Occurrences')
    plt.xticks(rotation=45)
    plt.grid(True)
    # plt.show()
    st.bar_chart(cyclone_frequency_by_year, x='Year', y='Count')
    st.caption('The bar chart above shows the frequency of tropical cyclone occurrences by year. This visualization '
               'helps in identifying any trends or patterns in the frequency of cyclones over the years, '
               'such as increases or decreases in cyclonic activity.')


def season_distribution():
    st.subheader('Seasonal Distribution of Tropical Cyclones', divider='blue')
    # Extracting month from the date for seasonal analysis
    data['Month'] = data['Time'].dt.month

    # Count the number of cyclones per month
    cyclone_frequency_by_month = data.groupby('Month').size().reset_index(name='Count')

    # Plotting the frequency of cyclones by month
    plt.figure(figsize=(10, 6))
    sns.barplot(data=cyclone_frequency_by_month, x='Month', y='Count', palette='viridis')
    plt.title('Seasonal Distribution of Tropical Cyclones')
    plt.xlabel('Month')
    plt.ylabel('Number of Cyclone Occurrences')
    plt.xticks(ticks=range(0, 12),
               labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    # plt.show()
    st.bar_chart(cyclone_frequency_by_month, x='Month', y='Count',)
    st.caption('The bar chart above displays the seasonal distribution of tropical cyclones, with the number of '
               'cyclone occurrences plotted against each month of the year. This visualization helps identify which '
               'months typically experience higher cyclonic activity, indicating the peak of the cyclone season.')


def avg_size_over_time():
    st.subheader('Average Cyclone Size Over Time from Cyclone Onset', divider='blue')
    # Step 1: Determine the start time for each cyclone
    start_times = data.groupby('Cyclone Number')['Time'].min().to_frame('Start Time')

    # Merging the start time back into the original dataframe
    cyclone_data = data.merge(start_times, on='Cyclone Number')

    # Step 2: Calculate the time elapsed since the start for each observation
    cyclone_data['Time Elapsed'] = (cyclone_data['Time'] - cyclone_data[
        'Start Time']).dt.total_seconds() / 3600  # Time in hours

    # Step 3: Group the data by cyclone and elapsed time intervals (e.g., every 6 hours)
    cyclone_data['Time Interval'] = (cyclone_data['Time Elapsed'] // 6) * 6  # Creating 6-hour intervals

    # Step 4: Calculate the average size for these intervals
    average_size_by_interval = cyclone_data.groupby(['Cyclone Number', 'Time Interval'])['SiR34'].mean().reset_index()

    # Calculating the overall average size for each time interval across all cyclones
    overall_average_size_by_interval = average_size_by_interval.groupby('Time Interval')['SiR34'].mean().reset_index()

    # Plotting the line chart
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=overall_average_size_by_interval, x='Time Interval', y='SiR34')
    plt.title('Average Cyclone Size Over Time from Cyclone Onset')
    plt.xlabel('Time Since Cyclone Onset (hours)')
    plt.ylabel('Average Cyclone Size (SiR34 in km)')
    plt.grid(True)
    # plt.show()
    st.line_chart(overall_average_size_by_interval, x='Time Interval', y='SiR34')
    st.caption("The line chart above illustrates the average size of cyclones (SiR34, in kilometers) over time, "
               "starting from the onset of each cyclone. The x-axis represents time since the cyclone's onset in "
               "hours, and the y-axis shows the average cyclone size. Each point on the line represents the average "
               "size of all cyclones at that particular time interval after their onset.")
    st.caption("This visualization helps in understanding how the size of cyclones typically evolves over time. It "
               "can provide insights into the lifecycle of cyclones, such as periods of intensification or weakening.")


def main():
    st.image(
        'https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
        '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Tropical Cyclone Size Data Analysis")

    size_by_year()

    pressure_wind_speed()

    wind_speed_size()

    frequency_year()

    season_distribution()

    avg_size_over_time()


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Storm Size Data Analysis",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()


