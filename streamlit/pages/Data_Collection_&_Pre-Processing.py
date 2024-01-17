import os

import streamlit as st
import pandas as pd




def get_dataset():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'tropical_cyclone_size.csv')

    df = pd.read_csv(file_path)

    return df


def get_jma_dataset():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'RSMC_Best_Track_Data.csv')

    df = pd.read_csv(file_path)

    return df


def get_cma_dataset():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    parent1_dir = os.path.abspath(os.path.join(parent_dir, '..'))

    file_path = os.path.join(parent1_dir, 'resource', 'CMA_Best_Track_Data.csv')

    df = pd.read_csv(file_path)

    return df

def size_data():
    st.subheader("For CMA Tropical Cyclone Size Dataset", divider='blue')
    st.caption("1980 - 2016 tropical cyclones size data in the north-west Pacific Ocean fom CMA")
    st.write('Data description table:')
    st.markdown(
        '''
        | Field   | Description                                      |
|---------|--------------------------------------------------|
| YYYY    | Year                                             |
| NN      | Tropical cyclone number, including tropical depressions |
| MMDDHH  | 2-digit month, 2-digit day, 2-digit hour (UTC)   |
| LAT     | Latitude of the tropical cyclone center, IBTrACS v03r02 |
| LONG    | Longitude of the tropical cyclone center, IBTrACS v03r02 |
| PRS     | Minimum central pressure of the tropical cyclone, IBTrACS v03r02 |
| WND     | Maximum sustained wind speed near the tropical cyclone center, obtained from IBTrACS v03r02 |
| SiR34   | Scale of the tropical cyclone (km, based on the 34-knot wind radius) |
| SATSer  | Satellite used for inversion, including GOES-1 to 13, Meteosat-2 to 9, GMS-1 to 5, MTSAT-1R, MTS-2, and FY2-C/E |   
        '''
    )
    st.caption('1. Download dataset from: https://tcdata.typhoon.org.cn/tcsize.html')
    st.caption("2. Set the columns name: 'Time', 'Latitude', 'Longitude', 'Pressure', 'Wind Speed', 'SiR34', 'SATSer'")
    st.caption("3. Make time-related columns as Time format")
    st.caption("4. Adjust unique identifier as  international cyclone number")
    st.dataframe(get_dataset().head(5), hide_index=True)
    st.caption('Full code available on Kaggle: https://www.kaggle.com/code/chriszhengao/data-processing-for-tropical-cyclone-size-dataset')


def jma_track_data():
    st.subheader("For JMA Tropical Cyclone Best Track Dataset", divider='blue')
    st.caption("Tropical cyclone information provided by Japan Meteorological Agency(JMA) from 1951 - 2023")
    st.caption("Data URL: https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/trackarchives.html")
    st.write('Raw Data:')
    st.code('''
66666 5102   37      5102 0 6              GEORGIA              20130319        
51031806 002 2 057 1583 1002                                                    
51031812 002 2 060 1594 1002                                                    
51031818 002 2 064 1604 1000                                                    
51031900 002 2 067 1614 1000                                                    
51031906 002 2 070 1625  998                                                        
...\n
23060218 002 3 297 1352  980     045     00000 0000 30300 0210                  
23060300 002 6 306 1378  984     000                                            
23060306 002 6 324 1408  990     000                                            
23060312 002 6 337 1447  996     000                                            
66666 2303  044 0005 2303 1 0               GUCHOL              20230906        
23060600 002 2 128 1350 1004     000                                            
23060606 002 2 131 1351 1002     000                                            
    ''')

    st.subheader('Format Explanation')

    st.write("1. Header Lines for Each Tropical Cyclone")
    st.code('''
----5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80
::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|
AAAAA BBBB  CCC DDDD EEEE F G HHHHHHHHHHHHHHHHHHHH              IIIIIIII
    
    ''')
    st.caption('  Sample for Header Lines:')
    st.code('''
66666 9119  150 0045 9119 0 6             MIRREILE              19920701
          

AAAAA    5 columns     <Indicator> '66666'
BBBB     4 columns     <International number ID> 
                          Last two digits of calendar year followed by 2-digit serial 
                          number ID of the storm of Tropical Storm (TS) intensity or 
                          greater
CCC      3 columns     <Number of data lines>
DDDD     4 columns     <Tropical cyclone number ID>
                          Serial number ID of the storm of intensity with maximum 
                          sustained wind speed of 28 kt (near gale) or  greater
EEEE     4 columns     <International number ID> Replicate BBBB
F        1 column      <Flag of the last data line> 
                          0 : Dissipation
                          1 : Going out of the responsible area of RSMC Tokyo-Typhoon Center
G        1 column      <Difference between the time of the last data and the time of 
                          the final analysis> Unit : hour
H...H   20 columns     <Name of the storm>
I...I    8 columns     <Date of the latest revision>
    ''')


    st.write("2. Data Lines for each Tropical Cyclone")
    st.code('''
----5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80
::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|
AAAAAAAA BBB C DDD EEEE FFFF     GGG     HIIII JJJJ KLLLL MMMM         P
    ''')

    st.caption("Sample for Data Lines:")
    st.code('''

91092706 002 5 325 1293  935     095     30180 0140 30400 0260         #
          
A...A    8 columns     <Time of analysis> yymmddhh (UTC)
BBB      3 columns     <Indicator> '002'
C        1 column      <Grade> 1 : Not used
                                     2 : Tropical Depression (TD)
                                     3 : Tropical Storm (TS)
                                     4 : Severe Tropical Storm (STS)
                                     5 : Typhoon (TY)
                                     6 : Extra-tropical Cyclone (L)
                                     7 : Just entering into the responsible area of
                                         RSMC Tokyo-Typhoon Center
                                     8 : Not used
                                     9 : Tropical Cyclone of TS intensity or higher
DDD      3 columns     <Latitude of the center> Unit : 0.1 degree
EEEE     4 columns     <Longitude of the center> Unit : 0.1 degree
FFFF     4 columns     <Central pressure> Unit : hPa
GGG      3 columns     <Maximum sustained wind speed> Unit : knot (kt)
H        1 column      <Direction of the longest radius of 50kt winds or greater>
                          0 : No direction (Longest radius of 50kt winds is 0)
                          1 : Northeast (NE)
                          2 : East (E)
                          3 : Southeast (SE)
                          4 : South (S)
                          5 : Southwest (SW)
                          6 : West (W)
                          7 : Northwest (NW)
                          8 : North (N)
                          9 : (symmetric circle)
IIII     4 columns     <The longest radius of 50kt winds or greater>
                          Unit : nautical mile (nm)
JJJJ     4 columns     <The shortest radius of 50kt winds or greater>
                          Unit : nautical mile (nm)
K        1 column      <Direction of the longest radius of 30kt winds or greater>
                          0 : No direction (Longest radius of 30kt winds is 0)
                          1 : Northeast (NE)
                          2 : East (E)
                          3 : Southeast (SE)
                          4 : South (S)
                          5 : Southwest (SW)
                          6 : West (W)
                          7 : Northwest (NW)
                          8 : North (N)
                          9 : (symmetric circle)
LLLL     4 columns     <The longest radius of 30kt winds or greater>
                          Unit : nautical mile (nm)
MMMM     4 columns     <The shortest radius of 30kt winds or greater>
                          Unit : nautical mile (nm)
P        1 column      <Indicator of landfall or passage>
                          Landfall or passage over the Japanese islands occurred within 
                          one hour after the time of the analysis with this indicator.

    ''')
    st.write("3. Convert into CSV file")
    st.caption("1. Merge Header Lines and Data Lines")
    st.caption("2. Assign proper column names")
    st.caption("3. Dropping unnecessary columns")
    st.caption("4. Adjust data types")
    st.caption("5. Scale the data into real-world scale")
    st.dataframe(get_jma_dataset().sample(5), hide_index=True)
    st.caption(
        'Full code available on Kaggle: https://www.kaggle.com/code/chriszhengao/data-processing-for-rsmc-tokyo-best-track-data#Format-of-RSMC-Best-Track-Data')

def cma_track_data():
    st.subheader("For CMA Tropical Cyclone Best Track Dataset", divider='blue')
    st.caption("Tropical cyclone information provided by China Meteorological Agency(CMA) from 1949 - 2022")
    st.caption("Data URL: https://tcdata.typhoon.org.cn/zjljsjj.html")
    st.write('Data Files:')
    st.caption("For CMA data, it's categorized by year, stored in numbers of different text files.")
    st.code('''
CH1949BST.txt
CH1949BST.txt
CH1949BST.txt
CH1949BST.txt
...
CH2020BST.txt
CH2021BST.txt
CH2022BST.txt
    ''')
    st.write("Raw Data:")
    st.code('''
66666 2201   45 0001 2201 0 6 Malakas                            20230327
2022040700 1  39 1485 1002      13
2022040706 1  40 1476 1002      13
2022040712 1  42 1470 1000      15
2022040718 1  45 1464 1000      15
...
66666 2225   14 0029 2225 0 6 Pakhar                             20230327
2022121000 1 143 1252 1004      13
2022121006 1 148 1247 1004      13
2022121012 1 156 1247 1004      13
    ''')
    st.subheader('Format Explanation')
    st.write('1. Header Lines for Each Tropical Cyclone')
    st.code('''
    ----5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80
    ::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|
    AAAAA BBBB  CCC DDDD EEEE F G HHHHHHHHHHHHHHHHHHHH              IIIIIIII

        ''')
    st.caption('Explanation of the format')
    st.markdown('''
| Field | Length | Description |
|-------|--------|-------------|
| AAAAA | 5      | Classification flag; '66666' indicates the best track data. |
| BBBB  | 4      | International number; the last two digits of the year + two-digit serial number. |
| CCC   | 3      | Number of rows in the track data record. |
| DDDD  | 4      | Serial number of tropical cyclones, including tropical depressions. |
| EEEE  | 4      | China's identification number for tropical cyclones. |
| F     | 1      | Tropical cyclone termination record: 0 for dissipation, 1 for moving out of the responsibility area of the Western Pacific Typhoon Committee, 2 for merging, 3 for quasi-stationary. |
| G     | 1      | Hourly interval between each row of the path; before 2017, it was 6 hours, starting from 2017, individual cases with a 3-hour encryption record are marked as 3, and others remain 6. |
| H...H | 20     | English name of the tropical cyclone; "(-1)n" is added after the name to indicate the secondary center and its serial number. |
| I...I | 8      | Date on which the dataset is formed. |
    ''')

    st.write("2. Data Lines for each Tropical Cyclone")
    st.code('''
    ----5   10   15   20   25   30   35   40
    ::::+::::|::::+::::|::::+::::|::::+::::|
    YYYYMMDDHH I LAT LONG PRES      WND  OWD
        ''')
    st.caption('Explanation of the format')
    st.markdown('''
| Field          | Description |
|----------------|-------------|
| YYYYMMDDHH     | Date and time in UTC: YYYY year, MM month, DD day, HH hour. |
| I              | Intensity marker based on the average wind speed within 2 minutes around the exact time point. Refer to the National Standard "Tropical Cyclone Grades" (GB/T 19201-2006): <br> 0 - Weaker than Tropical Depression (TD), or intensity unknown. <br> 1 - Tropical Depression (TD, 10.8-17.1 m/s). <br> 2 - Tropical Storm (TS, 17.2-24.4 m/s). <br> 3 - Severe Tropical Storm (STS, 24.5-32.6 m/s). <br> 4 - Typhoon (TY, 32.7-41.4 m/s). <br> 5 - Severe Typhoon (STY, 41.5-50.9 m/s). <br> 6 - Super Typhoon (SuperTY, ≥51.0 m/s). <br> 9 - Extratropical transition, the first digit indicates the completion of the transition. |
| LAT            | Latitude (0.1°N). |
| LONG           | Longitude (0.1°E). |
| PRES           | Central minimum pressure (hPa). |
| WND            | 2-minute average maximum sustained wind speed near the center (MSW, m/s). WND=9 indicates MSW &lt; 10 m/s, WND=0 indicates missing data. |
| OWD            | 2-minute average wind speed (m/s) with two cases: <br> (a) For tropical cyclones making landfall in China, it represents the wind speed of coastal strong winds. <br> (b) When a tropical cyclone is in the South China Sea, it represents the maximum wind speed within a range of 300-500 km from the center. |   

    ''')

    st.write('3. Convert into CSV file')
    st.caption('1. Merge the text files together')
    st.caption('2. According to the formatting guideline, transfer files into CSV file')
    st.caption('3. Adjust the format of attributes')
    st.caption('4. Adjust the scale of values, to fit the real world scale')
    st.dataframe(get_cma_dataset().sample(5), hide_index=True)
    st.caption('Full code available on Kaggle: https://www.kaggle.com/code/chriszhengao/data-processing-for-cma-best-track-data')





def main():
    st.image('https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
             '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Data Collection & Pre-processing")

    st.write(
        "The process of data collection and pre-processing."
    )

    size_data()

    jma_track_data()

    cma_track_data()

if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Data Processing",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()