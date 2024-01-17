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
    
    st.write("1. Header Line for Each Tropical Cyclone")
    st.code('''
    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80
::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|::::+::::|
AAAAA BBBB  CCC DDDD EEEE F G HHHHHHHHHHHHHHHHHHHH              IIIIIIII
    
    ''')
    st.write('Sample for Header Line:')
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



def main():
    st.image('https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
             '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Data Collection & Pre-processing")

    st.write(
        "The process of data collection and pre-processing."
    )

    size_data()

    jma_track_data()


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Data Processing",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()