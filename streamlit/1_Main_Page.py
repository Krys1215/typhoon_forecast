import streamlit as st


def show_statistical_data():
    st.write('Tropical cyclones, also known as typhoons or hurricanes, are among the most destructive weather '
             'phenomena. They are intense circular storms that originate over warm tropical oceans, and have maximum '
             'sustained wind speeds exceeding :red[119 kilometres per hour] and heavy rains.')

    st.write('From 1998-2017, storms, including tropical cyclones and hurricanes, were second only to earthquakes in '
             'terms of fatalities, killing :red[233 000] people. During this time, storms also affected an estimated '
             ':red[726]'
             ' million people worldwide, meaning they were :red[injured], :red[made homeless], :red[displaced] or '
             ':red[evacuated] during the'
             'emergency phase of the disaster.')

    st.write('Over the past 30 years the proportion of the world’s population living on cyclone-exposed coastlines '
             'has increased 192 percent, thus raising the risk of mortality and morbidity in the event of a tropical '
             'cyclone.')


def show_metric():
    st.metric(label="2022 Economic Lost", value="$4,000,000,000", delta="$2,110,000,000", delta_color='inverse')
    st.metric(label="2023 Economic Lost", value="$18,500,000,000", delta="$14,500,000,000", delta_color='inverse')


def main():

    st.image('https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
             '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("🌪️ Typhoon Forecast and Tracks Visualization by Using _CNN-LSTM_ and _Ensemble Learning_")

    st.subheader('', divider='blue')
    st.subheader('🧮 Statistical Data of _Typhoon_')

    show_statistical_data()

    st.subheader('', divider='blue')
    st.subheader('💵 Worldwide Economic Lost')

    # ---------------------------------------------------------------
    col_metric, col_damage_pic = st.columns(2)
    with col_metric:
        show_metric()
    with col_damage_pic:
        st.image("https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1500w,f_auto,"
                 "q_auto:best/msnbc/Components/Photos/060815/060815_typhoon_hmed_1p.jpg")


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Main Page",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    main()
