import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

def default_params():
    st.subheader("Default Hyper Parameter", divider='blue')
    tree_model_data = {
        'LightGBM': 2.84210729935653,
        'XGBoost': 2.9110839066300294,
        'Gradient Boosting Regressor': 3.058142032327553,
        'Random Forest': 3.577819693552305,
        'CatBoost': 4.1288838393165985
    }

    # 线性模型数据
    linear_model_data = {
        'Ridge Regression': 2.363364115303271,
        'Bayesian Ridge Regression': 2.3648845682876027,
        'Linear Regression': 2.3696121160329646,
        'Lasso Regression': 2.4184638097309645,
        'Elastic Net': 2.455301149915244
    }


    col1, col2 = st.columns(2)




    # 创建DataFrame
    tree_df = pd.DataFrame(list(tree_model_data.items()), columns=['Model', 'DTW Distance'])

    # 创建柱状图
    fig_tree = px.bar(tree_df, y='Model', x='DTW Distance', orientation='h',
                      text='DTW Distance', color_discrete_sequence=['skyblue'])

    # 反转 y 轴的顺序
    fig_tree.update_layout(yaxis={'categoryorder': 'total descending'})

    # 调整文本的显示位置和样式
    fig_tree.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 更新其它图表布局选项
    fig_tree.update_layout(
        xaxis_title='DTW Distance',
        yaxis_title='Model',
        showlegend=False
    )
    with col1:
        'Tree-based models'
        st.plotly_chart(fig_tree)





    # 创建DataFrame
    linear_df = pd.DataFrame(list(linear_model_data.items()), columns=['Model', 'DTW Distance'])

    # 创建柱状图
    fig_linear = px.bar(linear_df, y='Model', x='DTW Distance', orientation='h',
                        text='DTW Distance', color_discrete_sequence=['skyblue'])

    # 反转 y 轴的顺序
    fig_linear.update_layout(yaxis={'categoryorder': 'total descending'})

    # 调整文本的显示位置和样式
    fig_linear.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 更新其它图表布局选项
    fig_linear.update_layout(
        xaxis_title='DTW Distance',
        yaxis_title='Model',
        showlegend=False
    )

    with col2:
        'Linear Models:'
        st.plotly_chart(fig_linear)


def optimization():
    st.subheader("Bayesian Optimization", divider='blue')
    tree_model_data = {
        'LightGBM': 2.84210729935653,
        'Gradient Boosting(b)': 2.889665385440783,
        'XGBoost': 2.9110839066300294,
        'Gradient Boosting Regressor': 3.058142032327553,
        'Random Forest': 3.577819693552305,
        'CatBoost': 4.1288838393165985,
        'Ridge Regression(B)': 2.352044394882149,
        'Ridge Regression': 2.363364115303271,
        'Bayesian Ridge Regression(B)': 2.3641492906143258,
        'Bayesian Ridge Regression': 2.3648845682876027,
        'Linear Regression': 2.3696121160329646,
        'Lasso Regression(B)': 2.392828124530687,
        'Lasso Regression': 2.4184638097309645,
        'Elastic Net': 2.455301149915244
    }



    # 创建DataFrame
    tree_df = pd.DataFrame(list(tree_model_data.items()), columns=['Model', 'DTW Distance'])

    # 创建柱状图
    fig_tree = px.bar(tree_df, y='Model', x='DTW Distance', orientation='h',
                      text='DTW Distance', color_discrete_sequence=['skyblue'])

    # 反转 y 轴的顺序
    fig_tree.update_layout(yaxis={'categoryorder': 'total descending'})

    # 调整文本的显示位置和样式
    fig_tree.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 更新其它图表布局选项
    fig_tree.update_layout(
        xaxis_title='DTW Distance',
        yaxis_title='Model',
        showlegend=False
    )

    'All models after optimization:'
    st.plotly_chart(fig_tree)


def haversine_loss():
    ...

def ensemble_learning():
    st.subheader("Ensemble Learning - Stacking Models", divider='blue')
    tree_model_data = {
        'BR + LR + XGBoost + Ridge(S)' : 2.2895016179340186,
        'BR + Lasso + RR + EN + LR(S)' : 2.2953221204147067,
        'BR + LR + Ridge + EN + Ridge(S)' : 2.3587328124676588,
        'BR + LR + EN + XGBoost + Ridge(S)' : 2.3083044536018837,
        'BR + LR + LGBM + Ridge(S)' : 2.305323363006786,
        'BR + LR + LGBM + GB + Ridge(S)' : 2.311113746038734,
        'BR + Lasso + Ridge + LGBM + LR(S)' : 2.303272381336092,
        'BR + Lasso + LR + Ridge + LGBM(S)' : 2.4661401456599017,
        'LightGBM': 2.84210729935653,
        'Gradient Boosting(b)': 2.889665385440783,
        'XGBoost': 2.9110839066300294,
        'Gradient Boosting Regressor': 3.058142032327553,
        'Random Forest': 3.577819693552305,
        'CatBoost': 4.1288838393165985,
        'Ridge Regression(B)': 2.352044394882149,
        'Ridge Regression': 2.363364115303271,
        'Bayesian Ridge Regression(B)': 2.3641492906143258,
        'Bayesian Ridge Regression': 2.3648845682876027,
        'Linear Regression': 2.3696121160329646,
        'Lasso Regression(B)': 2.392828124530687,
        'Lasso Regression': 2.4184638097309645,
        'Elastic Net': 2.455301149915244
    }



    # 创建DataFrame
    tree_df = pd.DataFrame(list(tree_model_data.items()), columns=['Model', 'DTW Distance'])

    # 创建柱状图
    fig_tree = px.bar(tree_df, y='Model', x='DTW Distance', orientation='h',
                      text='DTW Distance', color_discrete_sequence=['skyblue'])

    # 反转 y 轴的顺序
    fig_tree.update_layout(yaxis={'categoryorder': 'total descending'})

    # 调整文本的显示位置和样式
    fig_tree.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 更新其它图表布局选项
    fig_tree.update_layout(
        xaxis_title='DTW Distance',
        yaxis_title='Model',
        showlegend=False
    )

    'All models after optimization:'
    st.plotly_chart(fig_tree)


def loss_function():
    st.subheader("Loss Function - Dynamic time warping(DTW) Distance", divider='blue')
    ''
    st.caption('Approaches comparison:')
    '1. _Haversine Distance_: is most directly applicable for point-to-point geographical accuracy in typhoon tracking'
    "2. _MAE_: gives a straightforward average error, useful for general purposes but doesn't emphasize larger errors."
    "3. _MSE_ puts more weight on larger errors, which can be critical in predicting severe weather events like typhoons."
    "4. _DTW_ Distance is unique in considering the entire path of the typhoon and temporal alignment, making it very suitable for evaluating the overall accuracy of a predicted typhoon path."
    ''
    st.subheader('**Dynamic Time Warping (DTW) Distance**:')
    st.write("Application: A measure used in time series analysis. "
             "It aligns two sequences in time, optimizing the match between them.")
    st.write("Characteristics: It can handle sequences that vary in speed and is robust to shifts in the time axis, "
             "making it useful for comparing temporal sequences that may not align perfectly.")
    st.write("Usage in Typhoon Prediction: Highly relevant for comparing the entire predicted path of a typhoon with the actual path, "
             "especially when the timing of the predictions is not perfectly aligned with the actual events.")
    st.image('https://rtavenar.github.io/blog/fig/dtw_path.gif',
             caption='Comparison between DTW and Euclidean distance by Romain Tavenard.)',
             width=500)


def main():
    st.image('https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
             '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Model Evaluation for Predicting the Storm Track - Ensemble Learning")
    'All metrics are using K-Fold = 5 Cross Validation average DTW Distance value'

    loss_function()

    default_params()

    optimization()

    ensemble_learning()


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Storm Track Model Evaluation(Ensemble Learning)",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()