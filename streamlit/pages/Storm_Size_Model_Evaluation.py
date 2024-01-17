import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go


def default_mae():
    st.subheader("Default Hyper Parameter", divider='blue')
    # Default hyper parameter
    data = [
        [1, "Linear Regression", 27.622],
        [2, "Ridge Regression", 27.722],
        [3, "Lasso Regression", 27.622],
        [4, "Random Forest", 26.697],
        [5, "Elastic Net", 27.624],
        [6, "LightGBM", 27.571],
        [7, 'XGBoost', 26.977],
        [8, "Gradient Boosting", 27.023]
    ]

    columns = ["No.", "Model", "MAE (Mean Absolute Error)"]

    df = pd.DataFrame(data, columns=columns)

    fig = px.bar(df, y='Model', x='MAE (Mean Absolute Error)', orientation='h',
                 text='MAE (Mean Absolute Error)',  # 显示 MAE 值
                 color_discrete_sequence=['skyblue'])  # 设置条形图颜色

    # 反转 y 轴的顺序
    fig.update_layout(yaxis={'categoryorder': 'total descending'})

    # 调整文本的显示位置和样式
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 更新其它图表布局选项
    fig.update_layout(
        title='Comparison of K-Fold MAE(average of K = 5) for Different Regression Models',
        xaxis_title='Mean Absolute Error (MAE)',
        yaxis_title='Model',
        showlegend=False
    )

    st.plotly_chart(fig)

    return df


def optimized_mae():
    st.subheader("Model Tuning & Optimization", divider='blue')
    st.write("Use Bayesian Optimization with K-Fold Validation, K = 5.")
    # Default hyper parameter
    gbr_optimized()

    lgbm_optimized()

    xgb_optimized()

    df_after_tuning()


def gbr_optimized():
    st.write("1. Gradient Boosting")
    params = [
        ['Gradient Boosting', 26.77, 0.4874, 0.01218, 7.981, 7.775, 19.72, 473.0, 0.7666, 26.80, 26.69]
    ]

    col = ["Model", "MAE iterate", "alpha", "learning rate", 'max_depth',
               'min_samples_split', 'min_samples_leaf',
               'n_estimators', 'subsample', 'MAE on testing set', 'Cross Validate MAE (5 AVG)'
               ]

    df = pd.DataFrame(params, columns=col)

    st.dataframe(df, hide_index=True)


def lgbm_optimized():
    st.write("2. LightGBM")
    params = [
        ['LightGBM', 28.8314, 1.0, 0.5, 0.01, 3.0, 20.0, 507.0322, 0.5]
    ]

    col = ["Model", "MAE iterate", "alpha", "colsample_bytree", 'learning_rate',
           'max_depth', 'min_child_samples', 'n_estimators', 'subsample',
           ]

    df = pd.DataFrame(params, columns=col)

    st.dataframe(df, hide_index=True)


def xgb_optimized():
    st.write("3. XGBoost")
    params = [
        ['XGBoost', 28.89, 0.3731, 0.9985, 0.9523, 0.1265, 3.236, 2.049, 110.4, 0.9879, 27.492]
    ]

    col = ["Model", "MAE iterate", "alpha", "colsample_bytree", 'gamma','learning_rate',
           'max_depth', 'min_child_samples', 'n_estimators', 'subsample',
           'MAE on tesing set'
           ]

    df = pd.DataFrame(params, columns=col)

    st.dataframe(df, hide_index=True)


def df_after_tuning():
    st.subheader('After Optimization', divider='blue')
    data = [
        [1, "Linear Regression", 27.622],
        [2, "Ridge Regression", 27.722],
        [3, "Lasso Regression", 27.622],
        [4, "Random Forest", 26.697],
        [5, "Elastic Net", 27.624],
        [6, "LightGBM", 27.571],
        [7, 'XGBoost', 26.977],
        [8, "Gradient Boosting", 27.023],
        [9, "Gradient Boosting(B)", 26.8],
        [10, 'LightGBM(B)', 28.8314],
        [11, 'XGBoost(B)', 27.492]
    ]

    columns = ["No.", "Model", "MAE (Mean Absolute Error)"]

    df = pd.DataFrame(data, columns=columns)

    fig = px.bar(df, y='Model', x='MAE (Mean Absolute Error)', orientation='h',
                 text='MAE (Mean Absolute Error)',  # 显示 MAE 值
                 color_discrete_sequence=['skyblue'])  # 设置条形图颜色

    # 反转 y 轴的顺序
    fig.update_layout(yaxis={'categoryorder': 'total descending'})

    # 调整文本的显示位置和样式
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 更新其它图表布局选项
    fig.update_layout(
        title='Comparison of K-Fold MAE(average of K = 5) for Different Regression Models',
        xaxis_title='Mean Absolute Error (MAE)',
        yaxis_title='Model',
        showlegend=False
    )

    st.plotly_chart(fig)

    return df


def model_ensemble():
    st.subheader("Ensemble Learning", divider='blue')
    st.write("Stacking models")

    gbr_rf_lr()

    gbr_rf_xgb_lr()


def gbr_rf_lr():
    st.write("1. Stacking Gradient Boosting(B) + Random Forest + Linear Regression")
    data = [
        ['Stacking Model 1', 'Gradient Boosting(B)', 'Random Forest', 'Linear Regression',
         26.4751, 26.4157]
    ]

    col = ['Model', 'Base model 1', 'Base model 2', 'Final estimator', 'MAE on testing set',
           'MAE K-fold Validation(5 AVG)']

    df = pd.DataFrame(data, columns=col)

    st.dataframe(df, hide_index=True)


def gbr_rf_xgb_lr():
    st.write("2. Stacking Gradient Boosting(B) + Random Forest + XGBoost + Linear Regression")
    data = [
        ['Stacking Model 2', 'Gradient Boosting(B)', 'Random Forest',
         'XGBoost', 'Linear Regression',
         26.4617, 26.4288]
    ]

    col = ['Model', 'Base model 1', 'Base model 2', 'Base model 3', 'Final estimator', 'MAE on testing set',
           'MAE K-fold Validation(5 AVG)']

    df = pd.DataFrame(data, columns=col)

    st.dataframe(df, hide_index=True)


def final_ranking():
    st.subheader("Final MAE Ranking", divider='blue')
    st.write("Including all the tested models")
    data = [
        [1, "Linear Regression", 27.622],
        [2, "Ridge Regression", 27.722],
        [3, "Lasso Regression", 27.622],
        [4, "Random Forest", 26.697],
        [5, "Elastic Net", 27.624],
        [6, "LightGBM", 27.571],
        [7, 'XGBoost', 26.977],
        [8, "Gradient Boosting", 27.023],
        [9, "Gradient Boosting(B)", 26.8],
        [10, 'LightGBM(B)', 28.8314],
        [11, 'XGBoost(B)', 27.492],
        [12, 'Stacking Model 1', 26.415],
        [13, 'Stacking Model 2', 26.428]
    ]

    columns = ["No.", "Model", "MAE (Mean Absolute Error)"]

    df = pd.DataFrame(data, columns=columns)

    fig = px.bar(df, y='Model', x='MAE (Mean Absolute Error)', orientation='h',
                 text='MAE (Mean Absolute Error)',  # 显示 MAE 值
                 color_discrete_sequence=['skyblue'])  # 设置条形图颜色

    # 反转 y 轴的顺序
    fig.update_layout(yaxis={'categoryorder': 'total descending'})

    # 调整文本的显示位置和样式
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 更新其它图表布局选项
    fig.update_layout(
        title='Comparison of K-Fold MAE(average of K = 5) for Different Regression Models',
        xaxis_title='Mean Absolute Error (MAE)',
        yaxis_title='Model',
        showlegend=False
    )

    st.plotly_chart(fig)

    return df


def main():
    st.image('https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
             '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Model Evaluation for Predicting the Storm Size - Ensemble Learning")

    st.write(
        "Tested regression models and ensemble learning."
    )

    default_mae()

    optimized_mae()

    model_ensemble()

    final_ranking()


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Storm Size Model Evaluation",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
