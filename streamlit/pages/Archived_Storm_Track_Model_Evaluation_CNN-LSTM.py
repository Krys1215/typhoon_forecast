import streamlit as st
import pandas as pd
import os
import keras

from PIL import Image
from pathlib import Path


def model_structure():
    st.subheader("Model Structure - CNN-LSTM model", divider='blue')

    structure = '''
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_features_x, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs_x))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss=haversine_loss, optimizer='adam')
    '''

    st.code(structure, language='python')

    st.caption("1. :blue[Conv1D Layer]: This is a one-dimensional convolutional layer. In this layer, your model "
               "learns to"
               "extract features from sequential data using filters (also known as convolution kernels). Each filter "
               "slides over the input data to capture local patterns. This is particularly useful for time-series "
               "data as it can detect patterns that change over time. In your model, you have two of these layers, "
               "each with 64 filters and a kernel size of 3. This setup helps in extracting intricate features from "
               "the data while maintaining the sequential nature.")
    st.caption("2. :blue[MaxPooling1D Layer]: The MaxPooling layer is used to reduce the dimensionality of the output "
               "from the convolutional layers. It does this by taking the maximum value in a specified region ("
               "defined by pool_size). This not only reduces computational requirements but also helps in extracting "
               "more significant features, as it focuses on the most prominent signals in each pooling window.")
    st.caption("3. :blue[Flatten Layer]: The Flatten layer transforms multi-dimensional inputs into a one-dimensional "
               "array. This is a necessary step when transitioning from convolutional or pooling layers to dense "
               "layers (fully connected layers). It essentially \"flattens\" the data, preparing it for the dense "
               "layer that follows.")
    st.caption("4. :blue[RepeatVector Layer]: This layer is used to repeat the input n times (in this case, "
               "n is the number of output features). This is a common technique when bridging feedforward networks "
               "and recurrent networks, especially in sequence-to-sequence models. It prepares the fixed-size output "
               "of the CNN layers for sequential processing in the LSTM layer.")
    st.caption("5. :blue[LSTM Layer]: Long Short-Term Memory (LSTM) layer is a special type of recurrent neural "
               "network suitable for learning long-term dependencies in sequence data. return_sequences=True means "
               "the layer outputs the output at each time step of the sequence, rather than only the output at the "
               "end of the sequence. This is necessary for the subsequent TimeDistributed layer.")
    st.caption("6. :blue[TimeDistributed Layer]: This layer allows applying the same Dense layer (fully connected "
               "layer) separately at each time step. This is a common practice when dealing with sequence data, "
               "especially when you want the network to output a vector at each time step.")
    st.caption("7. :blue[Dense Layer]: A Dense layer is a typical fully connected neural network layer where each "
               "input is connected to each output by weights. In this model, you have two Dense layers, one with 100 "
               "units and a relu activation function, and another with 1 unit (without an activation function) for "
               "the final prediction output.")


def loss_function():
    st.subheader("Loss Function", divider='blue')
    st.write("Custom Loss Functions")
    st.caption("In a standard regression problem, a loss function like the mean square error is typically used. "
               "However, when the problem involves geospatial data, using Haversine's formula gives a more direct "
               "picture of the accuracy of the prediction because it takes into account the curvature of the earth "
               "and the shortest path between two points.")
    st.caption("This loss function is very useful in machine learning, especially when working with geographic data, "
               "as it directly quantifies the difference in geographic distance between the predicted location and "
               "the actual location")

    st.latex(r'''
        a = \sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1) \cdot \cos(\text{lat}_2) \cdot \sin^2\left(\frac{\Delta \text{lon}}{2}\right) \\
        c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right) \\
        d = R \cdot c
        ''')
    code = '''
    def haversine_loss(y_true, y_pred):
        lat_true, lon_true = y_true[:, 1], y_true[:, 2]
        lat_pred, lon_pred = y_pred[:, 0], y_pred[:, 1]
        # Converting degrees to radians
        pi = tf.constant(np.pi)
        lat_true, lon_true, lat_pred, lon_pred = [x * (pi / 180) for x in [lat_true, lon_true, lat_pred, lon_pred]]

        dlat = lat_pred - lat_true
        dlon = lon_pred - lon_true

        a = tf.sin(dlat / 2)**2 + tf.cos(lat_true) * tf.cos(lat_pred) * tf.sin(dlon / 2)**2
        c = 2 * tf.asin(tf.sqrt(a))

        # Mean radius of the Earth in kilometres
        R = 6371.0
        return R * c
    '''

    st.code(code, language='python')


def callbacks_function():
    st.subheader("Callback Functions", divider='blue')
    st.write("1. Early Stopping")
    st.caption("Since our validation loss is decreasing but is not stable,"
               " use early stopping to further increase the validation loss."
               )
    st.code('''
    early_stopper = EarlyStopping(
    monitor='val_loss',  # monitor the validation loss
    patience=20,         # give 20 times without improvements
    verbose=1,           # print the log
    mode='min',          # the target should be minimize
    restore_best_weights=True  # restore to the best mode
    )
    ''')

    st.write("2. Learning Rate Scheduler")
    st.caption('This setup is useful in training deep learning models '
               'to adaptively change the learning rate, potentially leading to better training performance and convergence.')
    st.code('''
    def scheduler(epoch, lr):
    if epoch < 10:          # well remain unchanged when epoch < 10
        return lr
    else:                   # decrease the learning rate when epoch > 10
        return lr * np.exp(-0.1)
    ''')

def experiment1():
    st.subheader("Experiment 1")
    data = [
        [50, '16 * Replicas (8)', 0.0005, 'No', 'No']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L1.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S1.png')


def experiment2():
    st.subheader("Experiment 2")
    data = [
        [50, '6 * Replicas (8)', 0.0005, 'No', 'No']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L2.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S2.png')


def experiment3():
    st.subheader("Experiment 3")
    data = [
        [50, '16 * Replicas (8)', 0.0005, 'Yes', 'No']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L3.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S3.png')


def experiment4():
    st.subheader("Experiment 4")
    data = [
        [100, '16 * Replicas (8)', 0.0005, 'Yes', 'No']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L4.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S4.png')



def experiment5():
    st.subheader("Experiment 5")
    data = [
        [100, '50 * Replicas (8)', 0.0005, 'Yes', 'No']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L5.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S5.png')


def experiment6():
    st.subheader("Experiment 6")
    data = [
        [100, '20 * Replicas (8)', 0.0005, 'Yes', 'No']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L6.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S6.png')


def experiment7():
    st.subheader("Experiment 7")
    data = [
        [100, '20 * Replicas (8)', 0.0005, 'Yes', 'Yes']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L7.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S7.png')


def experiment8():
    st.subheader("Experiment 8")
    data = [
        [100, '20 * Replicas (8)', 0.0005, 'No', 'Yes']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L8.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S8.png')



def experiment9():
    st.subheader("Experiment 9")
    data = [
        [100, '20 * Replicas (8)', 0.0005, 'No', 'Yes']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L9.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S9.png')



def experiment10():
    st.subheader("Experiment 10")
    data = [
        [100, '20 * Replicas (8)', 0.0005, 'Yes, Patience = 20', 'Yes']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L10.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S10.png')

def experiment11():
    st.subheader("Experiment 11")
    data = [
        [100, '20 * Replicas (8)', 0.0005, 'Yes, Patience=20', 'Yes']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L11.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S11.png')


def experiment12():
    st.subheader("Experiment 12")
    data = [
        [100, '20 * Replicas (8)', 0.0005, 'Yes, Patience=20', 'Yes, Epoch>30']
    ]

    columns = ["Epochs",
               "Batch Size",
               "Validation Split Ratio",
               "Early Stopping",
               "LearningRateScheduler"
               ]

    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/L12.png')
    with col2:
        st.image('https://raw.githubusercontent.com/Krys1215/typhoon_forecast/main/streamlit/images/S12.png')




def model_training():
    st.subheader("Model Training", divider='blue')

    experiment1()
    experiment2()
    experiment3()
    experiment4()
    experiment5()
    experiment6()
    experiment7()
    experiment8()
    experiment9()
    experiment10()
    experiment11()
    experiment12()




def main():
    st.image('https://media.licdn.com/dms/image/D5616AQEfLf154Ai_vQ/profile-displaybackgroundimage-shrink_350_1400/0'
             '/1697114938690?e=1710979200&v=beta&t=d0iAStcRD9Mq2s0ae_M5qMBVX15An1wza0E_etroHmQ')

    st.title("Model Evaluation for Predicting the Storm Tracks - Deep Learning")

    # TODO introduction for this part
    st.write(
        ""
    )

    model_structure()

    loss_function()

    callbacks_function()

    model_training()


if __name__ == '__main__':
    st.set_page_config(
        page_title="Typhoon Forecast Dashboard - Storm Track Model Evaluation",
        page_icon="🌪️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
