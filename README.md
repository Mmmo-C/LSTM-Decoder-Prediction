# LSTM/Decoder Prediction
Sea-Surface Temperature Prediction using LSTM/Decoder

</p>
Xinqi Chen @22/05/2023

## Table of Content
- [Lorenz System Dynamics Forecasting](#lorenz-system-dynamics-forecasting)
  - [Abstract](#abstract)
  - [Overview](#overview)
  - [Theoretical Background](#theoretical-background)
  - [Algorithm Implementation and Development](#algorithm-implementation-and-development)
  - [Computational Results](#computational-results)
  - [Summary and Conclusions](#summary-and-conclusions)
  - [Acknowledgement](#acknowledgement)

## Abstract
This project focuses on predicting sea-surface temperature using a Long Short-Term Memory (LSTM) model with a decoder. The goal is to analyze the model's performance based on various factors, including the time lag variable, noise levels, and the number of sensors. By examining these factors, we aim to gain insights into the model's robustness and its ability to accurately predict sea-surface temperature under different conditions.

## Overview
The project utilizes an LSTM-based architecture along with a decoder to predict sea-surface temperature. The LSTM model is capable of capturing temporal dependencies and can effectively handle time-series data. The decoder component helps generate accurate predictions by reconstructing the input sequence. The project provides code and data that can be used to train the model and evaluate its performance.

## Theoretical Background
The Long Short-Term Memory (LSTM) network is a type of recurrent neural network (RNN) that is well-suited for processing sequential data. Unlike traditional RNNs, LSTM units have memory cells and gating mechanisms that allow them to retain and selectively update information over long sequences. This makes LSTMs effective for capturing long-term dependencies in time-series data, such as sea-surface temperature.

## Algorithm Implementation and Development
The project includes example code and data for training the sea-surface temperature prediction model using an LSTM/decoder architecture. The implementation follows these main steps:

1. Data Preparation:
   - Load the sea-surface temperature data.
   - Preprocess the data if necessary, such as normalization or splitting into training and testing sets.

2. Model Training:
   - Set up the LSTM/decoder architecture as defined in the code.
   - Train the model using the prepared training data.
   - Optimize the model's hyperparameters to improve performance if needed.

3. Model Evaluation:
   - Evaluate the trained model on the testing data.
   - Calculate performance metrics such as mean squared error (MSE) or mean absolute error (MAE).
   - Visualize the predicted sea-surface temperature values against the ground truth for analysis.

4. Performance Analysis:
   - Conduct a performance analysis based on the time lag variable:
     - Modify the code to experiment with different time lags between input and output.
     - Train and evaluate the model with different time lags.
     - Analyze performance metrics as a function of the time lag variable.

   - Conduct a performance analysis based on noise:
     - Add Gaussian noise to the sea-surface temperature data before training the model.
     - Vary the noise levels and observe the impact on the model's performance.

   - Conduct a performance analysis based on the number of sensors:
     - Modify the code to handle different numbers of sensors or inputs.
     - Train and evaluate the model with different numbers of sensors.
     - Analyze the model's performance as a function of the number of sensors.

Note: Detailed instructions and code usage can be found within the project's codebase.

## Computational Results


## Summary and Conclusions


## Acknowledgement
-[Jan Williams](https://github.com/Jan-Williams/pyshred)
