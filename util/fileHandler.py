# Abdullah AlShammari, aa62899@usc.edu
# ITP 216, Fall 2023
# Section: 32081
# Final Project
# Description: This handles all the file data preparation so the other file can focus on just websites
'''
This file handles the data, prepares it, trains it, and returns three files for the actual project to use.

FUNCTIONS IN THIS FILE:
-----------------------
(1) data_preparation() - This function takes the dataframe inputted and extracts only the values necessary for modeling ML
(2) split_scale_data() - Splits the data and scales it
(3) data_regression_modeling() - Use RandomForestRegressor (RFR) to predict player salary based on X data runs GridSearchCV
                                 to find best paramters then trains the RFR model on these best paramaters
(4) pred_player_salary() - runs predictions on all players after model training is done
(5) executeAll() - puts all functions together and returns two dataframes and an integer

'''
# PYTHON MODULES
# import user-installed modules
from flask import Flask, redirect, render_template, request, session, url_for
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import sqlite3 as sl
import pandas as pd
import numpy as np
import os

def data_preparation(X_train_list, Y_train_list, dataFrame,groupOn):
    '''
    This function takes the dataframe inputted and extracts only the values necessary for modeling ML
    :param X_train_list: list of column names the user wants to include from the dataframe X
    :param Y_train_list: list of column names the user wants to include from the dataframe Y
    :param dataFrame: dataframe to extract columns
    :param groupOn: the column to group by
    :return:X,Y (dataframes containing the data we need to split to train our model
    '''
    # Create empty dataframe
    pts_minutes_per_player = pd.DataFrame()

    # Loop through X_train list to group and extract first two columns (avg min/game, avg pts scored/game)
    for item in X_train_list:
        # Group based on user's preference (in our case it will be player)
        grouped_data = dataFrame.groupby(by=[groupOn], as_index=False).agg({item: 'mean'})

        # Concat/add the column to the empty dataframe of pts_min_per_player
        pts_minutes_per_player = pd.concat([pts_minutes_per_player, grouped_data[item]], axis=1)

    # Extract total pts scored
    grouped_data = dataFrame.groupby(by=[groupOn], as_index=False).agg({'pts': 'sum'})
    pts_minutes_per_player = pd.concat([pts_minutes_per_player, grouped_data[item]], axis=1)

    # Assign Dataframe to X variable for splitting later on
    X = pts_minutes_per_player

    # create a dataframe grouped by "player", where you only choose unique values in column "Salary".
    # Also asked to reset the index, but still each salary sits respective to the points contained in DF X
    Y = dataFrame.groupby([groupOn])[Y_train_list[0]].unique().reset_index() # Group the salaries by player, and keeping unique values only.
    Y['Salary'] = Y['Salary'].str[0].astype(float) # Removing the brackets
    return X,Y

def split_scale_data(X,Y):
    '''
    Splits the data and scales it
    :param X: the training data (avg min played per game, avg scored per game, total score)
    :param Y: the testing data (player, salary)
    :return:
    '''
    # Separating training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=2023)

    # Scale the Data
    scaler = StandardScaler()  # Instantiate scaler as an object of StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)  # Scale the training X dataset
    scaled_X_test = scaler.transform(X_test)  # Scale the testing X dataset
    return scaled_X_train, scaled_X_test, y_train, y_test

def data_regression_modeling(scaled_X_train, scaled_X_test, y_train, y_test):
    '''
    Use RandomForestRegressor (RFR) to predict player salary based on X data
    runs GridSearchCV to find best paramters then trains the RFR model on these best paramaters
    :param scaled_X_train: scaled train data (avg min played per game, avg scored per game, total score)
    :param scaled_X_test: scaled test data (avg min played per game, avg scored per game, total score)
    :param y_train: list of salaries
    :param y_test: list of salaries

    :return model: trained model that can intake scaled X values and spit out predicted salaries
    '''
    best_params = [] # Create an empty list to house the best parameters found by GridSearchCV
    param_grid = {'n_estimators': [15, 25, 50, 64, 100, 200], # How many trees to use?
                  'max_features': [3], # How many features to use? Since I have only 3, I have set it to use all of them all the time
                  'bootstrap': [True, False], # Do we resample data or not? True = YES, False = No
                  'oob_score': [True]} # Use the data not sampled to test accuracy (Out of Bag Score)
    # Instantiate the model
    rfc = RandomForestRegressor()

    # Instantiate GridSearchCV on rfc and param_grid
    grid = GridSearchCV(rfc, param_grid)

    # Test all possible combinations of parameters and extract the best parameters
    grid.fit(scaled_X_train, y_train['Salary'])

    # Instantiate another model with the best parameters taken from grid
    model = RandomForestRegressor(**grid.best_params_)
    best_params.append(grid.best_params_) # TODO: Do you need this?????? I don't think so!

    # Fit the model on the scaled_X_train data, and the y_train
    model.fit(scaled_X_train, y_train['Salary'])

    # Predict
    y_pred = model.predict(scaled_X_test)

    return model

def pred_player_salary(model, X, Y):
    '''
    After model is trained, we want to run predictions on ALL players.
    Takes in X, scales it, then uses model to predict player salaries.
    Converts panda series to DF, then concatenates on Y.
    :param model: trained ML model to predict values
    :param X: dataframe -> values to be used to predict on.
    :param Y: dataframe -> compare predicted with actual
    :return: dataframe with dataframe Y concatenated with predicted values, and thousands separated by comma
    '''
    # end goal: create a table that has Player, Salary, Predicted Salary
    # X Y
    # Scale All Player Data
    scaler = StandardScaler()  # Instantiate scaler as an object of StandardScaler()
    scaled_X = scaler.fit_transform(X)

    # Predict Salaries and transform to dataframe
    pred_Salaries = model.predict(scaled_X)
    pred_SalariesDF = pd.DataFrame({'Predicted Salary': pred_Salaries})

    # Concatenate Y (player, actual salary) with predicted salaries
    pd.options.display.float_format = '{:.2f}'.format
    Y = pd.concat([Y, pred_SalariesDF], axis=1)
    Y['Valuation'] = np.where(Y['Salary'] > Y['Predicted Salary'], 'Overvalued', 'Undervalued')
    salaryMean = Y['Salary'].mean()

    # # Set formatting of numbers to have comma separators for thousands
    # Y['Salary'] = Y['Salary'].apply(lambda x: '{:,.0f}'.format(x))
    # Y['Predicted Salary'] = Y['Predicted Salary'].apply(lambda x: '{:,.0f}'.format(x))

    # Create a new column that tells us if the player is overvalued or undervalued
    pd.set_option('display.max_rows', None)
    return Y, salaryMean
def executeAll(nba_player_log, player_salaries):
    '''
    :param nba_player_log: csv file containing information about player stats and game stats
    :param player_salaries: csv file containing player name and salary for next 5 years
    :return player_log_salary: player_log_salary is cleaned up data to be used for plotting
            predSalaryDF: predSalaryDF is dataframe containing salary predictions and
                          decisions on Undervalued vs Overvalued
            salaryMean: average salary of all the NBA league
    '''
    # Cleaning up player log
    player_log_cleaned = nba_player_log[
        ['game_id', 'game_date', 'H_A', 'Team_Abbrev', 'Opponent_Abbrev', 'player', 'pts', 'minutes']]
    player_log_cleaned = player_log_cleaned.drop_duplicates()

    # Cleaning up player salary
    salary = player_salaries[['Player', '2021/22']]
    salary = salary.drop_duplicates()

    # Merging tables 'player_log_cleaned' and 'salary' on the common key player
    # adding the salary information on the cleaned player log
    # columns = ['game_id', 'game_date', 'H_A', 'Team_Abbrev', 'Opponent_Abbrev', 'pts', 'minutes', 'Player', 'Salary']
    player_log_salary = pd.merge(player_log_cleaned, salary, how='left', left_on='player', right_on='Player')

    # Drop null values
    player_log_salary = player_log_salary.dropna(how='any')

    # Rename Columns
    player_log_salary.rename(columns={"2021/22": "Salary", "H_A": "Home_Away"}, inplace=True)

    # Call data preparation to split the data into X, and Y
    X, Y = data_preparation(['minutes', 'pts'], ['Salary'], player_log_salary, 'Player')

    # Call split_scale_data to get the four variables to train the data
    scaled_X_train, scaled_X_test, y_train, y_test = split_scale_data(X, Y)

    # Train the data
    model = data_regression_modeling(scaled_X_train, scaled_X_test, y_train, y_test)

    # Get dataframe holding dataframe [Player, Salary, Predicted Salary, Overvalued or Undervalued]
    predSalaryDF,salaryMean = pred_player_salary(model, X, Y)

    return player_log_salary, predSalaryDF,salaryMean


