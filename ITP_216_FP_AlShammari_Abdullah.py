# Abdullah AlShammari, aa62899@usc.edu
# ITP 216, Fall 2023
# Section: 32081
# Final Project
# Description:
'''
This file uses fileHandler.py to complete all the model training and data handling elsewhere to increase processing speed
** The 'back' buttons operate in html. They send to the home page url.

Functions on this file
home()  -   renders the landing page of the website at url "/"
predictSalary() - allows user to type in /predictSalary in the url and go straight to salary prediction page
payrollTeams() - allows user to type in /payrollTeams in the url and go straight to payroll pie chart page
userSelection()  - gets the user's selection in the landing page and sends to the respective page
fig() & create_figure() - create_figure() creates either a bar or pie chart based on the input. fig() saves the plot to an image
choosePlayer() - gets user player selection and populates the table on predictSalary page. It also populates the info
needed to make the bar graph
teamSalaryCosts() - retrieves player team selection and populates the info needed to make the pie chart
'''

# PYTHON MODULES
# import user-installed modules
from flask import Flask, Response, render_template, request, session, url_for, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn import metrics
import sqlite3 as sl
import pandas as pd
import numpy as np
import os
import io

# How do I create htmls easily?
# How do these work? any suggestions? Like the first home() just redirects to userSelection
from util.fileHandler import *
app = Flask(__name__)

# Reading CSVs
nba_player_log = pd.read_csv('nba_game_log_2021_22 (2).csv')
player_salaries = pd.read_csv('nba_salary_2021_22 (1).csv')

# Calling function executeAll() which takes the csvs and trains
player_log_salary, predSalaryDF,salaryMean = executeAll(nba_player_log,player_salaries)
# print(predSalaryDF)

@app.route("/")
def home():
    """
    Provides two options
    1) Check Player Value
    2) Visualize Team Costs per Player
    :return: renders respective page based on user's input
    """
    # TODO: your code goes here and replaces 'pass' below
    optionsList = ['Team Salary Costs', 'Player Valuation Analysis']
    return render_template("userSelection.html", message = "NBA Statistics and Player Valuation", options = optionsList)

@app.route("/predictSalary")
def predictSalary():
    # Empty dict to populate the table with
    dataDict = {"Choose Player first": "Empty"}

    # Set Andrew Wiggins as default plot
    name = "Andrew Wiggins"
    return render_template('playerValuation.html', nbaPlayerOptions=predSalaryDF['Player'],
                           playerValuation='Player SELECTED', data_dict=dataDict, input=name)

@app.route("/payrollTeams")
def payrollTeams():
    # List of team abbreviations to populate the drop down menu. Has to be in list form.
    team_abbrev_list = player_log_salary["Team_Abbrev"].unique().tolist()

    # Default Team set to Lakers
    abbrev = "LAL"
    return render_template('costPerTeam.html', input=abbrev,
                           teamAbbreviationOptions=team_abbrev_list)

@app.route("/action/userSelection", methods=["POST", "GET"])
def user_Selection():
    """
    :return: redirects to home if no valid input given, otherwise takes user's choice and
    sends to respective page
    """
    if request.method == "POST":
        # Gets the user input
        selected = request.form["data_request"]

        if selected == 'Team':
            # If they select the first one, render the costPerTeam.html
            # needs two inputs to render. user's selected abbreviation, and a list of team abbreviations
            team_abbrev_list = player_log_salary["Team_Abbrev"].unique().tolist()

            # Default Team set to Lakers
            abbrev = "LAL"
            return render_template('costPerTeam.html', input = abbrev,
                                   teamAbbreviationOptions = team_abbrev_list)
        elif selected == 'Player':
            # If they select the second one, render the playerValuation.html. Start off with empty dict to populate table
            # Default player set to Andrew Wiggins
            dataDict = {"Choose Player first" : "Empty"}
            name = "Andrew Wiggins"
            return render_template('playerValuation.html', nbaPlayerOptions=predSalaryDF['Player'],
                               playerValuation='Player SELECTED', data_dict=dataDict, input = name)
        else:
            return redirect(url_for("home"))

@app.route("/fig/<input>")
def fig(input):
    fig = create_figure(input)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype="image/png")
def create_figure(input):
    '''
    :param input: Player name
    :return:
    '''


    if len(input) < 5: # All names are more than 5 letters long
        # Group the player_log_salary by Player and Team
        team_salary_breakdown = player_log_salary.groupby(by=['Player', 'Team_Abbrev'],
                                                          as_index=False).agg({'Salary': 'mean'})

        team_salary_breakdown = team_salary_breakdown[team_salary_breakdown['Team_Abbrev'] == input].sort_values(
            by='Salary', ascending=False)

        top_players = []
        for i in range(0, 6):
            top_players.append(team_salary_breakdown.iloc[i]['Player'])

        # For each row, check if the player is part of the top 6, then assign him his own category of his name
        # All other players are put in the "Other Players" category
        team_salary_breakdown['top_players'] = team_salary_breakdown['Player'].apply(lambda x:
                                                                                     top_players[0] if x == top_players[0]
                                                                                     else
                                                                                     top_players[1] if x == top_players[1]
                                                                                     else
                                                                                     top_players[2] if x == top_players[2]
                                                                                     else
                                                                                     top_players[3] if x == top_players[3]
                                                                                     else
                                                                                     top_players[4] if x == top_players[4]
                                                                                     else
                                                                                     top_players[5] if x == top_players[5]
                                                                                     else 'Other Players')

        team_top_salaries = team_salary_breakdown.groupby(by=['top_players'], as_index=False).agg(
            {'Salary': 'sum'}).sort_values(by='Salary', ascending=False)

        # set title
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle('Total Payroll Costs by Team')
        # write code to create pie chart here
        ax.pie(x=team_top_salaries['Salary'], labels=team_top_salaries['top_players'], autopct='‘%1.1f%%')
        return fig
    else:
        player_row = predSalaryDF[predSalaryDF['Player'] == input] # 'input' is the player's name, retrieve selected player's data
        fig = Figure() # Instantiating a figure
        ax = fig.add_subplot(1, 1, 1) # ax is the first element in a 1x1 plot grid
        fig.suptitle('Compare Salary with AVG Salary') # Setting the Title

        # Retrieve data for the three columns
        categories = ['AVG Salary', 'Current Salary', 'Predicted Salary'] # Bar labels
        values = [salaryMean, player_row.iloc[0]['Salary'], player_row.iloc[0]['Predicted Salary']]  # Values for each bar

        # Create the bar graph
        ax.bar(categories, values, color=['blue', 'green', 'red'])
        return fig

@app.route("/action/choosePlayer", methods=["POST", "GET"])
def choosePlayer():
    if request.method == "POST":
        # Get user input
        player_name = request.form["nbaPlayers"]

        # Get the row of the player
        player_row = predSalaryDF[predSalaryDF['Player'] == player_name]

        # Variables for each column
        name = player_row.iloc[0]['Player']
        salary = player_row.iloc[0]['Salary']
        pSalary = player_row.iloc[0]['Predicted Salary']
        value = player_row.iloc[0]['Valuation']

        # Turn player row into a dictionary
        dataDict = {'Player Name': name , "Salary":salary, "Predicted Salary":pSalary, "Valuation":value}

    return render_template('playerValuation.html', nbaPlayerOptions=predSalaryDF['Player'],
                               playerValuation='Player SELECTED', data_dict=dataDict, input = player_name)

@app.route("/action/visualizeTeamSalary", methods=["POST", "GET"])
def teamSalaryCosts():
    team_abbrev_list = player_log_salary["Team_Abbrev"].unique().tolist()
    if request.method == "POST":
        # Gets the user's requested team
        abbrev = request.form["teamAbbreviation"]

        # needs two inputs to render. user's selected abbreviation (input), and a list of team abbreviations (teamAbbreviationOptions)
        return render_template('costPerTeam.html', input = abbrev,
                                   teamAbbreviationOptions = team_abbrev_list)
    else:
        return redirect(url_for("/payrollTeams"))

def main():
    app.secret_key = os.urandom(12)
    app.run(debug=True)

if __name__ == '__main__':
    main()

