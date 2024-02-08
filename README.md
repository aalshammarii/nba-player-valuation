# NBA Salary Prediction and Visualization

## Overview
This project, developed by Abdullah AlShammari for ITP 216 (Fall 2023), offers an insightful web interface for analyzing NBA player salaries and team payroll distributions. Utilizing Flask for the backend, it leverages data science and machine learning to predict player salaries based on performance metrics and visualizes team payroll structures.

## Features
- **Salary Prediction:** Predicts NBA players' salaries using performance indicators from recent seasons.
- **Team Payroll Visualization:** Provides interactive charts to explore team salary allocations.
- **Data-Driven Insights:** Uses actual NBA game logs and player salary data for analysis.
- **User Interaction:** A web interface allows users to select players or teams for detailed financial analysis.

## Installation
Ensure you have Python 3.x installed. Clone the repository and install dependencies:
```bash
git clone [repository-url]
cd [project-directory]
pip install -r requirements.txt
```

## Usage
To start the application:
```bash
python ITP_216_FP_AlShammari_Abdullah.py
```
Access the web interface at `http://localhost:5000`.

### Key Functionalities
- Navigate to `/` for the landing page.
- Use `/predictSalary` for direct access to the salary prediction page.
- Explore team payroll distributions at `/payrollTeams`.

## Technical Details
- **Back-end Framework:** Flask
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn for salary prediction models
- **Visualization:** Matplotlib for generating interactive charts

## File Descriptions
- `ITP_216_FP_AlShammari_Abdullah.py`: Main Flask application file. Sets up the web interface and integrates data processing and visualization functionalities.
- `util/fileHandler.py`: Handles data preparation, scaling, and machine learning model training.
- `templates/`: Contains HTML templates for the web interface.

## Data Sources
- `nba_salary_2021_22.csv`: Player salary data for the 2021/22 season.
- `nba_game_log_2021_22.csv`: Detailed game logs providing performance metrics.

## Contributing
Contributions to enhance functionalities or improve the data analysis are welcome. Please follow standard pull request procedures.

## Contact
For any queries or further information, please contact Abdullah AlShammari at aa62899@usc.edu.
