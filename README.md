# heidi_dash

- [Overview](#overview)
- [Motivation and Purpose](#motivation-and-purpose)
- [Installation](#installation)
- [Description](#description)
- [App Sketch](#app-sketch)
- [Contributing](#contributing)
- [License](#license)

## Overview

The `heidi_dash` Dash app provides visualizations and statistics on the models trained on HEiDi dataset. The app is
primarily targeted towards nurses, policymakers, health professionals, and anyone interested in exploring and analyzing
triage prediction with interactive capabilities that allow them to study the accuracy of the models.

## Motivation and Purpose

Engineering an app to help nurses to make a triage decision may have a significant impact on patients as well as medical
system overall.

## Data Pipeline

The data pipeline created during our analysis is shown below. The folders marked with '00' are the initial setup which create the necessary files to run the analysis.

Following this, we have the order of the pipeline as:
- 01-eda
- 02-preprocessing
- 03-models

![image](screenshots/VM_Repo.png)

The README file present in the repository below is a step by step guide for the capstone partners to fully explore the teams analysis.

The data pipeline also provides the foundations for further analyses which the partners can carry out however they wish.

![image](screenshots/README.png)

![image](screenshots/kanban_board.png)

## Installation

If you would like to help contribute to the app, you can set up the system as follows:

1. Clone this repo using `https://github.com/stepanz25/heidi_dash.git`
2. Setup the environment file and download the necessary packages listed in `requirements.txt` using `conda`
   and `pip `

```
conda create --name heidi-dash python=3.8
conda activate heidi-dash
pip install -r requirements.txt
```

### To run the app locally:

1. Navigate to the root of this repo
2. In the command line, enter

```
python src/app.py
```

3. Copy the address (http://127.0.0.1:8051/) printed out after "Dash is running on" to a browser to view the Dash app.

## Description

The dashboard consists of one web page that shows statistics and 2 main reactive plots:

## App Sketch


## Contributing

Interested in contributing? Check out the contributing guidelines. We welcome and recognize all contributions. Please
find the guide for contribution in [Contributing Document](). Please note that this project is released with a Code of
Conduct. By contributing to this project, you agree to abide by its terms outlined [here]().

| Author                | Github Username |
|-----------------------|-----------------|
| Flora Ouedraogo       | @florawendy19   |
| Tanmay Agarwal        | @tanmayag97     |
| Waiel Hussain Tinwala | @WaielonH       |
| Stepan Zaiatc         | @stepanz25      |

## License

The materials of this project are licensed under the [MIT license](). If re-using/re-mixing please provide attribution
and link to this webpage.
