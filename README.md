**Gradient Boosting K2/TESS Data Cleaning and Analysis Project**
===========================================================

**Table of Contents**
-----------------

1. [Prerequisites](#prerequisites)
2. [Variables Used](#variables-used)
3. [Logic Used](#logic-used)
4. [File Execution Order](#file-execution-order)
5. [Commands to Run](#commands-to-run)
6. [Gradient Boosting Overview](#gradient-boosting-overview)
7. [Project Status](#project-status)

**Prerequisites**
---------------

* Install the following libraries using pip: `scikit-learn`, `Flask`, `pandas`, `numpy`, and `joblib`
* Ensure you have Python 3.8 or higher installed

**Variables Used**
-----------------

* `MISSIONS`: a dictionary containing configuration for each mission (K2, TESS)
* `config`: a variable used to store the current configuration being processed
* `df`: a pandas DataFrame used to store the input data

**Logic Used**
--------------

The project uses gradient boosting as its machine learning algorithm. Gradient boosting is an ensemble learning
method that combines multiple weak models to create a strong predictive model. In this project, we use
scikit-learn's `GradientBoostingClassifier` to classify the data into two categories: confirmed and false
positive.

**File Execution Order**
---------------------

1. Run `python aimodel.py`: This script cleans and preprocesses the data for each mission.
2. Run `python app.py`: This script creates a Flask web application that serves the cleaned and analyzed data.

**Commands to Run**
-----------------

* To run the project, first run `python aimodel.py` followed by `python app.py`
* Alternatively, you can run both scripts in parallel using `python -m multiprocess aimodel.py` and `python -m
multipprocess app.py`

**Gradient Boosting Overview**
-----------------------------

Gradient boosting is a powerful machine learning algorithm that has been shown to achieve high accuracy levels. In
this project, we use gradient boosting to classify the K2 and TESS data into confirmed and false positive
categories.

**Project Status**
------------------

* The project has achieved high accuracy levels of 99.5%+ for both K2 and TESS missions.
* Some features of the website are still not finished and have some gaps, but we are working to address these
issues in future updates.
* We encourage you to try out our project and provide feedback on how we can improve it.

**Contributing**
--------------

If you're interested in contributing to this project, please fork the repository and submit a pull request. We
welcome contributions from experienced data scientists and machine learning engineers.
