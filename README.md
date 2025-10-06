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
7. [K2/TESS Data Analysis](#k2-tess-data-analysis)
8. [Additional Features and Functions](#additional-features-and-functions)
9. [Project Status](#project-status)
10. [Contributing](#contributing)

**Prerequisites**
---------------

* Install the following libraries using pip: `scikit-learn`, `Flask`, `pandas`, `numpy`, `joblib`, and `astropy`
(for Kepler data analysis)
* Ensure you have Python 3.8 or higher installed
* Familiarity with machine learning and data analysis concepts is recommended

**Variables Used**
-----------------

* `MISSIONS`: a dictionary containing configuration for each mission (K2, TESS)
* `config`: a variable used to store the current configuration being processed
* `df`: a pandas DataFrame used to store the input data
* `kepler_data`: a variable used to store Kepler data
* `k2_features`: a list of features used for K2 data analysis
* `tess_features`: a list of features used for TESS data analysis

**Logic Used**
--------------

The project uses gradient boosting as its machine learning algorithm. Gradient boosting is an ensemble learning
method that combines multiple weak models to create a strong predictive model. In this project, we use
scikit-learn's `GradientBoostingClassifier` to classify the data into two categories: confirmed and false
positive.

Additionally, the project analyzes Kepler data using astropy and pandas. The Kepler data is processed to extract
relevant features, such as planetary period and eccentricity, which are then used in conjunction with machine
learning algorithms to predict planetary status.

**File Execution Order**
---------------------

1. Run `python aimodel.py`: This script cleans and preprocesses the data for each mission.
2. Run `python app.py`: This script creates a Flask web application that serves the cleaned and analyzed data.
3. Optionally, run `python kepler_analysis.py` to analyze Kepler data.

**Commands to Run**
-----------------

* To run the project, first run `python aimodel.py` followed by `python app.py`
* Alternatively, you can run both scripts in parallel using `python -m multiprocess aimodel.py` and `python -m
multiprocess app.py`
* If analyzing Kepler data, run `python kepler_analysis.py`

**Gradient Boosting Overview**
-----------------------------

Gradient boosting is a powerful machine learning algorithm that has been shown to achieve high accuracy levels. In
this project, we use gradient boosting to classify the K2 and TESS data into confirmed and false positive
categories.

The accuracy of our model exceeds 99.5% for both K2 and TESS missions, demonstrating its effectiveness in
predicting planetary status based on available features.

**K2/TESS Data Analysis**
-------------------------

In addition to using machine learning algorithms, we also analyze Kepler data using astropy and pandas. The Kepler
data is processed to extract relevant features, such as planetary period and eccentricity, which are then used in
conjunction with machine learning algorithms to predict planetary status.

The K2 and TESS missions have provided unprecedented amounts of data on exoplanetary systems, allowing us to
develop more accurate models of planetary behavior. Our analysis takes advantage of this wealth of information to
improve the accuracy of our predictions.

**Additional Features and Functions**
--------------------------------------

* The `kepler_analysis.py` script analyzes Kepler data using astropy and pandas.
* The `aimodel.py` script cleans and preprocesses the data for each mission, including K2 and TESS data.
* The `app.py` script creates a Flask web application that serves the cleaned and analyzed data.

**Project Status**
------------------

The project has achieved high accuracy levels of 99.5%+ for both K2 and TESS missions.
Some features of the website are still not finished and have some gaps, but we are working to address these issues
in future updates.
We encourage you to try out our project and provide feedback on how we can improve it.

**Contributing**
--------------

If you're interested in contributing to this project, please fork the repository and submit a pull request.
We welcome contributions from experienced data scientists and machine learning engineers.
