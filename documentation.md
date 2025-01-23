# Assumptions!!!:
    It is assumed that the project folder contains a subfolder named `datasets`, which includes two files: `churn_train.csv` and `sms_spam_detection.csv`. Additionally, please ensure that the Python packages listed in the `requirements.txt` file are installed in your environment.

# Part A: Analysis of the SMS Spam Detection File

The `sms_spam_detection.ipynb` notebook is designed to implement a spam detection model using various machine learning algorithms. The analysis is structured into several key components:
You can test the spam detection model using the file named `spam_detection_gui.py` by executing the code within it.

### 1. Data Import and Preprocessing
    The preprocessing function is defined to clean the text data by converting it to lowercase, removing extra spaces, and eliminating non-alphabetic characters. 
    Additionally, the `SentenceTransformer` model, specifically `distiluse-base-multilingual-cased`, is employed to encode the cleaned text messages into vector representations, facilitating effective input for the machine learning algorithms.

### 2. Model Definition
    Three machine learning models are defined for spam detection:
    - **Support Vector Classifier (SVC)**
    - **Logistic Regression**
    - **Random Forest Classifier**

    These models are chosen for their effectiveness in classification tasks, particularly in text classification scenarios.

### 3. Model Training and Evaluation
    The models were trained and evaluated using a dataset of text messages labeled as either "spam" or "ham." The performance of each model was assessed using key metrics: F1 Score, Recall, and Precision.

    - **Support Vector Classifier (SVC)**
    - F1 Score: 0.971
    - Recall: 0.955
    - Precision: 0.989

    - **Random Forest Classifier**
    - F1 Score: 0.952
    - Recall: 0.925
    - Precision: 0.981

    - **Logistic Regression**
    - F1 Score: 0.950
    - Recall: 0.939
    - Precision: 0.962

### Conclusion
    The Support Vector Classifier (SVC) achieved the highest F1 Score of 0.971, indicating its superior performance in spam detection. 
    Both the Random Forest and Logistic Regression models also performed well, with F1 Scores of 0.952 and 0.950, respectively, but the SVC outperformed them in balancing precision and recall.

# Part B: Churn Analysis Documentation

For this part, two files have been prepared. The first file, `churn_prediction.ipynb`, encompasses feature extraction and model training processes. 
The second file, `churn_prediction_gui.py`, is designed to create a graphical user interface (GUI) for testing the performance of the trained model (**just run the file).
In the following sections, I will present the data processing and model evaluation conducted in the file `churn_prediction.ipynb`.

### 1. Exploratory Data Analysis (EDA)
    EDA is conducted to uncover patterns and insights within the data. This includes:
    - Checking for missing values and understanding their impact.
    - Visualizing the distribution of key features and the target variable (churn).
    - Identifying correlations between features to understand relationships that may influence churn.

### 2. Data Preprocessing and Feature Engineering
    Data preprocessing includes:
    - Encoding categorical variables to convert them into a numerical format suitable for machine learning algorithms.
    - Normalizing or scaling features if necessary to ensure that all features contribute equally to the model.

### 3. Model Training and Evaluation
    The performance of the models (Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient-Boosted Decision Trees) is assessed using metrics such as:
    Precision, Recall, F1 Score. I also employed grid search to identify the optimal model.
    Confusion matrices are also utilized to visualize model performance and identify areas for improvement.

    - **Gradient-Boosted Decision Trees**
        - Precision: 63%
        - Recall: 76%
        - F1 Score: 69%

    - **Logistic Regression**
        - Precision: 71%
        - Recall: 61%
        - F1 Score: 66%

    - **Decision Tree Classifier**
        - Precision: 54%
        - Recall: 74%
        - F1 Score: 62%

    - **Random Forest Classifier**
        - Precision: 64%
        - Recall: 72%
        - F1 Score: 68%

    - **Random Forest Classifier**
        - Precision: 64%
        - Recall: 72%
        - F1 Score: 68%


### 4. Conclusion and Recommendations
    Based on the results, the Gradient-Boosted Decision Trees demonstrated the highest performance in predicting customer churn, 
    achieving an F1 score of 69% and indicating its effectiveness as a reliable model for identifying at-risk customers.
    ## Future Work

    Future analyses could explore:
    - Incorporating additional data sources for a more comprehensive view of customer behavior.
    - Experimenting with advanced modeling techniques, such as ensemble methods or deep learning.
