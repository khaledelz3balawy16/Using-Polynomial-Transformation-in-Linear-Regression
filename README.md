Title: Using Polynomial Transformation in Linear Regression 

Description:
This code demonstrates how to perform polynomial transformation in linear regression 

Explanation:

Importing Libraries: The necessary libraries are imported, including pandas for data manipulation and sklearn for machine learning functionalities.

Loading Dataset: The dataset is loaded from a CSV file ('satf.csv') using pandas. The first 10 rows of the dataset are displayed to give an overview of the data.

Data Preparation: The independent variable (X) and dependent variable (y) are separated from the dataset. The dataset is then split into training and testing sets using the train_test_split function from sklearn.

Polynomial Transformation: Polynomial features are created for both the training and testing sets using the PolynomialFeatures class from sklearn. This allows us to transform the original features into polynomial features of a specified degree (in this case, degree=2).

Linear Regression: A linear regression model is trained using the transformed training data. The LinearRegression class from sklearn is used for this purpose.

Prediction and Evaluation: The model is used to make predictions on the transformed testing data. Mean Absolute Error, Mean Squared Error, and Median Absolute Error are calculated to evaluate the performance of the model.

Conclusion:
By using polynomial transformation, the model is able to handle high dimensionality features more effectively, potentially leading to better performance in predicting the target variable.
