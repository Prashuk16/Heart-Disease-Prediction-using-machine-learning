
# Heart Disease Prediction

This project focuses on predicting the likelihood of heart disease using a dataset that contains various patient health indicators. The machine learning model employed in this project is Random Forest, which is used to classify patients as having heart disease or not, based on features such as age, cholesterol levels, blood pressure, and more. The project includes data preprocessing, model training, evaluation, and feature importance analysis.




## Dataset

The dataset used in this project is a consolidated version of the Statlog, Cleveland, and Hungarian heart disease datasets. It contains several medical attributes such as:

- age: Age of the patient.
- sex: Gender of the patient.
- cp: Chest pain type.
- trestbps: Resting blood pressure.
- chol: Serum cholesterol level.
- fbs: Fasting blood sugar > 120 mg/dl.
- restecg: Resting electrocardiographic results.
- thalach: Maximum heart rate achieved.
- exang: Exercise-induced angina.
- oldpeak: ST depression induced by exercise relative to rest.
- slope: The slope of the peak exercise ST segment.
- ca: Number of major vessels colored by fluoroscopy.
- thal: A blood disorder test result.

The target variable is target, which indicates the presence of heart disease (1 = disease, 0 = no disease).


## Steps in the Project
1. Data Preprocessing

- Loading Data: The dataset is loaded and inspected for shape, missing values, and duplicates.
- Handling Missing and Duplicate Data: Any missing values or duplicates are checked and addressed.
- Data Normalization: The features are normalized using StandardScaler to ensure better model performance.

2. Exploratory Data Analysis

- Correlation Matrix: A heatmap is used to display correlations between the different features, helping to identify relationships and redundancy in data.
- Histograms: Visualizations are generated for key features to understand the distribution of data.

3. Model Building

- Random Forest Classifier: The Random Forest algorithm is trained using various settings to find the optimal number of trees (n_estimators). The performance is measured based on accuracy.

4. Model Evaluation

- Confusion Matrix: The confusion matrix is used to evaluate the model's performance, showing true positives, false positives, true negatives, and false negatives.
- Accuracy: The accuracy of the model is calculated and analyzed for different configurations.
- Feature Importance: The importance of each feature in determining heart disease is calculated and visualized.

## Results

- The Random Forest model achieves high accuracy based on the selected features.
- The feature importance analysis shows that certain attributes, such as thalach, oldpeak, and cp, play a crucial role in predicting heart disease.
## Conclusion

This project demonstrates how machine learning techniques like Random Forest can be used to predict heart disease effectively. The model can be further tuned and improved based on additional data and hyperparameter tuning.
## Future Improvements

- Hyperparameter Tuning: More thorough tuning of Random Forest parameters like max_depth and min_samples_split.
- Cross-Validation: Implement cross-validation to further generalize the model's performance.
- Model Comparison: Compare Random Forest with other models like SVM, Logistic Regression, and Neural Networks for better performance insights.