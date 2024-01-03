# Diabetes-prediction (female Only)

<h2>Objective:</h2>
<p>The primary goal is to predict whether a person is diabetic or not based on input features related to health and medical history.</p>
<h3>Components:</h3>

<ul>
  <li>Prediction Model: Utilizes Support Vector Machine (SVM) from the scikit-learn library for classification.</li>
  <li>Graphical User Interface (GUI): Built using the Tkinter library, providing a user-friendly interface for input and result display.</li>
  <li>Image Display: Shows an image related to diabetes in the GUI.</li>
  <li>Data Processing: Uses pandas and numpy for handling and processing input data.</li>
</ul>





Model Training:

The script reads a dataset ('diabetes.csv') and splits it into training and testing sets.
The SVM classifier is trained on the training set and saved to a file using pickle.
Prediction Functionality:

The predict function creates a new window with input fields for relevant health parameters.
Upon entering the values and clicking the "Predict" button, the SVM model predicts whether the person is diabetic or not.
The result is displayed along with an option to make another prediction.
File Paths:

Paths for images ('DIABTIES/Dib.jpg') and the saved SVM model ('diabetes_svm_model.pkl').
User Interaction:

The main window provides a button ("Check Now!") to initiate the prediction process.
A warning label indicates that the application is intended for females only.
Scalability:

The project can be extended by incorporating additional features, improving the model, and enhancing the GUI for better user experience.
Note:

It's important to note that for a real-world application, thorough validation, testing, and consideration of ethical implications are crucial, especially in the context of health-related predictions.
