# Diabetes-prediction

<p>This project is a <strong>Diabetes Prediction System</strong> built using Python, Tkinter for the GUI, and a Support Vector Machine (SVM) for the prediction model. The system predicts whether a person has diabetes based on input data.</p>
        
  <h2>Features</h2>
        <ul>
            <li>Data preprocessing using pandas and NumPy</li>
            <li>Training an SVM classifier using scikit-learn</li>
            <li>Saving the trained model using pickle</li>
            <li>Creating a Tkinter-based GUI for user interaction</li>
            <li>Real-time prediction results based on user input</li>
        </ul>
      <h2>How to Run</h2>
        <p>To run this project, follow these steps:</p>
        <ol>
            <li>Clone the repository to your local machine.</li>
            <li>Ensure you have Python and the necessary libraries installed. You can install the required libraries using:</li>
            <pre><code>pip install -r requirements.txt</code></pre>
            <li>Place the diabetes dataset (diabetes.csv) in the same directory as the script.</li>
            <li>Run the script:</li>
            <pre><code>python your_script_name.py</code></pre>
        </ol>
        <h2>Usage</h2>
        <p>Upon running the script, a GUI window will appear where you can enter the relevant data points:</p>
        <ul>
            <li>Pregnancies</li>
            <li>Glucose</li>
            <li>Blood Pressure</li>
            <li>Skin Thickness</li>
            <li>Insulin</li>
            <li>BMI</li>
            <li>Diabetes Pedigree Function</li>
            <li>Age</li>
        </ul>
        <p>Click the "Predict" button to get the prediction result indicating whether the person is diabetic or not.</p>
        <h2>Code Explanation</h2>
        <p>The main components of the project are:</p>
        <ul>
            <li><code>load_model(file_path)</code>: Loads the trained SVM model from a pickle file.</li>
            <li><code>prediction(...)</code>: Takes user input, preprocesses the data, makes a prediction, and displays the result.</li>
            <li><code>clear_widgets(window)</code>: Clears all widgets from the specified window.</li>
            <li><code>predict(new)</code>: Sets up the prediction input form in the GUI.</li>
            <li><code>main_window()</code>: Initializes and runs the main GUI window.</li>
        </ul>
        <h2>Example</h2>
        <p>Here's a snippet of the main function that sets up and runs the GUI:</p>
        <pre><code>if __name__ == '__main__':
    main_window()</code></pre>
        <p>For more details, please check the full source code in the repository.</p>



Note:
<p>Ensure you have the required Python libraries and the diabetes dataset to run the project successfully.</p>    

<p>It's important to note that for a real-world application, thorough validation, testing, and consideration of ethical implications are crucial, especially in the context of health-related predictions.</p>
