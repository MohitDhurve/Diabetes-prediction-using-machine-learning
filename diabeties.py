from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

IMAGE_PATH = 'C:/Users/dhurv/Desktop/pythonProjects/project1/DIABTIES/Dib.jpg'
MODEL_PATH = 'diabetes_svm_model.pkl'

dataframe = pd.read_csv("diabetes.csv")
X = dataframe.drop(columns='Outcome', axis=1)
Y = dataframe['Outcome']

scaler = StandardScaler()
scaler.fit(X)
std_data = scaler.transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(std_data, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

with open(MODEL_PATH, 'wb') as model_file:
    pickle.dump(classifier, model_file)

def load_model(file_path):
    with open(file_path, 'rb') as model_file:
        loaded_classifier = pickle.load(model_file)
    return loaded_classifier

def prediction(d, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    if '' in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]:
        messagebox.showerror("Error", "Please fill out all input fields.")
        return

    input_data = np.array(
        [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age], dtype=float)
    input_data_reshaped = input_data.reshape(1, -1)

    std_data = scaler.transform(input_data_reshaped)

    loaded_classifier = load_model(MODEL_PATH)

    Prediction = loaded_classifier.predict(std_data)

    accuracy = accuracy_score(Y_test, loaded_classifier.predict(X_test))
    print(f"Accuracy: {accuracy}")

    if Prediction[0] == 0:
        result_text = 'The person is not diabetic'
        result_color = 'green'
    else:
        result_text = 'The person is diabetic'
        result_color = 'red'

    clear_widgets(d)

    label = Label(d, text=result_text, font=("Helvetica", 20), fg=result_color)
    label.pack()

    button = Button(d, text='Predict another', command=lambda: predict(d, '', '', '', '', '', '', '', ''))
    button.pack()

def clear_widgets(window):
    for widget in window.winfo_children():
        widget.destroy()

def predict(new):
    clear_widgets(new)
    d = new
    d.title('Predict Diabetes')


    labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
    entries = {}

    for i, label_text in enumerate(labels):
        label = Label(d, text=label_text)
        label.pack()
        entry = Entry(d)
        entry.pack()
        entries[label_text] = entry


    button = Button(d, text='Predict', fg='Blue', font=("Helvetica", 20),
                    command=lambda: prediction(d,
                                               entries['Pregnancies'].get(),
                                               entries['Glucose'].get(),
                                               entries['Blood Pressure'].get(),
                                               entries['Skin Thickness'].get(),
                                               entries['Insulin'].get(),
                                               entries['BMI'].get(),
                                               entries['Diabetes Pedigree Function'].get(),
                                               entries['Age'].get()))
    button.pack()

def main_window():
    new = Tk()
    new.title('Diabetes Prediction')
    new.maxsize(600, 600)
    new.minsize(600, 600)
    image = Image.open(IMAGE_PATH)
    image = image.resize((400, 400))
    photo = ImageTk.PhotoImage(image)
    label = Label(new, image=photo)
    label.pack()
    l = Label(new, text='For females only *', fg='red')
    l.pack()

    button = Button(new, text='Check Now!', fg='Blue', font=("Helvetica", 20), command=lambda: predict(new))
    button.pack()
    new.mainloop()

if __name__ == '__main__':
    main_window()
