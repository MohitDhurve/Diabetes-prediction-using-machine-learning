from tkinter import *
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

# File paths
IMAGE_PATH = 'C:/Users/dhurv/PycharmProjects/aiproject/DIABTIES/Dib.jpg'
MODEL_PATH = 'diabetes_svm_model.pkl'


def save_model(classifier, file_path):
    # Save the trained model to a file using pickle
    with open(file_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)


def load_model(file_path):
    # Load the saved model for predictions
    with open(file_path, 'rb') as model_file:
        loaded_classifier = pickle.load(model_file)
    return loaded_classifier


def prediction(d, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    dataframe = pd.read_csv("diabetes.csv")
    X = dataframe.drop(columns='Outcome', axis=1)
    Y = dataframe['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    std_data = scaler.transform(X)
    X = std_data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    # Save the trained model
    save_model(classifier, MODEL_PATH)

    input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    input_data_to_numpy = np.asarray(input_data)
    input_data_reshaped = input_data_to_numpy.reshape(1, -1)

    std_data = scaler.transform(input_data_reshaped)

    # Load the saved model
    loaded_classifier = load_model(MODEL_PATH)

    Prediction = loaded_classifier.predict(std_data)

    if Prediction[0] == 0:
        label = Label(d, text='The person is not diabetic', font=("Helvetica", 20), fg='green')
        label.pack()
        button = Button(d, text='Predict another', command=lambda: predict(d))
        button.pack()
    else:
        label = Label(d, text='The person is diabetic', font=("Helvetica", 20), fg='red')
        label.pack()
        button = Button(d, text='Predict another', command=lambda: predict(d))
        button.pack()


def clear_widgets(window):
    for widget in window.winfo_children():
        widget.destroy()


def predict(new):
    clear_widgets(new)
    d = new
    d.title('Predict Diabetes')

    Pregnancies = StringVar()
    Glucose = StringVar()
    BloodPressure = StringVar()
    SkinThickness = StringVar()
    Insulin = StringVar()
    BMI = StringVar()
    DiabetesPedigreeFunction = StringVar()
    Age = StringVar()

    l1 = Label(d, text='Pregnancies')
    l1.pack()

    entry1 = Entry(d, textvariable=Pregnancies)
    entry1.pack()

    l2 = Label(d, text='Glucose')
    l2.pack()

    entry2 = Entry(d, textvariable=Glucose)
    entry2.pack()

    l3 = Label(d, text='Blood Pressure')
    l3.pack()

    entry3 = Entry(d, textvariable=BloodPressure)
    entry3.pack()

    l4 = Label(d, text='Skin Thickness')
    l4.pack()

    entry4 = Entry(d, textvariable=SkinThickness)
    entry4.pack()

    l5 = Label(d, text='Insulin')
    l5.pack()

    entry5 = Entry(d, textvariable=Insulin)
    entry5.pack()

    l6 = Label(d, text='BMI')
    l6.pack()

    entry6 = Entry(d, textvariable=BMI)
    entry6.pack()

    l7 = Label(d, text='Diabetes Pedigree Function')
    l7.pack()

    entry7 = Entry(d, textvariable=DiabetesPedigreeFunction)
    entry7.pack()

    l8 = Label(d, text='Age')
    l8.pack()

    entry8 = Entry(d, textvariable=Age)
    entry8.pack()

    button = Button(d, text='Predict', fg='Blue', font=("Helvetica", 20),
                    command=lambda: prediction(d, Pregnancies.get(), Glucose.get(), BloodPressure.get(),
                                               SkinThickness.get(), Insulin.get(), BMI.get(),
                                               DiabetesPedigreeFunction.get(), Age.get()))
    button.pack()

    d.mainloop()


def main_window():
    new = Tk()
    new.title('hello')
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
