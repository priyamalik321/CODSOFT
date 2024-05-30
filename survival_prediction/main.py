from flask import Flask, render_template, request
import pickle
import yaml
import os

import pandas as pd
from training import Train
from prediction import Predict


app = Flask(__name__)


if os.path.exists('config.yaml'):
    with open('config.yaml', 'r') as f:
        configs = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config file not found")


def training(configs):
    tr_obj = Train(configs)
    model_pipeline, x_train, x_test, y_train, y_test = tr_obj.train_model()
    tr_obj.save_test_dataset(x_test,y_test)
    return model_pipeline, x_train, x_test, y_train, y_test

model_pipeline, x_train, x_test, y_train, y_test=training(configs)


def prediction(model_pipeline, x_test, y_test):


    tm_obj = Predict()
    y_pred,score = tm_obj.test_model(model_pipeline, x_test, y_test)


    return  score


with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


score = prediction(model_pipeline, x_test, y_test)
print(f"Accuracy Score: {score}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:

            pclass_value = request.form.get('Pclass')
            passenger_id = request.form.get('PassengerId')
            sex_value = request.form.get('Sex')
            age_value = request.form.get('Age')
            sibsp_value = request.form.get('SibSp')
            parch_value = request.form.get('Parch')
            fare_value = request.form.get('Fare')
            embarked_value = request.form.get('Embarked')


            input_data = pd.DataFrame([[
                pclass_value, passenger_id, sex_value, age_value, sibsp_value, parch_value, fare_value, embarked_value
            ]], columns=configs['features_input'])

            print("Input Data:\n", input_data)
            input_data = input_data.astype({
                'Pclass': int,
                'PassengerId': int,
                'Age': float,
                'SibSp': int,
                'Parch': int,
                'Fare': float,
                'Sex':object,
                'Embarked':object
            })
            print("Converted Input Data:\n", input_data)

            result = model_pipeline.predict(input_data)[0]
            print('Model Prediction:', result)

            result_text = "Survived" if result == 1 else "Not Survived"
            return render_template("result.html", result=result_text)
        except Exception as e:
            print("Error during prediction:", e)
            return render_template("index.html", error=str(e))
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
