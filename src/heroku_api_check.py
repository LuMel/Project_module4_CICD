import json
import requests
    
test_data_neg = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

test_data_pos = {
    "age": 40,
    "workclass": "Private",
    "fnlgt": 193524,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}



if __name__ == "__main__":

    response = requests.post(
                            'https://udacitycicd-app.herokuapp.com/predict',
                            data=json.dumps(test_data_neg))
    print(response.status_code)
    print(response.json())

    response = requests.post(
                            'https://udacitycicd-app.herokuapp.com/predict',
                            data=json.dumps(test_data_pos))
    print(response.status_code)
    print(response.json())