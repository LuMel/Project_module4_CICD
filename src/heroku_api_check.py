import json
import requests
import argparse


def test_api_heroku_neg(args):
    """ Test Fast API predict route with a '<=50K' salary prediction result """

    app_url = args.url + "/predict"
    
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

    r = requests.post(app_url, data=json.dumps(test_data_neg), 
                      headers={"Content-Type": "application/json"})

    assert r.status_code == 200
    assert r.json()["predictions"] == "0", "Unexpected output of the model"
    return r.json()["predictions"]


def test_api_heroku_pos(args):
    """ Test Fast API predict route with a '>50K' salary prediction result """
    app_url = args.url + "/predict"

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

    r = requests.post(app_url, data=json.dumps(test_data_pos), 
                      headers={"Content-Type": "application/json"})

    assert r.status_code == 200
    assert r.json()["predictions"] == "1", "Unexpected output of the model"
    return r.json()["predictions"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Census Bureau Heroku App Predictions Test CLI"
    )

    parser.add_argument(
        "url",
        type=str,
        help="url[:port] of the app to test inferences (e.g. http://127.0.0.1:8000)",
    )

    args = parser.parse_args()

    print(f"testing live app prediction for {args.url}...")

    # Call live testing function
    print("positive example")
    res = test_api_heroku_pos(args)
    print(res)

    print("negative example")
    res = test_api_heroku_neg(args)
    print(res)