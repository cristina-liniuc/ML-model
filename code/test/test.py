import json

from azureml.core import Webservice


def main(service):
    # Creating input data
    print("Creating input data")
    data = {"data": X_test.values.tolist()}
    input_data = json.dumps(data)

    # Calling webservice
    print("Calling webservice")
    output_data = service.run(input_data)
    predictions = output_data.get("predict")
    assert type(predictions) == list


if __name__ == "__main__":
    main()
