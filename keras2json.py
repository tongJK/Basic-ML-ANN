def writeJSONfile():

    from keras.models import load_model

    model = load_model("model.hdf5")
    model_json = model.to_json()
    with open("json_model.json", 'w') as json_file:
        json_file.write(model_json)

    model.save_weights("weight_model.h5")
    print("Save model finished")

def loadJSONfile():

    from keras.models import model_from_json

    with open("json_model.json", 'r') as json_file:
        loaded_json = json_file.read()

    model = model_from_json(loaded_json)
    model.load_weights("weight_model.h5")
    print("Load model finished")

    return model


writeJSONfile()
model = loadJSONfile()

value_to_predict = 'variable that include the value to predict'
model_to_use = model.predict(value_to_predict)