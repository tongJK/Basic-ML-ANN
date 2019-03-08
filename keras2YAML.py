def writeYAMLfile():

    from keras.models import load_model

    model = load_model("model.hdf5")
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    model.save_weights("model.h5")
    print("Save model finished")

def loadYAMLfile():

    from keras.models import model_from_yaml

    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)

    loaded_model.load_weights("model.h5")
    print("Load model finished")

writeYAMLfile()
model = loadYAMLfile()

value_to_predict = 'variable that include the value to predict'
model_to_use = model.predict(value_to_predict)