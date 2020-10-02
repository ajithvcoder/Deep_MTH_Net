from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Model

def build_tower(in_layer):
    neck = Dropout(0.2)(in_layer)
    neck = Dense(128, activation="relu")(neck)
    neck = Dropout(0.15)(neck)
    neck = Dense(128, activation="relu")(neck)
    return neck


def build_head(name, in_layer,num_units):
    return Dense(
        num_units[name], activation="softmax", name=f"{name}_output"
    )(in_layer)

def deepmth(backbone,num_units):

    x,inputs = backbone

    # heads
    gender = build_head("gender", build_tower(x),num_units)
    image_quality = build_head("image_quality", build_tower(x),num_units)
    age = build_head("age", build_tower(x),num_units)
    weight = build_head("weight", build_tower(x),num_units)
    bag = build_head("bag", build_tower(x),num_units)
    footwear = build_head("footwear", build_tower(x),num_units)
    emotion = build_head("emotion", build_tower(x),num_units)
    pose = build_head("pose", build_tower(x),num_units)

    outputs = [gender, image_quality, age, weight, bag, footwear, pose, emotion]

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
