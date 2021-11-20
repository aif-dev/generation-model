from models import basic_model

def get_model(model, input_shape, output_shape, weights=None):
    if model == "basic_model":
        return basic_model.create_network(input_shape, output_shape, weights_filename=weights)
