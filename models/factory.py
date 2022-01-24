from models import lstm_model, music_transfomer

def get_model(model, input_shape, output_shape, weights=None):
    if model == "lstm":
        return lstm_model.create_network(input_shape, output_shape, weights_filename=weights)
    elif model == "transformer":
        return music_transfomer.create_network(input_shape, output_shape, weights_filename=weights)
    else:
        raise ValueError(f"There is no model with name {model}")
