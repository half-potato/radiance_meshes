import torch

def recombine_tensors(non_tensor_data, tensor_data):
    tensor_keys = non_tensor_data['_tensor_keys']
    del non_tensor_data['_tensor_keys']
    reconstructed_dict = {key: tensor for key, tensor in zip(tensor_keys, tensor_data)}
    reconstructed_dict.update(non_tensor_data)
    return reconstructed_dict

def split_tensors(tcam):
    tensors = []
    tensor_keys = []
    non_tensor_data = {}

    for key, value in tcam.items():
        if isinstance(value, torch.Tensor):
            tensors.append(value)
            tensor_keys.append(key)
        else:
            non_tensor_data[key] = value
    
    non_tensor_data['_tensor_keys'] = tensor_keys
    return non_tensor_data, tensors
