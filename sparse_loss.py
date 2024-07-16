# # sparse_loss.py
# import torch
# import torch.nn.functional as F
#
#
# def sparse_loss(model, images):
#     """
#     Computes the sparse regularization loss for a given model and input images.
#
#     The sparse regularization loss encourages the activations of the model to be sparse,
#     promoting the selection of a subset of relevant features and reducing the overall
#     complexity of the model.
#
#     Args:
#         model (torch.nn.Module): The neural network model.
#         images (torch.Tensor): The input images tensor of shape (batch_size, channels, height, width).
#
#     Returns:
#         torch.Tensor: The sparse regularization loss as a scalar tensor.
#     """
#
#     loss = 0
#     values = images
#     for module in model.children():
#         values = F.relu6(module(values))
#         loss += torch.mean(torch.abs(values))
#     return loss
import torch
import torch.nn as nn
def sparse_loss(model, images):
    """
    Computes the sparse regularization loss for a given model and input images.

    The sparse regularization loss encourages the activations of the model to be sparse,
    promoting the selection of a subset of relevant features and reducing the overall
    complexity of the model.
    """
    loss = 0
    values = images
    for module in model.children():
        if isinstance(module, nn.Linear):
            # Flatten the output of the previous layers before passing to the Linear layer
            values = torch.flatten(values, start_dim=1)
        values = module(values)
        if hasattr(module, 'activation'):  # Assuming that the ReLU6 is used as a named 'activation' attribute
            values = module.activation(values)
        loss += torch.mean(torch.abs(values))  # Calculate L1 norm for sparsity
    return loss
