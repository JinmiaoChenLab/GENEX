"""
This file contains the explainable AI utils needed for xAI-GAN to work

Date:
    August 15, 2020

Project:
    XAI-GAN

Contact:
    explainable.gan@gmail.com
"""

import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from captum.attr import Saliency, NoiseTunnel, IntegratedGradients
from utils.vector_utils import values_target, images_to_vectors
from lime import lime_image
from torch.autograd import Variable
import os.path

from numpy import asarray
from numpy import savetxt

import shutil

# defining global variables
global values
global discriminatorLime

# torch.cuda.set_device(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



def get_explanation(generated_data, discriminator, prediction, target, XAItype="saliency", cuda=True, trained_data=None, data_type="mnist") -> None:
    """
    This function calculates the explanation for given generated images using the desired xAI systems and the
    :param generated_data: data created by the generator
    :type generated_data: torch.Tensor
    :param discriminator: the discriminator model
    :type discriminator: torch.nn.Module
    :param prediction: tensor of predictions by the discriminator on the generated data
    :type prediction: torch.Tensor
    :param XAItype: the type of xAI system to use. One of ("shap", "lime", "saliency")
    :type XAItype: strx
    :param cuda: whether to use gpu
    :type cuda: bool
    :param trained_data: a batch from the dataset
    :type trained_data: torch.Tensor
    :param data_type: the type of the dataset used. One of ("cifar", "mnist", "fmnist")
    :type data_type: str
    :return:
    :rtype:
    """
    
    # initialize temp values to all 1s
    temp = values_target(size=generated_data.size(), value=1.0, cuda=cuda)
    #print("temp", temp.shape)

    # mask values with low prediction
    #mask = (prediction < 0.5).view(-1)
    mask = [prediction[i] < 0.5 for i in range(np.shape(prediction)[0])]
    #print(np.shape(mask))
    indices = [ind for ind, val in enumerate(mask) if True in val] # technically get the index of the data that we want to apply explainable to
    
    #indices = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()
    
    data = generated_data
    #print(data.shape)
    
    

    def model_wrapper(inputs, target):
        discriminator.zero_grad()
        output, cls = discriminator(inputs)
        # element-wise multiply outputs with one-hot encoded targets 
        # and compute sum of each row
        # This sums the prediction for all markers which exist in the cell
        
        full_out = torch.cat((output, cls), -1)
        
        return torch.sum(full_out * target, dim=0)
#        return output, cls
    
    if len(indices) > 1:
        if XAItype == "IntegratedGradients":
#             for i in range(len(indices)):
#            D_B = Discriminator()
            explainer = IntegratedGradients(model_wrapper)
            temp_2 = explainer.attribute(data[indices, :].detach(), additional_forward_args = target[indices, :])
            temp_2 =  temp_2.double()
            temp[indices, :] = temp_2
                #print(temp[indices[i], :])
        elif XAItype == "smooth_IG":
            explainer = IntegratedGradients(model_wrapper)
            nt = NoiseTunnel(explainer)
            temp[indices, :] = nt.attribute(data[indices, :].detach(), additional_forward_args = target[indices, :], nt_type = "smoothgrad", stdevs = 0.02, draw_baseline_from_distrib = True, nt_samples_batch_size = 50)
                
        else:
            raise Exception("wrong xAI type given")

    if cuda:
        temp = temp.cuda()
    set_values(normalize_vector(temp))


def explanation_hook(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    # get stored mask
    temp = images_to_vectors(get_values())
    
    #print("temp", temp.shape, len(grad_input), grad_input[0].shape, len(grad_output))#, grad_input[1].shape, grad_input[2].shape)

    # multiply with mask to generate values in range [1x, 1.2x] where x is the original calculated gradient
    new_grad = grad_input[0] + 0.5 * (grad_input[0] * temp)
    
    #print(new_grad.shape)

    return (new_grad, )
#     return new_grad


def normalize_vector(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)


def get_values() -> np.array:
    """ get global values """
    global values
    return values


def set_values(x: np.array) -> None:
    """ set global values """
    global values
    values = x

