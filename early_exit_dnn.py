import torchvision
import os, sys, time, math
from torchvision import transforms, utils, datasets
from PIL import Image
import torch, functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision.models as models
from pthflops import count_ops
from torch import Tensor
from typing import Callable, Any, Optional, List, Type, Union
#from .networks.mobilenet import MobileNetV2_2
#from .networks.resnet import resnet18, resnet152
#from .networks.vgg import vgg16_bn


class EarlyExitBlock(nn.Module):
  """
  This EarlyExitBlock allows the model to terminate early when it is confident for classification.
  """
  def __init__(self, input_shape, pool_size, n_classes, exit_type, device):
    super(EarlyExitBlock, self).__init__()
    self.input_shape = input_shape

    _, channel, width, height = input_shape
    self.expansion = width * height if exit_type == 'plain' else 1

    self.layers = nn.ModuleList()

    if (exit_type == 'bnpool'):
      self.layers.append(nn.BatchNorm2d(channel))

    if (exit_type != 'plain'):
      self.layers.append(nn.AdaptiveAvgPool2d(pool_size))
    
    #This line defines the data shape that fully-connected layer receives.
    current_channel, current_width, current_height = self.get_current_data_shape()

    self.layers = self.layers.to(device)

    #This line builds the fully-connected layer
    self.classifier = nn.Sequential(nn.Linear(current_channel*current_width*current_height, n_classes)).to(device)

  def get_current_data_shape(self):
    _, channel, width, height = self.input_shape
    temp_layers = nn.Sequential(*self.layers)

    input_tensor = torch.rand(1, channel, width, height)
    _, output_channel, output_width, output_height = temp_layers(input_tensor).shape
    return output_channel, output_width, output_height
        
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)
    output = self.classifier(x)
    return output


class Early_Exit_DNN(nn.Module):
  def __init__(self, model_name: str, n_classes: int, 
               pretrained: bool, n_branches: int, input_dim: int, 
               exit_type: str, device, distribution="linear"):
    super(Early_Exit_DNN, self).__init__()

    """
    This classes builds an early-exit DNNs architectures
    Args:

    model_name: model name 
    n_classes: number of classes in a classification problem, according to the dataset
    pretrained: 
    n_branches: number of branches (early exits) inserted into middle layers
    input_dim: dimension of the input image
    exit_type: type of the exits
    distribution: distribution method of the early exit blocks.
    device: indicates if the model will processed in the cpu or in gpu
    
    Note: the term "backbone model" refers to a regular DNN model, considering no early exits.

    """
    self.model_name = model_name
    self.n_classes = n_classes
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.input_dim = input_dim
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device
    self.temperature_overall = None
    self.temperature_branches = None
    self.temperature_all_samples = None


    build_early_exit_dnn = self.select_dnn_architecture_model()
    build_early_exit_dnn()

  def select_dnn_architecture_model(self):
    """
    This method selects the backbone to insert the early exits.
    """

    architecture_dnn_model_dict = {"mobilenet": self.early_exit_mobilenet,
                                   "resnet18": self.early_exit_resnet18,
                                   "vgg16": self.early_exit_vgg16,
                                   "resnet152": self.early_exit_resnet152}

    self.pool_size = 7 if (self.model_name == "vgg16") else 1
    return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)

  def select_distribution_method(self):
    """
    This method selects the distribution method to insert early exits into the middle layers.
    """
    distribution_method_dict = {"linear":self.linear_distribution,
                                "pareto":self.paretto_distribution,
                                "fibonacci":self.fibo_distribution}
    return distribution_method_dict.get(self.distribution, self.invalid_distribution)
    
  def linear_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a linear distribution.
    """
    flop_margin = 1.0 / (self.n_branches+1)
    return self.total_flops * flop_margin * (i+1)

  def paretto_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a pareto distribution.
    """
    return self.total_flops * (1 - (0.8**(i+1)))

  def fibo_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a fibonacci distribution.
    """
    gold_rate = 1.61803398875
    return total_flops * (gold_rate**(i - self.num_ee))

  def verifies_nr_exits(self, backbone_model):
    """
    This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    """
    
    total_layers = len(list(backbone_model.children()))
    if (self.n_branches >= total_layers):
      raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")

  def countFlops(self, model):
    """
    This method counts the numper of Flops in a given full DNN model or intermediate DNN model.
    """
    input = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
    flops, all_data = count_ops(model, input, print_readable=False, verbose=False)
    return flops

  def where_insert_early_exits(self):
    """
    This method defines where insert the early exits, according to the dsitribution method selected.
    Args:

    total_flops: Flops of the backbone (full) DNN model.
    """
    threshold_flop_list = []
    distribution_method = self.select_distribution_method()

    for i in range(self.n_branches):
      threshold_flop_list.append(distribution_method(i))

    return threshold_flop_list

  def invalid_model(self):
    raise Exception("This DNN model has not implemented yet.")
  def invalid_distribution(self):
    raise Exception("This early-exit distribution has not implemented yet.")

  def is_suitable_for_exit(self):
    """
    This method answers the following question. Is the position to place an early exit?
    """
    intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers))).to(self.device)
    x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
    current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
    return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

  def add_exit_block(self):
    """
    This method adds an early exit in the suitable position.
    """
    input_tensor = torch.rand(1, 3, self.input_dim, self.input_dim)

    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
    feature_shape = nn.Sequential(*self.stages)(x).shape
    self.exits.append(EarlyExitBlock(feature_shape, self.pool_size, self.n_classes, self.exit_type, self.device))#.to(self.device))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

  def set_device(self):
    """
    This method sets the device that will run the DNN model.
    """
    self.stages.to(self.device)
    self.exits.to(self.device)
    self.layers.to(self.device)
    self.classifier.to(self.device)

  def early_exit_resnet152(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    self.inplanes = 64

    n_blocks = 4

    backbone_model = models.resnet152(self.pretrained)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    self.threshold_flop_list = self.where_insert_early_exits()

    building_first_layer = ["conv1", "bn1", "relu", "maxpool"]
    for layer in building_first_layer:
      self.layers.append(getattr(backbone_model, layer))

    if (self.is_suitable_for_exit()):
      self.add_exit_block()

    for i in range(1, n_blocks+1):
      
      block_layer = getattr(backbone_model, "layer%s"%(i))

      for l in block_layer:
        self.layers.append(l)

        if (self.is_suitable_for_exit()):
          self.add_exit_block()
    
    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.classifier = nn.Sequential(nn.Linear(2048, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)
    self.set_device()

  def early_exit_alexnet(self):
    """
    This method inserts early exits into a Alexnet model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    # Loads the backbone model. In other words, Alexnet architecture provided by Pytorch.
    backbone_model = models.alexnet(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exit_alexnet(backbone_model.features)
    
    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    for layer in backbone_model.features:
      self.layers.append(layer)
      if (isinstance(layer, nn.ReLU)) and (self.is_suitable_for_exit()):
        self.add_exit_block()

    
    
    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))
    self.stages.append(nn.Sequential(*self.layers))

    
    self.classifier = backbone_model.classifier
    self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes, bias=True)
    self.softmax = nn.Softmax(dim=1)
    self.set_device()

  def verifies_nr_exit_alexnet(self, backbone_model):
    """
    This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    In AlexNet, we consider a convolutional block composed by: Convolutional layer, ReLU and he Max-pooling layer.
    Hence, we consider that it makes no sense to insert side branches between these layers or only after the convolutional layer.
    """

    count_relu_layer = 0
    for layer in backbone_model:
      if (isinstance(layer, nn.ReLU)):
        count_relu_layer += 1

    if (count_relu_layer > self.n_branches):
      raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")

  def early_exit_resnet18(self):
    """
    This method inserts early exits into a Resnet18 model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    self.inplanes = 64

    n_blocks = 4

    backbone_model = models.resnet18(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    building_first_layer = ["conv1", "bn1", "relu", "maxpool"]
    for layer in building_first_layer:
      self.layers.append(getattr(backbone_model, layer))

    if (self.is_suitable_for_exit()):
      self.add_exit_block()

    for i in range(1, n_blocks+1):
      
      block_layer = getattr(backbone_model, "layer%s"%(i))

      for l in block_layer:
        self.layers.append(l)

        if (self.is_suitable_for_exit()):
          self.add_exit_block()
    
    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.classifier = nn.Sequential(nn.Linear(512, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)
    self.set_device()


  def early_exit_resnet50_2(self):

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    backbone_model = models.resnet50(pretrained=True)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    x = torch.rand(1, 3, 224, 224)
    first_layers_list = ["conv1", "bn1", "relu", "maxpool"]
    for first_layer in first_layers_list:
      self.layers.append(getattr(backbone_model, first_layer))

    data = nn.Sequential(*self.layers)(x)

    n_layers = 4

    for n in range(1, n_layers+1):
      backbone_block = getattr(backbone_model, "layer%s"%(n))
      n_blocks = len(backbone_block)
      
      for j in range(n_blocks):
        bottleneck_layers = backbone_block[j]
        self.layers.append(bottleneck_layers)
        
        if (self.is_suitable_for_exit()):
          self.add_exit_block()

    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    self.classifier = nn.Sequential(nn.Linear(2048, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)


  def early_exit_resnet50(self):

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    backbone_model = models.resnet50(pretrained=True)

    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    x = torch.rand(1, 3, 224, 224)
    first_layers_list = ["conv1", "bn1", "relu", "maxpool"]
    for first_layer in first_layers_list:
      self.layers.append(getattr(backbone_model, first_layer))

    data = nn.Sequential(*self.layers)(x)

    bottleneck_list = ["conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "relu", "downsample"]

    bottleneck_short_list = bottleneck_list[:-1]
    n_layers = 4

    for n in range(1, n_layers+1):
      backbone_block = getattr(backbone_model, "layer%s"%(n))
      n_blocks = len(backbone_block)
      
      for j in range(n_blocks):
        bottleneck_layers = backbone_block[j]
        bottleneck_layers_list = bottleneck_list if (j==0) else bottleneck_short_list

        for layer in bottleneck_layers_list:
          temp_layer = getattr(bottleneck_layers, layer)
          if (layer == "downsample"):
            #pass
            self.layers.append(DownSample(temp_layer, data))
          else:
            self.layers.append(temp_layer)

          if (self.is_suitable_for_exit()):
            self.add_exit_block()
      
      data = backbone_block(data)

    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    self.classifier = nn.Sequential(nn.Linear(2048, self.n_classes))
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)


  def early_exit_vgg16(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0


    backbone_model = models.vgg16_bn(self.pretrained)
    backbone_model_features = backbone_model.features
    
    self.total_flops = self.countFlops(backbone_model)
    self.threshold_flop_list = self.where_insert_early_exits()

    for layer in backbone_model_features:
      self.layers.append(layer)
      if (self.is_suitable_for_exit()):
        self.add_exit_block()


    self.layers.append(backbone_model.avgpool)
    self.stages.append(nn.Sequential(*self.layers))
    self.classifier = backbone_model.classifier
    self.classifier[0] = nn.Linear(in_features=25088, out_features=4096)
    self.classifier[3] = nn.Linear(in_features=4096, out_features=4096)
    self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes)
    self.set_device()
    self.softmax = nn.Softmax(dim=1)


  def early_exit_inceptionV3(self):
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    backbone_model = models.inception_v3(self.pretrained)
    self.total_flops = self.countFlops(backbone_model)
    self.threshold_flop_list = self.where_insert_early_exits()

    architecture_layer_dict = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3",
                              "maxpool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3", "maxpool2",
                              "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d",
                              "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c", "avgpool", "dropout"]

    for block in architecture_layer_dict:
      layer_list.append(getattr(inception, block))
      if (self.is_suitable_for_exit()):
        self.add_exit_block()


    self.stages.append(nn.Sequential(*self.layers))
    self.classifier = backbone_model.fc
    self.set_device()
    self.softmax = nn.Softmax(dim=1)


  def early_exit_resnet56(self):

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    self.in_planes = 16
    n_layers = 3
    num_blocks =  [9, 9, 9]
    basic_block_list = ["conv1", "bn1", "relu", "conv2", "bn2"]

    backbone_model = ResNet(BasicBlock, num_blocks, num_classes=self.n_classes)
    
    self.total_flops = self.countFlops(backbone_model)
    self.threshold_flop_list = self.where_insert_early_exits()

    for i in range(1, n_layers + 1):
      intermediate_block_layer = getattr(backbone_model, "layer%s"%(i))

      for k in range(0, num_blocks[i-1]):

        basic_block = intermediate_block_layer[k]
        for layer in basic_block_list:
          self.layers.append(getattr(basic_block, layer))
          if (self.is_suitable_for_exit()):
            self.add_exit_block()

    

  def early_exit_mobilenet(self):
    """
    This method inserts early exits into a Mobilenet V2 model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    last_channel = 1280
    
    # Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
    backbone_model = models.mobilenet_v2(self.pretrained).to(self.device)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    self.verifies_nr_exits(backbone_model.features)
    
    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    for i, layer in enumerate(backbone_model.features.children()):
      
      self.layers.append(layer)    
      if (self.is_suitable_for_exit()):
        self.add_exit_block()

    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.stages.append(nn.Sequential(*self.layers))
    

    self.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(last_channel, self.n_classes),)

    self.set_device()
    self.softmax = nn.Softmax(dim=1)


  def get_processing_time(self, x, block):
    start = 0
    x = block(x)
    processing_time = time.time() - start
    return x, processing_time
  
  def measuring_inference_time_block_wise(self, x):

    inf_time_dict = {}
    inf_time, cont_block = 0, 0

    for i, exitBlock in enumerate(self.exits):

      for block in self.stages[i]:         
        x, processing_time = self.get_processing_time(x, block)
        inf_time += processing_time
        inf_time_dict["block_%s"%(cont_block)] = inf_time
        cont_block += 1

      output_branch, processing_time_branch = self.get_processing_time(x, exitBlock)
      inf_time += processing_time_branch
      inf_time_dict["exit_%s"%(i)] = inf_time

    x, processing_time = self.get_processing_time(x, self.stages[-1])
    inf_time += processing_time
    inf_time_dict["block_%s"%(cont_block)] = inf_time
    cont_block += 1

    if((self.model_name == "mobilenet") and (not self.pretrained)):
      pass
    else:
      x = torch.flatten(x, 1)

    output, processing_time = self.get_processing_time(x, self.classifier) 
    inf_time += processing_time
    inf_time_dict["block_%s"%(cont_block)] = inf_time

    print("Success")
    sys.exit()
    return inf_time_dict





