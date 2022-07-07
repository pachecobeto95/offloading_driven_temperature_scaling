import os, torch

DIR_NAME = os.path.dirname(__file__)

models_id_dict = {"mobilenet": 1, "resnet18": 2, "vgg16": 1, "resnet152": 4}

dataset_name = "caltech256"

# Standard Configuration of the Arguments 
split_ratio = 0.2
input_dim = 224
seed = 42 # the answer to life the universe and everything
cuda = True
distribution = "linear" 
exit_type = "bnpool"
batch_size_train = 256
batch_size_test = 1
pretrained = True
n_branches = 5
n_epochs = 1000

# Parameters of Data Augmentation

#To normalize the input images data, according to ImageNet dataset.
mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
h_flip_prob = 0.25
rotation_angle = 25
brightness = (0.80, 1.20)


root_dataset_path = os.path.join(DIR_NAME, "dataset")
root_indices_path = os.path.join(DIR_NAME, "indices")

dataset_path_dict = {"caltech256": os.path.join(root_dataset_path, "caltech256")}
idx_path_dict = {"caltech256": os.path.join(root_indices_path, "caltech256")}
nr_class_dict = {"caltech256": 258}

