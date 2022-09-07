""" Example on how to train on COCO from scratch
"""


import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os

#from detr_tf.data.coco import load_coco_dataset
from detr_tf.networks.detr import get_detr_model
from detr_tf.optimizers import setup_optimizers
from detr_tf.optimizers import gather_gradient, aggregate_grad_and_apply
from detr_tf.logger.training_logging import train_log, valid_log
from detr_tf.loss.loss import get_losses
from detr_tf.inference import numpy_bbox_to_image
from detr_tf.training_config import TrainingConfig, training_config_parser, CustomConfig
from detr_tf import training

#from detr_tf.data import load_wider
import logging
import sys

try:
    # Should be optional if --log is not set
    import wandb
except:
    wandb = None


import time

from random import shuffle
import pandas as pd
import imageio

from detr_tf import bbox
import numpy as np

from detr_tf.data import processing
from detr_tf.data.transformation import detr_transform

# Set the class name.
CLASS_NAMES = ["head"]
# Add the background class at the begining
CLASS_NAMES = ["background"] + CLASS_NAMES

def load_wider_data_from_index(index, class_names, filenames, train_val, anns, config, augmentation):
    # Open the image
    
    #image = imageio.imread(os.path.join(config.datadir, f"{train_val}", filenames[img_id]))
    image = imageio.imread(filenames[index])
    # Select all the annotatiom (bbox and class) on this image
    image_anns = anns[anns["filename"] == filenames[index]]    
    
    # Convert all string class to number (the target class)
    t_class = image_anns["class"].map(lambda x: class_names.index(x)).to_numpy()
    # Select the width&height of each image (should be the same since all the ann belongs to the same image)
    width = image_anns["width"].to_numpy()
    height = image_anns["height"].to_numpy()
    # Select the xmin, ymin, xmax and ymax of each bbox, Then, normalized the bbox to be between and 0 and 1
    # Finally, convert the bbox from xmin,ymin,xmax,ymax to x_center,y_center,width,height
    bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    bbox_list = bbox_list / [width[0], height[0], width[0], height[0]] 
    t_bbox = bbox.xy_min_xy_max_to_xcycwh(bbox_list)
    
    # Transform and augment image with bbox and class if needed
    image, t_bbox, t_class = detr_transform(image, t_bbox, t_class, config, augmentation=augmentation)

    # Normalized image
    image = processing.normalized_images(image, config)
            
    return image.astype(np.float32), t_bbox.astype(np.float32), np.ndarray.astype(np.expand_dims(t_class, axis=-1), np.int64)

#np.expand_dims(t_class, axis=-1)

def load_wider(train_val,batch_size, config, augmentation=False):
    """ Load the hardhat dataset
    """
    anns = pd.read_csv(f'notebooks/{train_val}.csv')
    anns['filename'] = anns['filename'].replace({'\.\./': ''}, regex=True)#to have the right paths
    print(anns)
    anns = anns.head(len(anns)//3)
    print("Number of faces taken :", len(anns))
    #taking only third of the complete dataset
    
    CLASS_NAMES = ["background"] + anns["class"].unique().tolist()
    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)
    
    # Set the background class to 0
    unique_class = anns["class"].unique()
    unique_class.sort()
    config.background_class = 0
    class_names = ["background"] + unique_class.tolist()

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(lambda idx: processing.numpy_fc(
        idx, load_wider_data_from_index, 
        class_names=class_names, filenames=filenames, train_val=train_val, anns=anns, config=config, augmentation=augmentation)
    ,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    

    # Filter labels to be sure to keep only sample with at least one bbox
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    dataset = dataset.map(processing.pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset, class_names


def build_model(config):
    """ Build the model with the pretrained weights. In this example
    we do not add new layers since the pretrained model is already trained on coco.
    See examples/finetuning_voc.py to add new layers.
    """
    # Load detr model without weight. 
    # Use the tensorflow backbone with the imagenet weights
    # detr = get_detr_model(config, nb_class=2, include_top=True, weights=None, tf_backbone=True)
    # detr.summary()

    #TransferLearning
    detr = get_detr_model(config, include_top=False, nb_class=2, weights="detr")
    #detr.load_weights("detr-learn-third10.ckpt")
    detr.summary()
    return detr


def run_finetuning(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)

    # Load the training and validation dataset
    # train_dt, coco_class_names = load_coco_dataset(
    #     config, config.batch_size, augmentation=True, img_dir="train2017", ann_fil="annotations/instances_train2017.json")
    # valid_dt, _ = load_coco_dataset(
    #     config, 1, augmentation=False, img_dir="val2017", ann_fil="annotations/instances_val2017.json")

    print("Loading WIDER dataset")
    train_dt, coco_class_names = load_wider("train", config.batch_size, config, 
        augmentation=False)
    valid_dt, _ = load_wider("val", config.batch_size, config, 
        augmentation=False)
    print("Finished loading WIDER dataset")

    # #Trasnfer Learning
    # config.train_backbone = tf.Variable(False)
    # config.train_transformers = tf.Variable(False)
    # config.train_nlayers = tf.Variable(True)

    #Trasnfer Learning
    config.train_backbone = tf.Variable(False)
    config.train_transformers = tf.Variable(True)
    config.train_nlayers = tf.Variable(True)

    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(detr, config)

    # Run the training for 100 epochs
    for epoch_nb in range(100):
        training.eval(detr, valid_dt, config, coco_class_names, evaluation_step=200)
        training.fit(detr, train_dt, optimzers, config, epoch_nb, coco_class_names)
        #Save the model
        detr.save_weights("detr-transfolearn-third.ckpt")


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = CustomConfig()
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    #config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    if config.log:
        wandb.init(project="detr-tensorflow", reinit=True)
        
    # Run training
    run_finetuning(config)
    # with open("test.out", 'w') as f:
    #     sys.stdout = f

    #     run_finetuning(config)





