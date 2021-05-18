#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Massimo Caccia, Pau Rodriguez,        #
# Lorenzo Pellegrini. All rights reserved.                                     #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-02-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

Getting Started example for the CVPR 2020 CLVision Challenge. It will load the data and create the submission file
for you in the cvpr_clvision_challenge/submissions directory.

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os  # Usefull for file system and os manipulation
import time  # for tracking time it takes to complete work
import copy  # provides for shallow and deep copy opperations
from dataset import CORE50  # Imports the custom module for dealing with the core50 dataset
import torch  # Imports the torch library
import numpy as np  # imports the numpy library 
from train_test import train_net, test_multitask, preprocess_imgs
import torchvision.models as models  # provides access to pytorch compatable datasets
from common import create_code_snapshot  # custom utils in the util module inside the sub utils folder


def main(args):
    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic tracks"
    # and assuming the core50 location in ./core50/data/
    # ???review CORE50 to see if there is a way to shorten dataset and runtime for testing
    # Original call to dataset. Takes a long time. 
    # dataset = CORE50(root='core50/data/', scenario=args.scenario,
    #                  preload=args.preload_data)
    # 
    # custom call to create CORE50 custom object
    # using train=True uses training set and allows more control over batches and other stuff.
    dataset = CORE50(root='core50/data/', scenario=args.scenario,
                     preload=args.preload_data)

    # Get the validation set
    print("Recovering validation set...")
    # default full validation set 
    # full_valdidset = dataset.get_full_valid_set()
    # reduced validation set 
    full_valdidset = dataset.get_full_valid_set(reduced=True)

    # model
    if args.classifier == 'ResNet18':
        classifier = models.resnet18(pretrained=True)  # classifier is a pretrained model 
        classifier.fc = torch.nn.Linear(512,
                                        args.n_classes)  # in features: 512 # out features: set below -> args.n_classes = 50  #  Applies a linear transformation to the incoming data

    opt = torch.optim.SGD(classifier.parameters(), lr=args.lr)  # Implements stochastic gradient descent
    criterion = torch.nn.CrossEntropyLoss()  # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    # vars to update over time
    valid_acc = []
    ext_mem_sz = []
    ram_usage = []
    heads = []

    # loop over the training incremental batches (x, y, t)
    for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        # Print current batch number 
        print("----------- batch {0} -------------".format(i))
        # Print current batch shape
        print("x shape: {0}, y shape: {1}"
              .format(train_x.shape, train_y.shape))
        # print task label type
        print("Task Label: ", t)

        # train_net: a custom function to train neural network. returns stats.
        _, _, stats = train_net(
            opt, classifier, criterion, args.batch_size, train_x, train_y, t,
            args.epochs, preproc=preprocess_imgs
        )

        # if multi-task-nc: make deep copy in list heads (aka nn brains)
        if args.scenario == "multi-task-nc":
            heads.append(copy.deepcopy(classifier.fc))
        ext_mem_sz += stats['disk']
        ram_usage += stats['ram']

        # test all nn models in list heads for performance. return stats for each.
        stats, _ = test_multitask(
            classifier, full_valdidset, args.batch_size, preproc=preprocess_imgs, multi_heads=heads
        )

        # print new stats on performance 
        valid_acc += stats['acc']
        print("------------------------------------------")
        print("Avg. acc: {}".format(stats['acc']))
        print("------------------------------------------")

    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + args.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # copy code
    # custom function in utils folder to deal with possible file path issues 
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print("Training Time: {}m".format(elapsed))
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    # run final full test 
    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()
    stats, preds = test_multitask(
        classifier, full_testset, args.batch_size, preproc=preprocess_imgs
    )

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")

    print("Experiment completed.")


#  Code Starts running here
if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # Setup module to recieve arguments
    # General
    parser.add_argument('--scenario', type=str, default="multi-task-nc",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='ResNet18',
                        choices=['ResNet18'])

    # Optimization
    # --lr: Set Learning rate 
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    # --batch_size: set batch size 
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    # --epochs: 
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')

    args = parser.parse_args()
    args.n_classes = 50
    args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    main(args)
