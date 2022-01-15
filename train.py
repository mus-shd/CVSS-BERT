#!/usr/bin/python3

import sys, getopt
import os
import pandas as pd
import numpy as np
from explainable_bert_classifier.data import fit_transform_LabelEncoder
from explainable_bert_classifier.data import tokenizer
from explainable_bert_classifier.data import CVEDataset
from explainable_bert_classifier.model import BertClassifier
from explainable_bert_classifier.model import early_stopping
from torch.utils.data import DataLoader
from transformers import AdamW

"""
Script to train CVSS-BERT classifiers
"""


def main(argv):
    input_data = ''
    output_dir = 'bert-classifier'
    metric_name = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:m:",["input_data=","output_dir=","metric_name="])
    except getopt.GetoptError:
        print ('train.py -i <input_data> -o <output_dir> -m <metric_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('train.py -i <input_data> -o <output_dir> -m <metric_name>')
            sys.exit()
        elif opt in ("-i", "--input_data"):
            input_data = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-m", "--metric_name"):
            metric_name = arg

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir+'/'+metric_name):
        os.makedirs(output_dir+'/'+metric_name)
    
    
    X_train = pd.read_csv(input_data+'_X_train.csv')
    y_train = pd.read_csv(input_data+'_y_train.csv')
    train_labels = y_train.loc[:, metric_name]
    encoded_train_labels = fit_transform_LabelEncoder(train_labels, save=True, filename=output_dir+'/'+metric_name+'/label.txt')

    mytokenizer = tokenizer()
    train_encodings = mytokenizer(X_train.loc[:,"Description"].tolist(), truncation=True, padding=True, max_length=128)
    train_dataset = CVEDataset(X_train, train_encodings, train_labels, encoded_train_labels)    

    print('Loading model...')
    NUM_CLASSES = len(set(train_labels))
    classifier =  BertClassifier(num_labels=NUM_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print('Freeze base model')
    classifier.freeze_base(verbose=False)
    print('Training...')
    classifier.fit(loader=train_loader, total_iterations=2)
    print('Unfreeze base model')
    classifier.unfreeze_base(verbose=False)
    print('Determining optimal number of training epochs using early stopping...')
    optimal_nb_epoch, history_early_stopping = early_stopping(classifier, X_train, train_labels, encoded_train_labels, mytokenizer, max_epoch=8)
    print('Optimal number of training epoch: ', optimal_nb_epoch)
    print('Training...')
    classifier.fit(loader=train_loader, total_iterations=optimal_nb_epoch)
    
    print('Saving model...')
    classifier.model.save_pretrained(output_dir+'/'+metric_name+'/model')
    print('End')

    

if __name__ == "__main__":
    main(sys.argv[1:])