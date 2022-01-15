import torch
from sklearn.preprocessing import LabelEncoder
import pickle
from transformers import BertTokenizerFast
import numpy as np


def fit_transform_LabelEncoder(labels, save=False, filename='label.txt'):
     """
    Encode target labels with value between 0 and n_classes-1. Fit label encoder and return encoded labels.
    Args:
        labels: list of categorical labels to encode
        save (default False): if True save the list that holds the label for each class.
        filename: path to the file to save to.
    """
    le = LabelEncoder()
    le.fit(labels)
    NUM_CLASSES = len(le.classes_)
    print("total number of classes:", NUM_CLASSES)
    print("classes:", le.classes_)

    if save==True:
        with open(filename, "wb") as f:
            pickle.dump(le.classes_, f)

    encoded_train_labels = le.transform(labels)
    
    return encoded_train_labels


def train_test_LabelEncoder(train_labels, test_labels, save=False, filename='label.txt'):
     """
    Encode target labels with value between 0 and n_classes-1. Fit label encoder and return encoded labels on the training set.
    Also return the encoded labels on the test set.
    Args:
        train_labels: list of categorical training labels to encode
        test_labels: list of categorical test labels to encode
        save (default False): if True save the list that holds the label for each class.
        filename: path to the file to save to.
    """
    le = LabelEncoder()
    le.fit(train_labels)
    NUM_CLASSES = len(le.classes_)
    print("total number of classes:", NUM_CLASSES)
    print("classes:", le.classes_)

    if save==True:
        with open(filename, "wb") as f:
            pickle.dump(le.classes_, f)

    encoded_train_labels = le.transform(train_labels)
    encoded_test_labels = le.transform(test_labels)
    
    return encoded_train_labels, encoded_test_labels



def tokenizer(tokenizer_name='prajjwal1/bert-small', **kwargs):
    """
    initialize a tokenizer.
    """
    return BertTokenizerFast.from_pretrained(tokenizer_name, **kwargs)


def split_dataset(dataset, labels, encoded_labels, tokenizer, val_proportion=0.2, shuffle=True):
    """
    split dataset, labels, encoded_labels and return two dataset objects: one for training and another for validation/testing.
    Args:
        dataset: dataset to split
        labels: labels to split
        encoded_labels: corresponding encoded labels to split
        tokenizer: tokenizer to use for tokenization
        val_proportion (default=0.2): proportion of the data to be used for the validation dataset
        shuffle (default=True): if True shuffle before splitting the data
        
    """
    dataset_size = dataset.shape[0]
    #print("size: ", dataset_size)
    indices = np.arange(dataset_size)
    if shuffle==True:
        np.random.shuffle(indices)
    #print("indices: ", indices.shape, indices)
    split_index = int(val_proportion*dataset_size)
    val_indices = indices[:split_index]
    train_indices = indices[split_index:]
    #print("train indices: ", train_indices.shape, train_indices)
    #print("val indices: ", val_indices.shape, val_indices)
    number_of_common_indices = [1 for i in val_indices if i in train_indices]
    #print(number_of_common_indices)
    
    X_train = dataset.iloc[train_indices,:]
    X_val = dataset.iloc[val_indices,:]
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    
    encoded_train_labels = encoded_labels[train_indices]
    encoded_val_labels = encoded_labels[val_indices]

    train_encodings = tokenizer(X_train.loc[:,"Description"].tolist(), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(X_val.loc[:,"Description"].tolist(), truncation=True, padding=True, max_length=128)
    
    
    train_dataset = CVEDataset(X_train, train_encodings, train_labels, encoded_train_labels)
    val_dataset = CVEDataset(X_val, val_encodings, val_labels, encoded_val_labels)
    
    return train_dataset, val_dataset


class CVEDataset(torch.utils.data.Dataset):
    """
    CVEDataset object to handle CVE vulnerability description data for training and testing using PyTorch
    """
    def __init__(self, X, encodings, labels, encoded_labels):
        """
        Args:
        X: CVE vulnerability description dataset. Must contain two columns: "CVE_ID" and the actual "Description".
        encodings: tokenized representation of the descriptions contained in X
        labels: the text form labels of the description contained in X
        encoded_labels: corresponding encoded labels
        """
        self.cve_id = X.loc[:,"CVE_ID"].tolist()
        self.texts = X.loc[:,"Description"].tolist()
        self.encodings = encodings
        self.labels = labels.tolist()
        self.encoded_labels = encoded_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['text_labels'] = self.labels[idx]
        item['encoded_labels'] = torch.tensor(self.encoded_labels[idx])
        item['CVE_ID'] = self.cve_id[idx]
        item['vulnerability_description'] = self.texts[idx]
        
        return item

    def __len__(self):
        return len(self.labels)