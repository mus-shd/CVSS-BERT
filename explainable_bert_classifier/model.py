import torch
import torch.nn.functional as F
import transformers
from transformers import BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
import copy
from explainable_bert_classifier.data import split_dataset





class BertClassifier():
    """
    BERT classifier object.
    """
    
    def __init__(self, model_name='prajjwal1/bert-small', **kwargs):
        """
        initialize a BERT model for sequence classification tasks
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = BertForSequenceClassification.from_pretrained(model_name, **kwargs)
        
        
    
    def freeze_base(self, verbose=True):
        """
        Freeze the base BERT model.
        """
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        if verbose==True:
            self.print_trainable_parameters()
            
                
    def unfreeze_base(self, verbose=True):
        """
        Unfreeze the base BERT model
        """
        for param in self.model.base_model.parameters():
            param.requires_grad = True
        if verbose==True:
            self.print_trainable_parameters()
                
                
    def print_trainable_parameters(self):
        """
        print the total number of parameters and how many of them are trainable
        """
        print("total number of trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print('trainable parameters:', name, param.data)
                
                
    def fit(self, loader, total_iterations, optim=AdamW, lr=5e-5):
        """
        Fit the BERT classifier to a specific dataset.
        Args:
            loader: Pytorch loader object to feed the model batch by batch
            total_iterations: total number of epochs to train for
            optim (default=AdamW): optimizer to use
            lr (default=5e-5): learning rate to use
        """
        optim = optim(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        self.model.train()
        
        training_loss_epoch = []
        training_loss_batch = []
        training_accuracy_epoch = []

        for epoch in range(total_iterations):
            training_loss = 0
            num_correct = 0 
            num_examples = 0
            for batch in loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['encoded_labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
                training_loss_batch.append(loss.data.item())
                training_loss += loss.data.item() * input_ids.size(0)
                correct = torch.eq(torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1], labels)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            
            training_loss /= len(loader.dataset)
            training_loss_epoch.append(training_loss)
            training_accuracy_epoch.append(num_correct / num_examples)        
    
            print('Epoch: {}, Training Loss: {}, Training Accuracy = {}'.format(epoch, training_loss, num_correct/num_examples))
        
        history = {'training_loss_per_epoch': training_loss_epoch, 'training_loss_per_batch': training_loss_batch, 'training_accuracy_per_epoch': training_accuracy_epoch}
        return history
        
    
    
    def evaluate_batch_by_batch(self, loader):
        """
        Evaluate data batch by batch (to avoid loading all the samples in memory when the dataset is large). Compute loss and accuracy.
        """
        self.model.to(self.device)
        self.model.eval()
        num_correct = 0 
        num_examples = 0
        test_loss = 0
        predicted_labels_list = []
        predicted_labels_score_list = []
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['encoded_labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            test_loss += loss.data.item() * input_ids.size(0)
            predicted_labels = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1]
            predicted_labels_list.extend(predicted_labels.tolist())
            predicted_labels_score = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[0]
            predicted_labels_score_list.extend(predicted_labels_score.tolist())
            correct = torch.eq(predicted_labels, labels)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        test_loss /= len(loader.dataset)
        accuracy = num_correct / num_examples

        print('Loss: {}, Accuracy = {}'.format(test_loss, accuracy))
        #print('predicted labels:', predicted_labels_list, 'scores:', predicted_labels_score_list)
    
        return {'predicted_labels': predicted_labels_list, 'predicted_scores': predicted_labels_score_list, 'accuracy': accuracy, 'loss': test_loss}
        
    
    
    def predict(self, batch_tokenized):
        """
        predict the labels for a batch of samples
        """
        self.model.to(self.device)
        self.model.eval()
        
        if torch.is_tensor(batch_tokenized['input_ids']):
            input_ids = batch_tokenized['input_ids'].to(self.device)
        else:
            input_ids = torch.tensor(batch_tokenized['input_ids']).to(self.device)
            
        if torch.is_tensor(batch_tokenized['attention_mask']): 
            attention_mask = batch_tokenized['attention_mask'].to(self.device)
        else:
            attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(self.device)
        
        if len(list(input_ids.size())) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        predicted_labels = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1]
        predicted_labels_score = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[0]
    
        return {'predicted_labels': predicted_labels, 'predicted_scores': predicted_labels_score}
            




def early_stopping(classifier, dataset, labels, encoded_labels, tokenizer, val_proportion=0.2, max_epoch=10, optim=AdamW, lr=5e-5):
    """
    Compute the optimal number of epochs using early stopping. Note that the classifier given as input is not modified. A copy of it is created instead.
    """
    classifier_copy =  BertClassifier(num_labels=len(set(labels)))
    classifier_copy.model = copy.deepcopy(classifier.model)
    analysis_dataset, assessment_dataset = split_dataset(dataset, labels, encoded_labels, tokenizer, val_proportion, shuffle=True)
    
    train_loader = DataLoader(analysis_dataset, batch_size=16)
    val_loader = DataLoader(assessment_dataset, batch_size=16)
    
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    for i in range(max_epoch):
        #print("Epoch ", i)
        #print("Training...")
        train_eval = classifier_copy.fit(loader=train_loader, total_iterations=1, optim=optim, lr=lr)
        history['train_acc'].append(train_eval['training_accuracy_per_epoch'][0])
        history['train_loss'].append(train_eval['training_loss_per_epoch'][0])
    
        
        print("Validation")
        val_eval = classifier_copy.evaluate_batch_by_batch(val_loader)
        history['val_acc'].append(val_eval['accuracy'])
        history['val_loss'].append(val_eval['loss'])
        
        if len(history['val_loss'])>=2:
            #print('testing condition')
            if history['val_loss'][-1]>history['val_loss'][-2]:
                break
    
    optimal_nb_epoch = len(history['val_loss'])-1
    print("optimal number of epochs: ", optimal_nb_epoch)
    return optimal_nb_epoch, history
        
    