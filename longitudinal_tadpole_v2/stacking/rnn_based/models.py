import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, unpad_sequence
from torch.nn import RNN, LSTM, GRU, Sequential, Linear
from torch.nn import ReLU, Tanh, Softmax
from torch.nn import Dropout, BatchNorm1d, LayerNorm
import numpy as np
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    '''
    A general torch layer for time-distributed processing on packed sequence objects.
    '''
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, packed_seq):
        '''
        Pads with -inf assuming layer won't transform values to -inf.
        '''

        # Convert PackedSequence to flattened and unpadded 2D tensor.
        padded, lengths = pad_packed_sequence(packed_seq, batch_first=True, padding_value=float('-inf'))
        batch, pad_length, features = padded.shape
        flat = padded.view(-1, features) # flat.shape = (batch * pad_length, features)
        mask = (flat != float('-inf')).any(dim=1)
        flat_unpadded = flat[mask]

        # Apply the layer
        processed = self.layer(flat_unpadded)
        
        # Reassemble into PackedSequence
        processed_features = processed.shape[1]
        if processed_features <= features: # Reshape tensor for insertion of processed unpadded rows
            flat = flat[:, : processed_features] 
        else:
            flat = torch.cat([flat, flat[:,:processed_features-features]], dim=1)
        flat[mask] = processed
        processed_padded = flat.view(batch, pad_length, processed_features)
        processed_packed_seq = pack_padded_sequence(processed_padded, lengths=lengths, batch_first=True, enforce_sorted=False)

        return processed_packed_seq

class LongitudinalStacker(nn.Module):
    '''
    Cofigurable Longitudinal Stacker. Can customize architecture, cell type, regularization, and classification head.
    '''
    def __init__(self, cell, input_size, hidden_state_sizes, dropout, reg_layer, classifier):
        super(LongitudinalStacker, self).__init__()
        self.cell = cell
        self.input_size = input_size
        self.hidden_state_sizes = hidden_state_sizes
        self.dropout = dropout
        self.reg_layer = reg_layer
        self.classifier = classifier
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.layers = nn.ModuleList([
            cell(input_size if i == 0 else hidden_state_sizes[i-1], hidden_state_sizes[i], batch_first=True, dtype=torch.float64, device=self.device)
            for i in range(len(hidden_state_sizes))
        ])

        self.norm_layers = nn.ModuleList([
            TimeDistributed(self.reg_layer(hidden_state_sizes[i], dtype=torch.float64, device=self.device)) for i in range(len(hidden_state_sizes))
        ])
        
        if self.dropout != 0:
            self.recurrent_dropouts = nn.ModuleList([
                TimeDistributed(Dropout(p=self.dropout)) for i in range(len(hidden_state_sizes))
        ])
        
        if self.classifier == 'longitudinal':
            self.output_layer = cell(hidden_state_sizes[-1], 3, batch_first=True,  dtype=torch.float64, device=self.device)
        
        elif self.classifier == 'time distributed':
            h = hidden_state_sizes[-1]
            classifying_mlp = [Linear(h, h // 2, dtype=torch.float64, device=self.device),
                               Tanh(),
                               Dropout(self.dropout),
                               Linear(h // 2, h // 4, dtype=torch.float64, device=self.device),
                               Tanh(),
                               Dropout(self.dropout),
                               Linear(h // 4, 3, dtype=torch.float64, device=self.device)]
            self.output_layer = TimeDistributed(Sequential(*classifying_mlp))

    def forward(self, x, lengths):
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        for i, (norm, layer) in enumerate(zip(self.norm_layers, self.layers)):
            x, (h_n,c_n) = layer(x)
            x = norm(x)
            if self.dropout != 0:
                x = self.recurrent_dropouts[i](x)
        
        if self.classifier == 'longitudinal':
            x, (h_n,c_n) = self.output_layer(x)
        
        elif self.classifier == 'time distributed':
            x = self.output_layer(x)

        x = pad_packed_sequence(x, batch_first=True, padding_value=float('-inf'))[0]
        return x

# Helper functions for losses
def filter_padding(tensor):
    '''
    Return dense tensors for loss computation.
    '''
    if tensor.dim() == 1: # labels at time point
        return tensor[tensor > float('-inf')]

    elif tensor.dim() == 2: # logits at time point
        mask = (tensor > float('-inf')).all(dim=1)
        return tensor[mask]

def timed_class_weights(targets, classes=[0,1,2]):
    '''
    Get time-dependent class weights from batch.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timed_weights = []
    for y_time in torch.transpose(targets,0,1): #iterate over time slices
        y_time = filter_padding(y_time)
        y_time = torch.tensor([(y_time==class_label).sum() for class_label in classes])
        weights = sum(y_time) / (len(y_time) * y_time)
        timed_weights.append(weights.to(device, dtype=torch.long))

    return timed_weights

def ordinal_weights(pred, target, gamma=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_class = torch.argmax(pred, dim=1).to(torch.float64)

    target[target != 0] += gamma
    pred_class[pred_class != 0] += gamma

    ordinal_weight_tensor = torch.abs(pred_class - target) / (2 + gamma)
    return ordinal_weight_tensor.to(device)

class OCWCCE(nn.Module):
    def __init__(self, gamma=0):
        super(OCWCCE, self).__init__()
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, logits, targets):
        timepoints = logits.shape[1]
        class_weights = timed_class_weights(targets)

        loss = 0
        for t in range(timepoints):
            logits_t = filter_padding(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)

            targets_t = filter_padding(targets[:,t])
  
            wc_t = class_weights[t][targets_t.to(int)]

            if self.gamma is not None:
                wo_t = ordinal_weights(preds_t, targets_t, gamma=self.gamma)
            else:
                wo_t = torch.ones(len(targets_t), dtype=torch.float64, device=self.device) # tensor of all ones for ablating ordinal weights
            
            class_preds = preds_t[torch.arange(preds_t.size(0)), targets_t.to(int)]
            cce = -torch.log(class_preds)
            
            loss_t = (wo_t * wc_t * cce).mean()
            loss_t = cce.mean()
            loss += loss_t
        
        return loss / timepoints

class OCE(nn.Module):
    '''
    Ordinal Cross Entropy Loss. A log loss term for each class probability weighted by ordinal distance.
    The term for the true class is -log(x), all other terms are -log(1-x).
    Changing distance between adjacent classes is accounted for with tuneable gamma hyperparameter.
    '''
    def __init__(self, gamma=0):
        super(OCE, self).__init__()
        self.gamma = gamma
    
    def _adjust_proba(self, proba, true_class):
        ''' 
        A function for determining how to penalize a predicted probability.
        '''
        if true_class:
            return proba
        else:
            return 1 - proba

    def forward(self, logits, targets):
        batch, timepoints, classes = logits.shape
        class_weights = timed_class_weights(targets)
        
        loss = 0
        for t in range(timepoints):
            # Filter and activate
            logits_t = filter_padding(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)
            targets_t = filter_padding(targets[:,t])
            
            # Get batch length vector of class weights
            wc_t = class_weights[t][targets_t.to(int)].unsqueeze(1)
            
            # Construct matrix of shape (batch, classes) indicating true class of each sample
            true_class_mask = torch.zeros_like(logits_t).scatter_(1, targets_t.unsqueeze(1).to(torch.int64), 1.0)
            
            # Construct matrix of ordinal distances of shape (batch, classes).
            g = np.vectorize(lambda x: x+self.gamma if x!=0 else x)
            indices_t = torch.tensor(g(torch.arange(classes))).expand(len(targets_t), classes)
            adjusted_targets_t = torch.tensor(g(targets_t)).unsqueeze(1)
            ordinal_dist = (torch.abs(indices_t-adjusted_targets_t) / (classes - 1 + self.gamma) + 1) * (1-true_class_mask)

            log_true_class = -torch.log(preds_t)
            log_false_class = -torch.log(1 - preds_t)

            log_loss_matrix = wc_t * (true_class_mask * log_true_class + ordinal_dist * log_false_class)
            loss += torch.mean(torch.mean(log_loss_matrix, dim=1))
        
        return loss / timepoints
            

class MEE(nn.Module):
    '''
    Mean Expected Error Loss. Compute the MSE between true class and expected class index.
    Changing distance between adjacent classes is accounted for with tuneable gamma hyperparameter.
    '''
    def __init__(self, gamma=0):
        super(MEE, self).__init__()
        self.gamma = gamma
    

    def forward(self, logits, targets):
        batch, timepoints, classes = logits.shape
        class_weights = timed_class_weights(targets)
        
        loss = 0
        for t in range(timepoints):
            # Filter and activate
            logits_t = filter_padding(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)
            targets_t = filter_padding(targets[:,t])
            
            # Get batch length vector of class weights
            wc_t = class_weights[t][targets_t.to(int)].unsqueeze(1)
            
            g = np.vectorize(lambda x: x+self.gamma if x!=0 else x)
            ordinal_dist = torch.tensor(g(torch.arange(classes))).to(torch.float64)

            loss_t = torch.mean(wc_t * (preds_t @ ordinal_dist - targets_t)**2)
            loss += loss_t

        return loss / timepoints
            

            

            
            





# class TimeWeightedMSE(nn.Module):
#     def __init__(weighted=True):
#         super(TimeWeightedMSE).__init__()
    
#     def _class_weights_t(self, y_time):
#         y_time = self._label_count(self._filter_padding(y_time))
#         weights = sum(y_time) / (len(y_time) * y_time)

#         return weights.to(self.device, dtype=torch.long)
    
#     def _weighted_dist(true, pred, weighted=True, p=2):
#         if p==1:
#             #fix to make differentiable
#             pred = np.argmax(pred, axis=-1)
#             lp = np.abs(true-pred)
#         elif p==2:
#             pred = pred @ torch.tensor([0,1,2])
#             lp = torch.sqrt((true-pred)**2)
        
#         if weighted:
#             weights = torch.tensor([class_weights[i][int(value)] for i,value in enumerate(true)])
#             return torch.mean(lp * weights)
#         else:
#             return torch.mean(lp)

#     def forward(self, logits, targets):
#         timepoints = logits.shape[1]
#         class_weights = [self._class_weights_t(targets[:,t]) for t in range(timepoints)]

#         loss = 0
        