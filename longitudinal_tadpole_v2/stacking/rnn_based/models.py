import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, unpad_sequence
from torch.nn import RNN, LSTM, GRU, Sequential, Linear
from torch.nn import ReLU, Tanh, Softmax
from torch.nn import Dropout, BatchNorm1d, LayerNorm
import numpy as np
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence

class TimeDistributed(nn.Module):
    '''
    A general torch wrapper for time-distributed processing on packed sequence objects.
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
    Cofigurable Longitudinal Stacker. 
    Can customize architecture, cell type, regularization, and classification head.
    '''
    def __init__(self, cell, input_size, hidden_state_sizes, dropout, reg_layer, classifier, output_size=3):
        super(LongitudinalStacker, self).__init__()
        self.cell = cell
        self.input_size = input_size
        self.hidden_state_sizes = hidden_state_sizes
        self.dropout = dropout
        self.reg_layer = reg_layer
        self.classifier = classifier
        self.output_size = output_size
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
            self.output_layer = cell(hidden_state_sizes[-1], self.output_size, batch_first=True,  dtype=torch.float64, device=self.device)
        
        elif self.classifier == 'time distributed':
            h = hidden_state_sizes[-1]
            classifying_mlp = [Linear(h, h // 2, dtype=torch.float64, device=self.device),
                               Tanh(),
                               Dropout(self.dropout),
                               Linear(h // 2, h // 4, dtype=torch.float64, device=self.device),
                               Tanh(),
                               Dropout(self.dropout),
                               Linear(h // 4, self.output_size, dtype=torch.float64, device=self.device)]
            
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
def unpad(tensor):
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
    for y_time in targets.T: #iterate over time slices
        y_time = unpad(y_time)
        y_time = torch.tensor([(y_time==class_label).sum() for class_label in classes])
        weights = sum(y_time) / (len(y_time) * y_time)
        timed_weights.append(weights.to(device))

    return timed_weights

def g(vec, gamma=0):
    '''
    Shifts ordinal representations to reflect true diagnosis relationships.
    '''
    vec[vec != 0] += gamma
    return vec

def ordinal_weights(pred, target, gamma=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_class = torch.argmax(pred, dim=1).to(torch.float64)

    target = g(target, gamma=gamma)
    pred_class = g(pred_class, gamma=gamma)
    # target[target != 0] += gamma
    # pred_class[pred_class != 0] += gamma

    ordinal_weight_tensor = torch.abs(pred_class - target) / (2 + gamma)
    return ordinal_weight_tensor.to(device)

def exp_arg(preds, gamma=0):
    '''
    A regularizing term for penalizing the distance between the expected class
    and the argmax of predicted probabilities. Specifically, this term prevents
    MCI from being the least predicted class (U shaped distributions).
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    indices = g(torch.arange(preds.shape[1]), gamma=gamma).to(device).double()
    exp = preds @ indices

    arg = torch.argmax(preds, dim=-1)

    # ranges from 0 to 1
    return ((exp - arg) / (preds.shape[1] - 1))**2

def retrogression(preds, t, t_prime):
    '''
    A weight for discouraging retrogressive predictions.
    Once the model predicts dementia (at time t_prime) it should predict
    dementia at all future time points. 
    '''
    return (1 + (torch.argmax(preds, dim=-1) - 2) / 2)**torch.max(torch.tensor([0, t-t_prime]))

class OCWCCE(nn.Module):
    '''
    Ordinally weighted categorical cross-entropy loss with class weighting across time.
    '''
    def __init__(self, gamma=0):
        super(OCWCCE, self).__init__()
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, logits, targets):
        timepoints = logits.shape[1]
        class_weights = timed_class_weights(targets)

        loss = 0
        for t in range(timepoints):
            logits_t = unpad(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)

            targets_t = unpad(targets[:,t])
  
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
    Ordinal Cross Entropy loss. A log loss term for each class probability weighted by ordinal distance.
    The term for the true class is -log(x), all other terms are -log(1-x).
    Changing distance between adjacent classes is accounted for with tuneable gamma hyperparameter.
    Similar to: https://aclanthology.org/2022.coling-1.407/ with term for positive class and variable ordinal distances.
    '''
    def __init__(self, ea_reg=False, gamma=0):
        super(OCE, self).__init__()
        self.gamma = gamma
        self.ea_reg = ea_reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
            logits_t = unpad(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)
            targets_t = unpad(targets[:,t])

            # Get batch length vector of class weights
            wc_t = class_weights[t][targets_t.to(int)].unsqueeze(1)

            # Construct matrix of shape (batch, classes) indicating true class of each sample
            true_class_mask = F.one_hot(targets_t.long(), classes)

            # Construct matrix of ordinal distances of shape (batch, classes).
            indices_t = g(torch.arange(classes), gamma=self.gamma).expand(len(targets_t), classes).to(self.device)
            adjusted_targets_t = g(targets_t, gamma=self.gamma).unsqueeze(1)
            wo_t = (torch.abs(indices_t-adjusted_targets_t) / (classes - 1 + self.gamma) + 1) * (1-true_class_mask)

            log_true_class = -torch.log(preds_t)
            log_false_class = -torch.log(1 - preds_t)
            
            # push wc_t outside reg term
            loss_t = wc_t * (true_class_mask * log_true_class + wo_t * log_false_class)#+ self.ea_reg * exp_arg(preds_t, gamma=self.gamma))
            loss += torch.mean(loss_t)

        return loss / timepoints
            

class MEE(nn.Module):
    '''
    Mean Expected Error loss. Compute the MSE between true class and expected class index.
    Changing distance between adjacent classes is accounted for with tuneable gamma hyperparameter.
    '''
    def __init__(self, ea_reg = False, gamma=0):
        super(MEE, self).__init__()
        self.gamma = gamma
        self.ea_reg = ea_reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, logits, targets):
        batch, timepoints, classes = logits.shape
        class_weights = timed_class_weights(targets)
        
        loss = 0
        for t in range(timepoints):
            # Filter and activate
            logits_t = unpad(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)
            targets_t = unpad(targets[:,t])
            
            # Get batch length vector of class weights
            wc_t = class_weights[t][targets_t.to(int)].unsqueeze(1)
            
            ordinal_dist = g(torch.arange(classes, dtype=torch.float64), gamma=self.gamma).to(self.device)
            targets_t = g(targets_t, gamma=self.gamma)
            loss_t = wc_t * ((preds_t @ ordinal_dist - targets_t)**2 + self.ea_reg * exp_arg(preds_t, gamma=self.gamma))
            loss += torch.mean(loss_t)

        return loss / timepoints

class MPE(nn.Module):
    '''
    Mean Predicted Error loss. Compute the MSE between true class and expected class index.
    Changing distance between adjacent classes is accounted for with tuneable gamma hyperparameter.
    Used as an ablation of the class expectation in the MEE loss.
    '''
    def __init__(self, ea_reg = False, gamma=0):
        super(MPE, self).__init__()
        self.gamma = gamma
        self.ea_reg = ea_reg

    def forward(self, logits, targets):
        batch, timepoints, classes = logits.shape
        class_weights = timed_class_weights(targets)

        loss = 0
        for t in range(timepoints):
            # Filter and activate
            logits_t = unpad(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)
            targets_t = unpad(targets[:,t])
            
            # Get batch length vector of class weights
            wc_t = class_weights[t][targets_t.to(int)]
            wo_t = ordinal_weights(preds_t, targets_t, gamma=self.gamma)

            loss_t = wc_t * (wo_t + self.ea_reg * exp_arg(preds_t, gamma=self.gamma))
            loss += torch.mean(loss_t)

        return loss / timepoints

class KLBeta(nn.Module):
    '''
    We represent the true diagnoses and predictions across time
    as the parameters of a non-stationary concave beta distribution
    reparameterized in terms of the mode, w, and concentration, c.
    mean = w + 1/c
    '''
    def __init__(self, gamma=0):
        super(KLBeta, self).__init__()
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _reparameterize(self, params):
        w,c = params[:, 0], params[:, 1]
        alpha = c * w + 1
        beta = c * (1 - w) + 1
        return alpha, beta
    
    def labels_to_beta(dx, classes=3, gamma=0):
        ''' 
        Function for transforming a label sequence into 
        parameters of a non-stationary beta distribution.
        Called into training file.

        If the dx is stationary between consecutive time points, 
        increase c and shift w towards the starting mode of the dx.
        (Change this)

        If the dx changes, reset w to the appropriate starting mode
        and reset c to the base confidence.
        '''
        # Update to include gamma shifting
        dx = dx.int().cpu().numpy()
        w_0 = [(c + 1) / (classes + 1) for c in range(classes)]
        c = [5]
        w = [w_0[int(dx[0])]]

        for t in range(1,len(dx)):
            if dx[t] == dx[t-1]:
                w.append(np.clip(w[t-1] + (dx[t] - 1) / 10, 0,1))
                c.append(c[t-1] + 1)
            else:
                w.append(w_0[dx[t]] + (dx[t-1] - dx[t]) / 10)
                c.append(5)
        return torch.tensor(np.stack([w, c, dx])).T

    def forward(self, logits, targets):
        batch, timepoints, params = logits.shape
        class_weights = timed_class_weights(targets[:, :, -1])

        loss = 0 
        for t in range(timepoints):
            targets_t = unpad(targets[:, t, :])
            logits_t = unpad(logits[:, t, :])
            preds_t = F.sigmoid(logits_t) 
            # Shift concentration to match true labels range [5, 5+T-1]
            preds_t = preds_t * torch.tensor([1,timepoints-1], device=self.device) + torch.tensor([0,5], device=self.device)

            wc_t = class_weights[t][targets_t[:,-1].to(int)]

            target_alpha, target_beta = self._reparameterize(targets_t[:, :-1])
            pred_alpha, pred_beta = self._reparameterize(preds_t)

            target_dist = Beta(target_alpha, target_beta)
            pred_dist = Beta(pred_alpha, pred_beta)

            loss_t = kl_divergence(target_dist, pred_dist)
            loss += torch.mean(wc_t * loss_t)
        
        return loss / timepoints