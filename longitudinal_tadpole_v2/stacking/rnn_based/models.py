import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, unpad_sequence
from torch.nn import RNN, LSTM, GRU, Sequential, Linear
from torch.nn import ReLU, Tanh, Softmax
from torch.nn import Dropout, BatchNorm1d, LayerNorm
import numpy as np
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    def __init__(self, layer, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)

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
            flat_reshaped = flat[:, : processed_features] 
        else:
            flat_reshaped = torch.cat([flat, flat[:,:processed_features-features]], dim=1)
        flat_reshaped[mask] = processed
        processed_padded = flat_reshaped.view(batch, pad_length, processed_features)
        processed_packed_seq = pack_padded_sequence(processed_padded, lengths=lengths, batch_first=True, enforce_sorted=False)

        return processed_packed_seq

class LongitudinalStacker(nn.Module):
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
            TimeDistributed(self.reg_layer, hidden_state_sizes[i], dtype=torch.float64, device=self.device) for i in range(len(hidden_state_sizes))
        ])
        
        if self.dropout != 0:
            self.recurrent_dropouts = nn.ModuleList([
                TimeDistributed(Dropout, p=self.dropout) for i in range(len(hidden_state_sizes))
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
            self.output_layer = TimeDistributed(Sequential, *classifying_mlp,)

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

class OCWCCE(nn.Module):
    def __init__(self, gamma=0):
        super(OCWCCE, self).__init__()
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _filter_padding(self, tensor):
        if tensor.dim() == 1: # labels at time point
            return tensor[tensor > float('-inf')]

        elif tensor.dim() == 2: # logits at time point
            mask = (tensor > float('-inf')).all(dim=1)
            return tensor[mask]

    def _label_count(self, array):
        return torch.tensor([(array==label).sum() for label in [0,1,2]])
        
    def _class_weights_t(self, y_time):
        y_time = self._label_count(self._filter_padding(y_time))
        weights = sum(y_time) / (len(y_time) * y_time)

        return weights.to(self.device, dtype=torch.long)
    
    def _ordinal_weights(self, pred, target, gamma):
        pred_class = torch.argmax(pred, dim=1).to(torch.float64)

        target[target != 0] += gamma
        pred_class[pred_class != 0] += gamma

        ordinal_weight_tensor = torch.abs(pred_class - target) / (2 + gamma)
        return ordinal_weight_tensor.to(self.device)
        
    def forward(self, logits, targets):
        timepoints = logits.shape[1]
        class_weights = [self._class_weights_t(targets[:,t]) for t in range(timepoints)]

        loss = 0
        for t in range(timepoints):
            logits_t = self._filter_padding(logits[:,t,:])  
            preds_t = F.softmax(logits_t, dim=-1)

            targets_t = self._filter_padding(targets[:,t])
  
            wc_t = class_weights[t][targets_t.to(int)]

            if self.gamma is not None:
                wo_t = self._ordinal_weights(preds_t, targets_t, gamma=self.gamma)
            else:
                wo_t = torch.ones(len(targets_t), dtype=torch.float64, device=self.device) # tensor of all ones for ablating ordinal weights
            
            class_preds = preds_t[torch.arange(preds_t.size(0)), targets_t.to(int)]
            cce = -torch.log(class_preds)
            
            loss_t = (wo_t * wc_t * cce).mean()
            loss_t = cce.mean()
            loss += loss_t
        
        return loss / timepoints

# class TimeDistributed(nn.Module):
#     def __init__(self, layer, *args, **kwargs):
#         super().__init__()
#         self.layer = layer(*args, **kwargs)

#     def forward(self, packed_seq):
#         padded_seq, seq_lengths = pad_packed_sequence(packed_seq, batch_first=True, padding_value=float('-inf'))
#         batch, longest_seq, features = padded_seq.shape
#         flat_seq = padded_seq.view(-1, features)
#         mask = (flat_seq > float('-inf')).all(dim=1)
#         flat_seq = flat_seq[mask]

#         processed_flat_seq = self.layer(flat_seq)

#         processed_list = []
#         start_idx = 0
#         for size in seq_lengths:
#             end_idx = start_idx + size
#             processed_list.append(processed_flat_seq[start_idx:end_idx])
#             start_idx = end_idx

#         padded_seq = pad_sequence(processed_list, batch_first=True, padding_value=float('-inf'))
#         packed_seq = pack_padded_sequence(padded_seq, lengths=seq_lengths, batch_first=True, enforce_sorted=False)

#         return packed_seq