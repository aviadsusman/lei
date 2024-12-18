import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.nn import LSTM, GRU, RNN, LayerNorm, BatchNorm1d
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import unpad_sequence, pad_sequence
import torch.nn.functional as F
from models import LongitudinalStacker, OCWCCE, OCE, MEE, MPE, KLBeta
import numpy as np
import yaml
import argparse
import pickle as pkl
import os
from copy import deepcopy

def build_model(cell, input_size, hidden_state_sizes, dropout,
                regularization_layer, classifier, optimization, loss, gamma, batch, output_size):
    
    cell_type = {'LSTM': LSTM, 'GRU': GRU, 'RNN': RNN}
    reg_type = {'LayerNorm': LayerNorm, 'BatchNorm1d': BatchNorm1d}
    optimizer_type = {'Adam': Adam, 'SGD': SGD}
    loss_type = {'OCWCCE': OCWCCE, 'OCE': OCE, 'MEE': MEE, 'MPE': MPE, 'KLBeta': KLBeta}
    
    cell = cell_type[cell]
    reg_layer = reg_type[regularization_layer]
    optimizer = optimizer_type[optimization['optimizer']]
    loss = loss_type[loss]
    
    model = LongitudinalStacker(cell=cell, input_size=input_size,
            hidden_state_sizes=hidden_state_sizes, dropout=dropout, 
            reg_layer=reg_layer, classifier=classifier, output_size=output_size)
    
    loss_fn = loss(gamma)

    optim = optimizer(model.parameters(), lr=optimization['lr'])
    
    return model, loss_fn, optim, batch

def training_loop(path_to_data, device, config_dict, num_epochs=100):
    with open(path_to_data, "rb") as file:
        data = pkl.load(file=file)

    all_y_true = []
    all_y_pred = []
    #bring this outside function call and save after every split
    for split_idx, cv in enumerate(data):
        
        y_true = []
        y_pred = []

        for fold_idx, fold_data in enumerate(cv):
            print(f'\n~~~~Training and evaluating on CV split {split_idx+1}, fold {fold_idx+1}~~~~')
            X_train, y_train = fold_data['X_train'].to(device), fold_data['y_train'].to(device)
            X_val, y_val = fold_data['X_val'].to(device), fold_data['y_val'].to(device)
            X_test, y_test = fold_data['X_test'].to(device), fold_data['y_test'].to(device)
            train_lengths, val_lengths, test_lengths = fold_data['train lengths'].to('cpu'), fold_data['val lengths'].to('cpu'), fold_data['test lengths'].to('cpu')
            
            # When using KLBeta loss, we estimate two parameters rather than 3 probabilities.
            # Labels are true non-stationary beta parameters.
            if config_dict['loss'] == 'KLBeta':
                config_dict['output_size'] = 2
                
                def y_params(labels, lengths):
                    y_list = unpad_sequence(labels, lengths=lengths, batch_first=True)
                    params = [KLBeta.labels_to_beta(y) for y in y_list]
                    return pad_sequence(params, batch_first=True, padding_value=float('-inf')).to(device)
                
                y_train = y_params(y_train, train_lengths)
                y_val = y_params(y_val, val_lengths)
                y_test = y_params(y_test, test_lengths)

            else:
                config_dict['output_size'] = 3
            
            model, loss_fn, optimizer, batch_size = build_model(**config_dict)

            train_dataset = TensorDataset(X_train, y_train, train_lengths)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            
            # Early stopping setup. Add to configuration?
            patience = 5
            count = 0
            best_val_loss = float('inf')

            epoch = 0
            while count < patience:
                model.train()
                train_losses = []
                for X_batch, y_batch, lengths_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch, lengths_batch)
                    train_loss = loss_fn(outputs, y_batch)
                    train_loss.backward()
                    optimizer.step()
                    train_losses.append(float(train_loss))
                report = f'Epoch {epoch}: Train loss: {round(np.mean(train_losses), 4)}'
                
                # Early stopping
                if epoch >= 5:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(X_val, val_lengths)
                        val_loss = loss_fn(outputs, y_val)
                        if float(val_loss) < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = deepcopy(model.state_dict())
                            count = 0
                        else:
                            count += 1
                
                    report = report + f': Val loss: {round(float(val_loss), 4)}'
                
                print(report)
                epoch += 1

            model.load_state_dict(best_model_state)
            with torch.no_grad():
                outputs = model(X_test, test_lengths)
                predicted = F.softmax(outputs, -1)

                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(y_test.cpu().numpy())
    
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
    
    return all_y_true, all_y_pred

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load in data and model configs")
    parser.add_argument('--data', type=str, default='data/bps/no_sampling/40_5_24_with_mode_data/split_cv_tensors.pkl', help='Path to data directory.')
    parser.add_argument('--config', type=int, help='Which model config to use from yaml')

    args = parser.parse_args()
    data_path = args.data
    config = args.config
    with open("model_config.yaml", "rb") as file:
        config_dict = yaml.safe_load(file)[f'model_config_{config}']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    y_true, y_pred = training_loop(path_to_data=data_path, device=device, config_dict=config_dict, num_epochs=100) #change to early stopping
    
    results_path = data_path.replace('data/bps', 'results').replace('split_cv_tensors.pkl', f'config_{config}.pkl')
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results = {'y_true': y_true, 'y_pred': y_pred}

    with open(results_path, "wb") as file:
        pkl.dump(obj=results, file=file)