import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.nn import LSTM, GRU, RNN, LayerNorm, BatchNorm1d
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from models import LongitudinalStacker, OCWCCE
import yaml
import argparse
import pickle as pkl
import os

def build_model(cell, input_size, hidden_state_sizes, dropout,
                regularization_layer, classifier, optimization, loss, gamma, batch):
    
    cell_type = {'LSTM': LSTM, 'GRU': GRU, 'RNN': RNN}
    reg_type = {'LayerNorm': LayerNorm, 'BatchNorm1d': BatchNorm1d}
    optimizer_type = {'SGD': SGD, 'Adam': Adam}
    loss_type = {'OCWCCE': OCWCCE}
    
    cell = cell_type[cell]
    reg_layer = reg_type[regularization_layer]
    optimizer = optimizer_type[optimization['optimizer']]
    loss = loss_type[loss]
    
    model = LongitudinalStacker(cell=cell, input_size=input_size,
            hidden_state_sizes=hidden_state_sizes, dropout=dropout, reg_layer=reg_layer, classifier=classifier)
    
    if loss == OCWCCE:
        loss_fn = OCWCCE(gamma)
    else:
        loss_fn = loss

    optim = optimizer(model.parameters(), lr=optimization['lr'])
    
    return model, loss_fn, optim, batch


#implement early stopping?
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

            model, loss_fn, optimizer, batch_size = build_model(**config_dict)

            train_dataset = TensorDataset(X_train, y_train, train_lengths)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(num_epochs):
                for X_batch, y_batch, lengths_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch, lengths_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                #ADD EARLY STOPPING FOR METRIC OF CHOOSING. FIRST ADD TENSORBOARD TO MONITOR TRAINING
                print(f'Epoch {epoch+1}: Training loss:', round(float(loss), 4))

            model.eval()
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