import wandb

from dataloader_places import PlacesDataset
import matplotlib as mpl
mpl.use('Agg')
from torch.utils.data import DataLoader
import torch
import torch.autograd as autograd
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch.nn as nn
import numpy as np
import torch.optim as optim
import tqdm
import torchvision

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class g_class(nn.Module):

    def __init__(self, feat_size=256):
        super(g_class, self).__init__()
        self.linear1 = spectral_norm(nn.Linear(feat_size, 64)) # was 1536
        self.linear2 = spectral_norm(nn.Linear(64, 32))
        self.linear3 = spectral_norm(nn.Linear(32, 16))
        self.fc = spectral_norm(nn.Linear(16, 1, bias=False))
        self.relu = nn.ReLU()
        self.ln1 = LayerNorm_SN([64])
        self.ln2 = LayerNorm_SN([32])
        self.ln3 = LayerNorm_SN([16])
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.relu(out)
        
        out = self.linear2(out)
        out = self.ln2(out)
        out = self.relu(out)
        
        out = self.linear3(out)
        out = self.ln3(out)
        out = self.relu(out)
        
        out = self.fc(out)
        out = 0.5 * (1 - self.tanh(out/10))
        return out


def q_train_epoch(device, model, model_g, dataloaders, optimizer, criterion, digit, phase):
    
    losses = AverageMeter()
    all_preds = []
    all_inputs = []
    all_groups = []
    model.eval()
    if phase == 'train':
        model_g.train()
    else:
        model_g.eval()
        
    
    #tqdm_loader = tqdm(dataloaders[digit])
    #for inputs, labels, groups in tqdm_loader:
    for data in tqdm(dataloaders[digit]):
        (inputs, labels, groups), name = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.no_grad():
            features = model.features(inputs)
            features = nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
            logits = model.last_linear(features)
            
        pred_q = model_g(features)
        if phase == 'train':
            optimizer.zero_grad()
        
        loss = criterion(logits, labels, pred_q)

        if phase == 'train':
            loss.backward()
            optimizer.step()
            
        losses.update(loss.item(), inputs.size(0))
        #tqdm_loader.set_postfix(loss=losses.avg)
        all_preds += list(pred_q.cpu().data.numpy())
        if phase == 'val':
            #all_inputs += list(inputs.cpu().data.numpy())
            all_groups += list(groups.cpu().data.numpy())
            #print(name)
            all_inputs += list(name)
    #print(all_inputs)
    return losses.avg, all_inputs, all_preds, all_groups


def train_epoch(device, model, dataloaders, criterion, optimizer, phase,
                batches_per_epoch=None):
    config = wandb.config
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_preds = []
    all_labels = []
    all_groups = []
    if phase == 'train':
        model.train()
    else:
        model.eval()
    #tqdm_loader = tqdm(dataloaders[phase])
    for data in tqdm(dataloaders[phase]):
        (inputs, labels, groups), name = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
           
            features = model.features(inputs)
            features = nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
            outputs = model.last_linear(features)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        #acc = torch.sum(preds == labels.data).item() / preds.shape[0]
        #accuracies.update(acc)
        all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
        all_labels += list(labels.cpu().data.numpy())
        all_groups += list(groups.cpu().data.numpy())
        #all_feats[name[0]] = outputs[0].cpu().detach().numpy()
        #tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    # Calculate multiclass AUC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    #auc = roc_auc_score(all_labels, all_preds[:, 1])
    all_labels_oh = np.eye(config.n_classes)[all_labels]
    
    avg_auc = 0
    for c in range(config.n_classes):
        auc = roc_auc_score(all_labels_oh[:,c], all_preds[:,c])
        print('label {} : auc {}'.format(c, auc))
        avg_auc += auc
    avg_auc = avg_auc / config.n_classes
    
    #for idx in range(len(all_names)):
    #        dict_preds[all_names[idx]] = all_preds[idx, 1]
    
    # Confusion Matrix
    #print('\nConfusion matrix')
    cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    #print(cmn)
    acc = np.trace(cmn) / cmn.shape[0]

    return losses.avg, avg_auc, acc



class EIILLoss(nn.Module):
    def __init__(self, gamma=0.01):
        super(EIILLoss, self).__init__()
        self.gamma = gamma

    def __call__(self, logits, labels, q_preds):
        # 'logits' of the fixed network, the one that is providing the spurious features
        # labels are w.r.t. to classes
        # g_preds are the predictions of the set prediction network.
        scale = torch.tensor(1.).cuda().requires_grad_()
        #l_phi = F.cross_entropy(logits * scale, labels, reduction='none')
        #print('l_phi: ', l_phi)
        q_preds = torch.flatten(q_preds)
        #print('q_preds: ', q_preds)
        #risk_e = torch.mul(l_phi,q_preds)
        #print('l_phi * q_preds: ', risk_e)
        R1 = torch.div(torch.sum(torch.mul(F.cross_entropy(logits * scale, labels, reduction='none'), q_preds)), torch.sum(q_preds))
        R2 = torch.div(torch.sum(torch.mul(F.cross_entropy(logits * scale, labels, reduction='none'), 1.0 - q_preds)), torch.sum(1.0 - q_preds))
        #risk_e = torch.mean(risk_e)
        #print('risk_e:', risk_e)
        #print('R1: {} | R2: {}'.format(R1, R2))
        #grad = autograd.grad(risk_e, [scale], create_graph=True)[0]
        #print('grad:', grad)
        #c_ei = torch.sum(grad**2)
        G1aux = torch.div(torch.sum(torch.mul(F.cross_entropy(logits * scale, labels, reduction='none'), q_preds)), torch.sum(q_preds))
        G2aux = torch.div(torch.sum(torch.mul(F.cross_entropy(logits * scale, labels, reduction='none'), 1.0 - q_preds)), torch.sum(1.0 - q_preds))
        
        #c_ei = torch.sum(grad**2)
        G1 = torch.sum(autograd.grad(G1aux, [scale], create_graph=True)[0]**2)
        G2 = torch.sum(autograd.grad(G2aux, [scale], create_graph=True)[0]**2)
        #print('c_ei:', c_ei)
        #return -1 * (risk_e + self.alpha * c_ei)
        #if (G1+G2) / (R1+R2+1e-12) < 0.2:
        #    self.gamma = 10.0
        #elif (G1+G2) / (R1+R2+1e-12) > 8.0:
        #    self.gamma = 0.1
        #else:
        #    self.gamma = 1.0
        #print(self.gamma, G1+G2, R1+R2)
        return -0.5 * (R1+R2 + self.gamma*(G1+G2))

def main():
    exp_name = "first_test"
    wandb.init(name=exp_name, project="eiil_test", entity='andiapsantos', save_code=True)

    config = wandb.config
    #comet logging

    config.model = model_name
    config.lr = lr
    config.batch_size = batch_size
    config.gamma = gamma
    config.epochs_part = epochs_part
    config.epochs = epochs
    config.epochs_fe = epochs_fe
    # config.fix_test_size = fix_test_size
    config.n_classes = n_classes
    config.load_model = load_model
    config.lr_part = lr_part
    
    data_path = "Places8_paths_and_labels_complete_train.npy"
    train_ds = PlacesDataset(data_path)
    val_ds = PlacesDataset(data_path)
    datasets = {
        '0': train_ds,
        '1': val_ds
    }
    
    # Dataloaders
    train_dl = DataLoader(datasets[0], batch_size=batch_size)
    val_dl = DataLoader(datasets[1], batch_size=batch_size )
    
    
    # Dataloaders per label (partition)
    data_sampler = None
    shuffle = True
    train_part_ds = [
        PlacesDataset(data_path, onlylabels=[k]) for k in range(n_classes)]
    train_dataloaders_class = {k: DataLoader(train_part_ds[k], batch_size=batch_size) for k in range(n_classes)}
    
    val_part_ds = [
        PlacesDataset(data_path, onlylabels=[k]) for k in range(n_classes)]
    val_dataloaders_class = {k: DataLoader(val_part_ds[k], batch_size=batch_size) for k in range(n_classes)}
    

    # Feature Extractor
    model = torchvision.models.resnet18(pretrained=True)
    
    # Train Feature Extractor 
    # Train especialized Nets  (partitions - extract biases/reference models?)

