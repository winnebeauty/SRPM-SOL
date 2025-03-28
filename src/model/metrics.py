import torch
import torch.nn as nn
from torchmetrics.classification import(
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
    BinaryMatthewsCorrCoef
)

def calculate_acc(pred, targets,device):
    '''
    Calculate the accuracy of the model.
    '''
    
    pred=nn.Sigmoid()(pred)
    print(f'pred:{(pred>0.5).float()}\n targets: {targets}')
    accuracy=BinaryAccuracy().to(device)(pred,targets)
    return accuracy

def calculate_precision(pred, targets, device):
    '''
    Calculate the precision of the model.
    '''
    pred=nn.Sigmoid()(pred)
    precision=BinaryPrecision().to(device)(pred,targets)
    
    return precision

def calculate_recall(pred, targets, device):
    '''
    Calculate the recall of the model.
    '''
    pred=nn.Sigmoid()(pred)
    recall=BinaryRecall().to(device)(pred,targets)
    
    return recall

def calculate_f1_score(pred, targets, device):
    '''
    Calculate the F1 score of the model.
    '''
    pred=nn.Sigmoid()(pred)
    f1_score=BinaryF1Score().to(device)(pred,targets)
    return f1_score

def calculate_mcc(pred, targets, device):
    '''
    Calculate the Matthews Correlation Coefficient (MCC) of the model.
    '''
    pred=nn.Sigmoid()(pred)
    mcc=BinaryMatthewsCorrCoef().to(device)(pred,targets)
    
    return mcc


class Metrics:

    def __init__(self, num_classes
                 ,device
                 ):
        self.num_classes = num_classes
        self.device=device
        self.pred = torch.tensor([])
        self.target = torch.tensor([])

    def update(self, pred, target):
        self.pred = torch.cat([self.pred.to(self.device), pred.to(self.device)])
        self.target = torch.cat([self.target.to(self.device), target.to(self.device)])

    def compute(self):
        acc = calculate_acc(self.pred, self.target,self.device)
        precision = calculate_precision(self.pred, self.target,self.device)
        recall = calculate_recall(self.pred, self.target,self.device)
        f1 = calculate_f1_score(self.pred, self.target,self.device)
        mcc = calculate_mcc(self.pred, self.target,self.device)
        return {'acc': acc.item(), 'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item(), 'mcc': mcc.item()}

        
    def reset(self):
        self.pred = torch.tensor([])
        self.target = torch.tensor([])
