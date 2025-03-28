import os
import sys
import torch
import wandb
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from accelerate import Accelerator
from argparse import ArgumentParser
from accelerate.utils import set_seed
from torch.nn.functional import cross_entropy
from data.dataset import DataModule
from model.metrics import Metrics
from model.lora import lora_custom
from model.ProsTask import prosSouTask

sys.path.append('../')
sys.path.append('../../')


def create_parser():
    # parse arguments
    args_parser = ArgumentParser()

    #PATH
    args_parser.add_argument('--file_path', type=str, default=r'/home/limc/gwh/SPRM-Sol/PDE_Sol/json')
    args_parser.add_argument('--pdb_folder', type=str, default=r'/home/limc/gwh/SPRM-Sol/PDE_Sol/pdb')
    args_parser.add_argument('--pdb_index', type=str, default=r"/home/limc/gwh/SPRM-Sol/PDE_Sol/pdb_index.json")
    args_parser.add_argument('--convbert_config_dir', type=str, default=r"/home/limc/gwh/SPRM-Sol/config/convbert/config.json")
    
    #model     
    args_parser.add_argument('--type', type=str, default='esm3')                            
    args_parser.add_argument('--dropout', type=float, default=0.25)
    args_parser.add_argument('--sequence_max_length', type=int, default=128)
    args_parser.add_argument('--classification_type',type=str,choices=['mlp','convbert','light_Attention','lstm'],default='convbert')
    
    
    # train
    args_parser.add_argument('--seed', type=int, default=0)
    args_parser.add_argument('--epoch', type=int, default=2)
    args_parser.add_argument('--lr', type=float, default=0.0001)
    args_parser.add_argument('--lora', type=bool, default=False)
    args_parser.add_argument('--batch_size', type=int, default=4)
    args_parser.add_argument('--num_classes', type=int, default=2)
    args_parser.add_argument('--freeze', type=bool, default=False)
    args_parser.add_argument('--c_f', type=bool, default=False)
    args_parser.add_argument('--num_workers', type=int, default=0)
    args_parser.add_argument('--freeze_rounds', type=int, default=0)
    args_parser.add_argument('--lr_finetueing', type=float, default=0.00001)
    args_parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    args_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args_parser.add_argument('--sampling_num', type=dict, default={'train': 30000, 'valid': 30000, 'test': 30000})


    
    ##save model
    args_parser.add_argument('--mode',type=str,default='min')
    args_parser.add_argument('--patience', type=int, default=4)
    args_parser.add_argument("--model_name", type=str, default=None, help="model name")
    args_parser.add_argument("--model_dir", type=str, default="ckpt", help="model save dir")
    args_parser.add_argument("--epoch_idx", type=int, default=0, help="the idx of epoch to continue training")
    args_parser.add_argument("--auto_continue_train", type=str, default=False, help="auto extract epoch idx from history")
    
    #wandb
    args_parser.add_argument('--wandb_run_name', type=str, default=None)
    args_parser.add_argument('--wandb_project', type=str,default='PSS-Sol')
    
    args = args_parser.parse_args()
    
    return args


class Trainer():
    
    def __init__(self, model, args):
        super(Trainer, self).__init__()
        
        #params and train
        self.args = args
        self.model = model
        
        
        #train params
        self.lr = args.lr
        self.lora = args.lora
        self.freeze=args.freeze
        self.epoch = args.epoch
        self.device = args.device
        self.patience = args.patience
        self.num_classes = args.num_classes
        self.lr_finetueing = args.lr_finetueing
        self.freeze_rounds = args.freeze_rounds if args.freeze else args.epoch
        
        
        #metrics
        self.accelerator = Accelerator(
                                       gradient_accumulation_steps=args.gradient_accumulation_steps,
                                       )
        self.optimizer =  torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr,weight_decay=1e-2)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epoch, eta_min=args.lr/10)
        
        self.metrics = Metrics(num_classes=args.num_classes,device=self.device)
        self.loss_fn = cross_entropy if args.num_classes > 2 else nn.BCEWithLogitsLoss()
        
        #info
        self.base_info = f"{args.classification_type}_LORA_" if self.lora else f"{args.classification_type}_"
        if args.freeze_rounds > 0:
            self.base_info += f"DoubleIr_freeze{args.freeze_rounds}_"


           
    def train(self,data_module):
        
        #prepare data and model
        self.type = data_module.type
        train_dataset, valid_dataset, test_dataset = data_module.forward()
        self.model, self.optimizer, train_dataset = self.accelerator.prepare(
            self.model, self.optimizer, train_dataset
        )
        
        
        #model_info
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model_name=f'{self.type}_epoch{self.epoch}_lr{self.lr}_{params//1000}K'
        self.model_name=self.base_info+self.model_name

        #continue_training
        history={}
        monitor = 'valid/loss'
        start_epoch = 1
        model_path = os.path.join(self.args.model_dir, self.model_name)
        if  self.args.auto_continue_train:
            history_df = pd.read_csv(os.path.join(self.args.model_dir, "history.csv"))
            names = history_df.columns
            self.model.load_state_dict(torch.load(model_path)["state_dict"])
            if self.args.epoch_idx:
                print(f">>>>>>Train from epoch_idx = {self.args.epoch_idx} ")
            else:
                if self.args.mode == "min":
                    self.args.epoch_idx = int(history_df[history_df[monitor] == history_df[monitor].min()]["epoch"])
                elif self.args.mode == "max":
                    self.args.epoch_idx = int(history_df[history_df[monitor] == history_df[monitor].max()]["epoch"])
                print(f">>>>>>Auto continue to train from epoch_idx = {self.args.epoch_idx} ")
            for name in names:
                history[name] = list(history_df[name][:int(self.args.epoch_idx)])       
            start_epoch += self.args.epoch_idx
        
        
        ###training-valid-test
        for epoch in range(start_epoch,self.epoch+1):
            
            tl = []; vl = []
            self.model.train()
            with tqdm(total=len(train_dataset)) as pbar:
                pbar.set_description(f'Training Epoch {epoch}/{self.epoch}')
                for batch_idx, batch in enumerate(train_dataset):
                    tl.append(self.training_step(batch, batch_idx))
                    pbar.set_postfix({'current loss': sum(tl)/len(tl)})
                    pbar.update(1)
            wandb.log({f"train/loss": sum(tl)/len(tl), "train/epoch": epoch+1})
            print(f'>>>epoch {epoch}/{self.epoch} train loss:', sum(tl)/len(tl))
            
            
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_dataset):
                    vl.append(self.validation_step(batch, batch_idx))
                metrics = self.metrics.compute()
                for metric_name, metric_value in zip(
                    ['valid/acc', 'valid/precision', 'valid/recall', 'valid/f1', 'valid/mcc', 'valid/loss','epoch'],
                    [metrics['acc'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['mcc'], sum(vl)/len(vl),epoch]
                ):
                    history[metric_name] = history.get(metric_name, []) + [metric_value]
                wandb.log({f"valid/loss": sum(vl)/len(vl),"valid/acc":metrics['acc'],"valid/precision":metrics['precision'],"valid/recall":metrics['recall'],"valid/f1":metrics['f1'],"valid/mcc":metrics['mcc'],"valid/epoch": epoch})
                self.metrics.reset()
                print(f">>>epoch {epoch}/{self.epoch} val loss: {sum(vl)/len(vl)} \n>>>acc: {metrics['acc']}; precision: {metrics['precision']}\n>>>recall: {metrics['recall']}; f1: {metrics['f1']}; mcc: {metrics['mcc']}")
                
                ###early_stop and save model
                arr_scores = history[monitor]
                best_score_idx = np.argmax(arr_scores) if self.args.mode == "max" else np.argmin(arr_scores)
                if best_score_idx == len(arr_scores) - 1:
                    os.makedirs(model_path, exist_ok=True)
                    torch.save({
                        "state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "history": history,
                        }, os.path.join(model_path, "checkpoint.pth"))
                    print(f">>> reach best {monitor} : {'%.3f'%arr_scores[best_score_idx]}")
                
                history_df = pd.DataFrame(history)
                history_df.to_csv(os.path.join(self.args.model_dir, "history.csv"), index=False)
                
                if self.args.patience > 0 and len(arr_scores) - best_score_idx > self.args.patience:
                    print(f">>> {monitor} without improvement in {self.args.patience} epoch, early stopping")
                    break
                
                    
        print(f'>>>>>>training finished')        
        print(f'>>>>>>Testing......')
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataset):
                self.test_step(batch, batch_idx)
            metrics = self.metrics.compute()
            wandb.log({f"test/acc":metrics['acc'],"test/precision":metrics['precision'],"test/recall":metrics['recall'],"test/f1":metrics['f1'],"test/mcc":metrics['mcc'],"test/epoch": epoch+1})
            print(f">>>acc: {metrics['acc']}; precision: {metrics['precision']}; recall: {metrics['recall']}; f1: {metrics['f1']}; mcc: {metrics['mcc']}")
            
         
    
    def training_step(self, batch, batch_idx):
        with self.accelerator.accumulate(self.model):
            pred = self.model(batch).to(self.device) 
            loss=self.loss_fn(pred , batch[-1].float().to(self.device) )
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()    
        return round(loss.item(), 4)    


    def validation_step(self, batch, batch_idx):
        pred = self.model(batch).to(self.device)  
        loss=self.loss_fn(pred , batch[-1].float().to(self.device) )
        self.metrics.update(pred.to(self.device), batch[-1].to(self.device))
        return loss.item()
    
    def test_step(self, batch, batch_idx):
        pred=self.model(batch)
        self.metrics.update(pred.to(self.device), batch[-1].float().to(self.device))

    def save_model(self):
        save_path=f'checkpoint/{self.model_name}.pth'
        torch.save(self.model_state_dict, save_path)
        print(f'model saved at {save_path}')



def run():
    args = create_parser()
    set_seed(args.seed)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    #wandb init
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.type}"
        
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, 
                mode="offline",
                 config=vars(args))
    
    data_module = DataModule(file_path=args.file_path,
                             pdb_folder=args.pdb_folder, 
                             sampling_num=args.sampling_num,
                             type=args.type, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers,
                             seq_max_length=args.sequence_max_length,
                             pdb_index=args.pdb_index)
    
    model=prosSouTask(freeze=args.freeze,
                    sequence_max_length=args.sequence_max_length,
                    dropout=args.dropout,
                    device=args.device,
                    classification_type=args.classification_type,
                    pdb_index=args.pdb_index,
                    convbert_dir=args.convbert_config_dir,
                    c_f=args.c_f)
    
    
    all_params = sum(p.numel() for p in model.parameters()) // 1e6
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6
    print(f'>>>>>>Model Size: {all_params}M')
    print(f'>>>>>>Trainable Parameters: {params}m')
   
    trainer = Trainer(model, args)
    trainer.train(data_module)
    wandb.finish()
    
if __name__ == '__main__':
    
    run()      
        
    
    
