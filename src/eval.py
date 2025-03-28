import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from accelerate.utils import set_seed
from data.dataset import DataModule
import json
from model.ProsTask import prosSouTask
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef



device='cuda' if torch.cuda.is_available() else 'cpu'

def create_parser():
    # parse arguments
    args_parser = ArgumentParser()

    #PATH
    args_parser.add_argument('--test_file_path', type=str, default=None)
    args_parser.add_argument('--pdb_folder', type=str, default=None)
    args_parser.add_argument('--pdb_index', type=str, default=None)
    args_parser.add_argument('--convbert_config_dir', type=str, default=r"/home/limc/gwh/SPRM-Sol/config/convbert/config.json")
    
    #model     
    args_parser.add_argument('--type', type=str, default='esm3')                            
    args_parser.add_argument('--dropout', type=float, default=0.25)
    args_parser.add_argument('--sequence_max_length', type=int, default=128)
    args_parser.add_argument('--classification_type',type=str,choices=['mlp','convbert','light_Attention','lstm'],default='convbert')
    
    #evaluation
    args_parser.add_argument('--seed', type=int, default=0)
    args_parser.add_argument('--lr', type=float, default=0.0001)
    args_parser.add_argument('--batch_size', type=int, default=16)
    args_parser.add_argument('--freeze', type=bool, default=True)
    args_parser.add_argument('--c_f', type=bool, default=True)
    args_parser.add_argument('--num_workers', type=int, default=0)
    args_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args_parser.add_argument('--sampling_num', type=dict, default={'train': 0, 'valid': 0, 'test': 100})

    
    # model loading
    args_parser.add_argument('--pretrained_path', type=str, default=None)

    # evaluation saving
    args_parser.add_argument('--save_pred_path', type=str, default='test_predictions.json')
    
    args = args_parser.parse_args()
    
    return args



def run():
    args = create_parser()
    set_seed(args.seed)
    
    metrics_dict = {
        "acc": BinaryAccuracy().to(device),
        "recall": BinaryRecall().to(device),
        "precision": BinaryPrecision().to(device),
        "mcc": BinaryMatthewsCorrCoef().to(device),
        "auroc": BinaryAUROC().to(device),
        "f1": BinaryF1Score().to(device),
    }
    for metric in metrics_dict.values():
        metric.reset()
    
    
    with open(args.pdb_index, 'r') as f:
        pdb_index = json.load(f)

   
    def get_protein_name_from_id(pid_tensor):
        pid = str(pid_tensor.item())  
        path = pdb_index.get(pid, None)
        if path is None:
            return None
        return os.path.splitext(os.path.basename(path))[0]  
    
    data_module = DataModule(file_path=args.test_file_path,
                             pdb_folder=args.pdb_folder, 
                             sampling_num=args.sampling_num,
                             type=args.type, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers,
                             seq_max_length=args.sequence_max_length,
                             pdb_index=args.pdb_index,
                             train_ratio=0, valid_ratio=0)
    _, _, test_dataset = data_module.forward()

    model=prosSouTask(freeze=args.freeze,
                        sequence_max_length=args.sequence_max_length,
                        dropout=args.dropout,
                        device=args.device,
                        classification_type=args.classification_type,
                        pdb_index=args.pdb_index,
                        convbert_dir=args.convbert_config_dir,
                        c_f=args.c_f)
    
    checkpoint = torch.load(args.pretrained_path, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(args.device)
    model.eval()

    print("模型权重加载完成，开始评估")
    
    # 存储最终结果
    sample_results = []
    

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataset, desc="Evaluating"):
            pdb_file_ids=batch[4].to(device)
            pred=model(batch)
            labels=batch[-1].float()
            probs=torch.sigmoid(pred)
            model_pred= (probs > 0.5).float()

            for name,metrics_fn in metrics_dict.items():
                metrics_fn.update(probs.to(device),labels.to(device))
                
            
            for i in range(len(pdb_file_ids)):
                name = get_protein_name_from_id(pdb_file_ids[i].to('cpu'))
                if name is None:
                    continue  
                sample_results.append({
                    'name': name,
                    'label': float(labels[i].item()),
                    'pred_label': float(model_pred[i].item())
                })
            
            
            

    # results
    print("\n========== Evaluation Results ==========")
    for name, metric_fn in metrics_dict.items():
        score = metric_fn.compute()
        print(f"{name.upper():<10}: {score:.4f}")
    print("========================================")
    
    # save results
    with open(args.save_pred_path, 'w') as f:
        json.dump(sample_results, f, indent=2)
    print(f"\n预测结果已保存至：{args.save_pred_path}")
            
        



if __name__ == '__main__':
    run()

