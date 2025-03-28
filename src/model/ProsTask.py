import torch
import json
import torch.nn as nn
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from esm.sdk.api import ESM3InferenceClient, ESMProtein
from model.classification import ClassifincationHead
from data.ConvertSeq import index_to_sequence



class prosSouTask(nn.Module):
    def __init__(self, 
                freeze=True,
                sequence_max_length=128,
                dropout=0.1,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                classification_type=None,
                pdb_index=None,
                convbert_dir=None,
                c_f=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.device=device
        self.model:ESM3InferenceClient=ESM3.from_pretrained("esm3_sm_open_v1").to(self.device)
        
        self.classification_type=classification_type
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.sequence_max_length=sequence_max_length
        self.dropout=dropout
        self.type='esm3'
        self.pdb_index=pdb_index
        self.convbert_dir=convbert_dir
        self.alpha=nn.Parameter(torch.tensor(1.0))
        self.beta=nn.Parameter(torch.tensor(0.0))
        self.classification=ClassifincationHead(input_dim=1536,type=self.type,dropout=self.dropout,num_classes=2,device=self.device,
                                                classification_type=self.classification_type,convbert_dir=self.convbert_dir,c_f=c_f).to(self.device)
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True    
    
    def forward(self, batch):
        seqs,coordinates,origin_length,gravy,pdb_file_id,labels=batch
        gravy = gravy.to(self.device)
        gravy = gravy.view(gravy.shape[0], 1, 1)
        gravy_expanded = gravy.expand(-1, self.sequence_max_length+2, 1536)
        seqs = seqs.to(self.device) 
        coordinates = coordinates.to(self.device) 
        origin_length = origin_length.to(self.device)  
        labels = labels.to(self.device) 
        a=nn.Parameter()
        with open(self.pdb_index, "r") as f:
            pdb_file_map=json.load(f)
        pdb_file_map = {int(k): v for k, v in pdb_file_map.items()}
        origin_length = [int(x) for x in origin_length.tolist()]  
        
        
        batch_seq=[]
        batch_sasa=[]
        batch_secondary_structure=[]
        batch_struct=[]
        batch_coords=[]
        batch_attention_mask=[]
        for i in range(seqs.shape[0]):
          
            seqs_org=seqs[i].tolist()
            seqs_org = seqs_org[:origin_length[i]]  
            seqs_org = index_to_sequence(seqs_org)    
            
            coords = coordinates[i, :origin_length[i]].tolist() 
            org_coords = torch.Tensor(coords).to(self.device) 
            
            for id,path in pdb_file_map.items():
                if id==pdb_file_id[i]:
                    pdb_file=path
                    break
            
             
            
            output=self.model.encode(ESMProtein().from_pdb(path=pdb_file,chain_id='A',is_predicted=True)).to(self.device)
            
            if output.sequence.shape[0]>self.sequence_max_length+2:     
                padding_sequence=output.sequence[:self.sequence_max_length+2]
                padding_structure=output.structure[:self.sequence_max_length+2]
                padding_coords=output.coordinates[:self.sequence_max_length+2]
                padding_sasa=output.sasa[:self.sequence_max_length+2]
                padding_secondary_structure=output.secondary_structure[:self.sequence_max_length+2]
                attention_mask = torch.ones(padding_sequence.shape[0], dtype=torch.long, device=self.device)
                
            elif output.sequence.shape[0]<self.sequence_max_length+2:
                padding_length = self.sequence_max_length + 2 - output.sequence.shape[0]
                padding_sequence = torch.cat([output.sequence, torch.ones(padding_length, dtype=torch.long, device=self.device)])
                padding_sasa=torch.cat([output.sasa, torch.zeros(padding_length, dtype=torch.long, device=self.device)])
                padding_secondary_structure=torch.cat([output.secondary_structure, torch.zeros(padding_length, dtype=torch.long, device=self.device)])
                padding_structure = torch.cat([output.structure, torch.full((padding_length,), 4099, dtype=torch.long, device=self.device)])
                padding_coords = torch.cat([output.coordinates, torch.full((padding_length, 37, 3), float('inf'), dtype=torch.float, device=self.device)])
                attention_mask = torch.cat([torch.ones(output.sequence.shape[0], dtype=torch.long, device=self.device), 
                                torch.zeros(padding_length, dtype=torch.long, device=self.device)])
                
            else:
                padding_sequence=output.sequence
                padding_sasa=output.sasa
                padding_secondary_structure=output.secondary_structure
                padding_structure=output.structure
                padding_coords=output.coordinates
                attention_mask = torch.ones(padding_sequence.shape[0], dtype=torch.long, device=self.device)
            
            batch_seq.append(padding_sequence.unsqueeze(0))
            batch_sasa.append(padding_sasa.unsqueeze(0))
            batch_secondary_structure.append(padding_secondary_structure.unsqueeze(0))
            batch_struct.append(padding_structure.unsqueeze(0))
            batch_coords.append(padding_coords.unsqueeze(0))
            batch_attention_mask.append(attention_mask.unsqueeze(0))
              
        model_output=self.model.forward(sequence_tokens=torch.cat(batch_seq,dim=0),
                                        sasa_tokens=torch.cat(batch_sasa,dim=0),
                                        ss8_tokens=torch.cat(batch_secondary_structure,dim=0),
                                        structure_tokens=torch.cat(batch_struct,dim=0),
                                        structure_coords=torch.cat(batch_coords,dim=0))
        
        model_output = self.alpha*model_output+self.beta* gravy_expanded   #fusion

        if self.classification_type=='mlp':
            embeddings = model_output[:, 1:-1, :]  
            output=self.pool(embeddings.permute(0,2,1)).squeeze(-1)
            pred=self.classification(output.to(self.device))
        elif self.classification_type=='convbert':
            output=model_output
            output=output.to(torch.float)
            pred=self.classification(output.to(self.device),torch.cat(batch_attention_mask,dim=0).unsqueeze(1).unsqueeze(1)  )
        elif self.classification_type=='light_Attention':
            output=model_output
            pred=self.classification(output.to(self.device),torch.cat(batch_attention_mask,dim=0)  )
        elif self.classification_type=='lstm':
            output=model_output
            pred=self.classification(output.to(self.device),torch.cat(batch_attention_mask,dim=0)  )
        
        return pred
        
