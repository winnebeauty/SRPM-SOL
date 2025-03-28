import torch 
import torch.nn as nn 
from transformers import (
    EsmModel, 
    AutoTokenizer, 
    BertTokenizer,
    BertModel,
    T5EncoderModel,
    T5Tokenizer,
    T5ForConditionalGeneration)
import re

from data.dataset import DataModule


class Encoder(nn.Module):
    def __init__(self,
                 plm_dir,
                 sequence_max_length=None,
                 freeze=False,
                 type=None,
                 device='cuda:1' if torch.cuda.is_available() else 'cpu',
                 classification_type=None):
        super(Encoder, self).__init__()
        self.type=type
        self.device=device
        self.classification_type=classification_type
        if type == 'esm':
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=plm_dir)
            self.plm_encoder = EsmModel.from_pretrained(pretrained_model_name_or_path=plm_dir)
            self.output_dim = self.plm_encoder.config.hidden_size
        if type == 'prot-bert':
            self.tokenizer = BertTokenizer.from_pretrained(plm_dir)
            self.plm_encoder = BertModel.from_pretrained(plm_dir).to(self.device)
            self.output_dim = self.plm_encoder.config.hidden_size
        if type == 'protTrans':
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=plm_dir)
            self.plm_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path=plm_dir).to(self.device)
            self.output_dim = self.plm_encoder.config.d_model
        if type == 'ankh':
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=plm_dir)
            self.plm_encoder = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=plm_dir)
            self.plm_encoder=self.plm_encoder.encoder
            self.output_dim = self.plm_encoder.config.d_model

        self.sequence_max_length = sequence_max_length
        if freeze:
            for param in self.plm_encoder.parameters():
                param.requires_grad = False
    
    def freeze(self):
        for param in self.plm_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        for param in self.plm_encoder.parameters():
            param.requires_grad = True
                    
    def get_per_protein_embedding(self,embedding_repr, attention_mask):
        """
        计算整个batch中每个序列的表示，去除填充和特殊标记。
        
        :param embedding_repr: 模型输出的嵌入表示，形状为 (batch_size, seq_length, hidden_size)
        :param attention_mask: 注意力掩码，形状为 (batch_size, seq_length)，表示每个位置是否为填充
        :return: 每条序列的整体表示，形状为 (batch_size, hidden_size)
        """
        batch_size, seq_length, hidden_size = embedding_repr.last_hidden_state.shape
        protein_embeddings = []
        
        # 遍历批次中的每个序列
        for i in range(batch_size):
            
            valid_token_mask = attention_mask[i].bool()  
            
            valid_embeddings = embedding_repr.last_hidden_state[i][valid_token_mask]
            
            protein_embedding = valid_embeddings.mean(dim=0)  # shape: (hidden_size)
            
            protein_embeddings.append(protein_embedding)
    
        return torch.stack(protein_embeddings)  

    
      
    def forward(self, batch):
        sequences = batch   
        #print(sequences)
        if self.type == 'esm':
            inputs=self.tokenizer(sequences,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True,
                                max_length=self.sequence_max_length
                                ).to(self.device)
            outputs=self.plm_encoder(**inputs,output_attentions=True,output_hidden_states=True)
            input_ids=inputs['input_ids']
            attention_mask=inputs['attention_mask']
            effective_lengths = attention_mask.sum(dim=1).long()  
            valid_positions = torch.zeros((input_ids.size(0), self.sequence_max_length), device=self.device)
            for i in range(valid_positions.size(0)):
                valid_positions[i, :effective_lengths[i]] = 1  
            if self.classification_type=='convbert':
                attention_mask = valid_positions.unsqueeze(1).unsqueeze(1) 
            else:
                attention_mask = valid_positions
            return outputs.last_hidden_state,attention_mask
        
        if self.type =='prot-bert' or self.type == 'ankh' or self.type == 'protTrans':
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]   
            inputs=self.tokenizer(sequences,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True,
                                max_length=self.sequence_max_length
                                ).to(self.device)
            if self.type == 'ankh' or self.type == 'protTrans':
                outputs=self.plm_encoder(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
            else:
                outputs=self.plm_encoder(**inputs)
            input_ids=inputs['input_ids']
            attention_mask=inputs['attention_mask']
            effective_lengths = attention_mask.sum(dim=1).long()  
            valid_positions = torch.zeros((input_ids.size(0), self.sequence_max_length), device=self.device)
            for i in range(valid_positions.size(0)):
                valid_positions[i, :effective_lengths[i]] = 1  
            if self.classification_type=='convbert':
                attention_mask = valid_positions.unsqueeze(1).unsqueeze(1) 
            else:
                attention_mask = valid_positions
            return outputs.last_hidden_state,attention_mask
        

        