import torch.nn as nn
import torch 
from transformers import ConvBertModel, ConvBertConfig


class ClassifincationHead(nn.Module):
    def __init__(self,input_dim,type,dropout, num_classes,device,classification_type=None,convbert_dir=None,c_f=None,kernel_size=9):
        super(ClassifincationHead, self).__init__()
        self.num_classes=num_classes-1
        self.classification_type=classification_type
        self.device=device
        self.size=input_dim 
        self.convbert_dir=convbert_dir
        self.sigmod = nn.Sigmoid()

        if self.classification_type=='convbert':
            self.config=ConvBertConfig.from_pretrained(self.convbert_dir)
            self.encoder=ConvBertModel(config=self.config).to(self.device).encoder
            if c_f is True:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        elif self.classification_type=='mlp':
            self.pool=nn.AdaptiveAvgPool1d(1)
            
        elif self.classification_type=='light_Attention':
            self.feature_convolution = nn.Conv1d(self.size, self.size, kernel_size, stride=1,
                                             padding=kernel_size // 2)
            self.attention_convolution = nn.Conv1d(self.size, self.size, kernel_size, stride=1,
                                               padding=kernel_size // 2)
            self.softmax = nn.Softmax(dim=-1)
            self.conv_dropout = nn.Dropout(0.25)
            self.linear = nn.Sequential(
                nn.Linear(2 * self.size, 32),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            self.feature_linear= nn.Sequential(
                nn.Linear(self.size, 32),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )

            self.output = self.output = nn.Sequential(*[
                nn.Linear(32, self.num_classes),
            ])


        elif self.classification_type=='lstm':
            hidden_size=256
            self.lstm = nn.LSTM(self.size, hidden_size, bidirectional=True, batch_first=True)
            num_channels = [512,512,512]
            kernel_sizes = [9,6,3]
            self.conv_dropout = nn.Dropout(0.25)
            self.convs = nn.ModuleList()
            for c, k in zip(num_channels, kernel_sizes):
                self.convs.append(nn.Conv1d(hidden_size*2, c, k))
            self.feature_linear= nn.Sequential(
                nn.Linear(self.size, 512),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
            self.decoder = nn.Linear(sum(num_channels), self.num_classes)

        self.fc=nn.Sequential(
            nn.LayerNorm(self.size),
            nn.Dropout(dropout),
            nn.Linear(self.size,self.size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.size,self.num_classes)
        ).to(self.device)
        
        
        
    def forward(self, x,attention_mask=None,feature_embed=None):
        if self.classification_type=='convbert':
            encoder_outputs = self.encoder(
                hidden_states=x,
                attention_mask=attention_mask
            )
            hidden_state = encoder_outputs.last_hidden_state  # [batch_size, hidden_size]
            cls_output=hidden_state[:,0,:]  
            if feature_embed is not None:
                cls_output=cls_output+feature_embed
            logits=self.fc(cls_output)
            
        elif self.classification_type=='mlp':
            x=x.float()
            if x.dim()==3:
                x=x.permute(0,2,1)
                x=self.pool(x)
                x = x.squeeze(-1) 
            if feature_embed is not None:
                x=x+feature_embed
            logits=self.fc(x)
            
        elif self.classification_type=='light_Attention':
            x=x.float()
            x=x.permute(0,2,1)
            o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
            o = self.conv_dropout(o)
            attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
            attention = attention.masked_fill(attention_mask[:, None, :] == False, -1e9)
            o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
            o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
            o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
            o = self.linear(o)  # [batchsize, 32]
            if feature_embed is not None:
                feature_embed=self.feature_linear(feature_embed)
                o=o+feature_embed
            logits = self.output(o)  # [batchsize, num_classes]

        elif self.classification_type=='lstm':
            x=x.float()
            lstm_output, _ = self.lstm(x)
            lstm_output = lstm_output.permute(0, 2, 1)
            cnn_outputs = [conv(lstm_output) for conv in self.convs]
            pooled_outputs = [torch.max(cnn_output, dim=2)[0] for cnn_output in cnn_outputs]   ##batch_size,channels
            if feature_embed is not None:
                feature_embed=self.feature_linear(feature_embed)
                pooled_outputs+=feature_embed
            combined_features = torch.cat(pooled_outputs, dim=1)
            logits=self.decoder(self.conv_dropout(combined_features))

        logits=logits.squeeze(dim=1)

        return logits
