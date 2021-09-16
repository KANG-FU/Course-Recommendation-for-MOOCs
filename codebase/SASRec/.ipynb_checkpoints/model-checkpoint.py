import numpy as np
import torch


class Recommendation(torch.nn.Module):
    def __init__(self, user_num, item_num, hidden_units, num_heads, num_blocks, maxlen, dropout_rate, device='cpu'):
        super(Recommendation, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.num_blocks = num_blocks
    
        self.item_embedding = torch.nn.Embedding(self.item_num+1, hidden_units, padding_idx=0)
        self.pos_embedding = torch.nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        # Define the self-attentive layers, which will be expaned to multiple blocks.
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.conv1 = torch.nn.ModuleList()
        self.dropout1 = torch.nn.ModuleList()
        self.conv2 = torch.nn.ModuleList()
        self.dropout2 = torch.nn.ModuleList()
        
        
        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
        self.relu = torch.nn.ReLU()
        
        # define for each block
        self.attention_layernorms.extend([torch.nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(self.num_blocks)])
        self.attention_layers.extend([torch.nn.MultiheadAttention(hidden_units, num_heads, dropout_rate) for _ in range(self.num_blocks)])
        self.forward_layernorms.extend([torch.nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(self.num_blocks)])
        self.conv1.extend([torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) for _ in range(self.num_blocks)])
        self.dropout1.extend([torch.nn.Dropout(p=dropout_rate) for _ in range(self.num_blocks)])
        self.conv2.extend([torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) for _ in range(self.num_blocks)])
        self.dropout2.extend([torch.nn.Dropout(p=dropout_rate) for _ in range(self.num_blocks)])
        

    def nets(self, log_seqs):
        seqs = self.item_embedding(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_embedding.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_embedding(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        seqs_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~seqs_mask.unsqueeze(-1) 

        tl = seqs.shape[1] # time dim len for enforce causality
        attn_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(self.num_blocks):
            seqs = torch.transpose(seqs, 0, 1)
            query = self.attention_layernorms[i](seqs)
            attention_outputs, _ = self.attention_layers[i](query, seqs, seqs, attn_mask=attn_mask)

            seqs = query + attention_outputs # residual connection
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            ff_output = self.dropout2[i](
                         self.conv2[i](
                         self.relu(
                         self.dropout1[i](
                         self.conv1[i](seqs.transpose(-1, -2)))))) # Two-layer feed-forwad
            ff_output = ff_output.transpose(-1, -2) 
            seqs += ff_output # Residual connection
            seqs *= ~seqs_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) 

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs): # for training        
        output = self.nets(log_seqs) 

        pos_embeddings = self.item_embedding(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embeddings = self.item_embedding(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (output * pos_embeddings).sum(dim=-1)
        neg_logits = (output * neg_embeddings).sum(dim=-1)


        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, log_seqs, item_indices):
        output = self.nets(log_seqs)

        final = output[:, -1, :] 

        item_embeddings = self.item_embedding(torch.LongTensor(item_indices).to(self.dev)) 

        logits = item_embeddings.matmul(final.unsqueeze(-1)).squeeze(-1)
        return logits # preds # (U, I)
