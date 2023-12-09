import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import GPT2Config, GPT2Model
from tqdm.notebook import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class GPTRegression(nn.Module):
    def __init__(self, time_step, features_dim):
        super(GPTRegression, self).__init__()
        
        # Initialize GPT-2 model and config
        self.gpt_config = GPT2Config()
        self.gpt = GPT2Model(self.gpt_config)
        
        # Add an input layer to adapt continuous values to the model's expected input size
        self.input_layer = nn.Linear(time_step * features_dim, self.gpt_config.n_embd)
        
        # Regression head
        self.regression_head = nn.Linear(self.gpt_config.n_embd, 1)

    def forward(self, input_sequences):
        batch_size = input_sequences.shape[0]
        
        # Reshape the input data
        reshaped_input = input_sequences.view(batch_size, -1)
        
        # Adapt the continuous sequences to the model's expected input size
        input_data = self.input_layer(reshaped_input)
        
        gpt_output = self.gpt(inputs_embeds=input_data).last_hidden_state
        
        # Take the output from the last token in the sequence
        reg_output = self.regression_head(gpt_output.view(batch_size, -1))
        return reg_output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        # Scaled dot-product attention
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.embed_size**0.5
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = torch.nn.functional.softmax(attention, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(0.05),  # Replaced ReLU with LeakyReLU
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.norm1(attention + query)
        x = self.dropout(x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        out = self.dropout(out)
        return out

def positional_encoding(position, d_model):
    """
    Compute the positional encoding for a given position and model size.
    """
    def _calc_freq(position, i):
        return position / torch.pow(10000, 2 * (i // 2) / d_model)
    
    position = torch.arange(position).unsqueeze(1).float()
    div_term = torch.arange(0, d_model, 2).float()
    
    # Apply sin to even positions in the array; 2i
    pe = torch.zeros([position.shape[0], d_model])
    pe[:, 0::2] = torch.sin(_calc_freq(position, div_term))
    
    # Apply cos to odd positions in the array; 2i+1
    pe[:, 1::2] = torch.cos(_calc_freq(position, div_term))
    
    return pe

class GPT(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, max_length, dropout):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.model_type = 'GPT'
        self.classification_head = nn.Linear(embed_size, 1)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
        # Using the model's device
        device = next(self.parameters()).device
        self.pe = positional_encoding(max_length, embed_size).to(device)


    def forward(self, x, mask=None):
        N, seq_length, _ = x.shape
        
        out = self.dropout(x + self.pe[:x.size(1), :])  # Use sinusoidal positional encoding

        for layer in self.layers:
            out = layer(out, out, out, mask)

        # Take the output from the last token in the sequence
        seq_last_output = out[:, -1, :]
        logits = self.classification_head(seq_last_output)
        
        return logits
    
def binary_accuracy(predictions, targets):
    """
    Computes accuracy for binary classification.

    Args:
    - predictions (torch.Tensor): The model's predictions (after sigmoid and thresholding).
    - targets (torch.Tensor): The ground truth labels.

    Returns:
    - accuracy (float): The accuracy of the predictions with respect to the targets.
    """
    assert predictions.shape == targets.shape, "Predictions and targets should have the same shape"
    
    correct = (predictions == targets).float().sum().item()  # Count correct predictions
    total = targets.numel()  # Total number of predictions
    accuracy = correct / total

    return accuracy

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)): 
            # He initialization for ReLU activations
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:  # Bias should be initialized to zero
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)


# Helper function for undersampling
def undersample_data(train_x, train_y,close2close):
    positive_indices = np.where(train_y == 1)[0]
    negative_indices = np.where(train_y == 0)[0]
    
    num_samples = min(len(positive_indices), len(negative_indices))
    
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)
    
    undersampled_positive_indices = positive_indices[:num_samples]
    undersampled_negative_indices = negative_indices[:num_samples]
    
    undersampled_indices = np.concatenate([undersampled_positive_indices, undersampled_negative_indices])
    
    return train_x[undersampled_indices], train_y[undersampled_indices] , close2close[undersampled_indices]



def smooth_binary_labels_torch(labels, epsilon=0.1):
    """
    Apply label smoothing to binary labels using PyTorch.

    Args:
    - labels (torch.Tensor): Binary labels [0,1].
    - epsilon (float): Smoothing factor.

    Returns:
    - Smoothed labels.
    """
    return (1 - epsilon) * labels + epsilon * (1 - labels)