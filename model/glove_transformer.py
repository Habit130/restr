from unicodedata import bidirectional
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from utils.torchutils import init_He
import pdb
import os.path as osp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from timm.models.layers import trunc_normal_
from model.transformers.utils import init_weights, positionalencoding1d, positionalencoding2d
from model.transformers.blocks import LangBlock
from utils.text_assets import get_dataset_text_spec
affine_par = True     

class TR_GloVe(nn.Module):
    # (batch_size, n, ) torch already know, you don't need to let torch know
    def __init__(self, 
        emb_name, 
        d_lang, 
        d_model, 
        n_layers,
        d_ff,
        n_heads,
        dropout=0.0,
        drop_path_rate=0.1,
        data_root='data',
        text_len=20,
        ):
        super().__init__()
        
        self.d_model = d_model
        text_spec = get_dataset_text_spec(emb_name, data_root=data_root)
        glove_np = np.load(text_spec["embedding_path"])
        print(f"Loaded embedding npy at {text_spec['embedding_path']}")
        self.glove = torch.from_numpy(glove_np)  # [vocab_size, 300]
        self.embedding = nn.Embedding.from_pretrained(self.glove, padding_idx=0, freeze=False)
        self.embedding.weight[0].data.zero_()

        # Positional embedding for TR
        self.register_buffer(
            "pos_emb_l",
            positionalencoding1d(d_lang, text_len).unsqueeze(0),
            persistent=False,
        )

        # TR
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [LangBlock(d_lang, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # Projection dim d_lang to d_model
        self.proj_lang = nn.Linear(d_lang, d_model)
        self.lang_norm = nn.LayerNorm(d_model)

        self.apply(init_weights)

    def forward(self, x):
        x = self.embedding(x) # y.shape = [B, C, 1, 1]

        pos_emb_l = self.pos_emb_l.to(x.device)
        x += pos_emb_l
        for blk in self.blocks:
            x = blk(x)

        x = self.proj_lang(x)
        x = self.lang_norm(x)
    
        return x
