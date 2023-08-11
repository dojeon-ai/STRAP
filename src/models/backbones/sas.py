import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import TransformerBlock, Time2vecEncoding
from src.models.backbones.base import BaseBackbone
import ipdb
from einops import rearrange


class SAS(BaseBackbone):
    name='sas'
    def __init__(self,
                 categorical_ids,
                 n_blocks,
                 n_heads,                 
                 hid_dim,
                 p_drop,
                 max_seq_length,
                 use_categorical): 

        super().__init__(categorical_ids)
        self.max_seq_length = max_seq_length 
        self.hid_dim = hid_dim
        self.use_categorical = use_categorical

        # embedding
        cis = self.categorical_ids
        self.embed_transaction = nn.Linear(cis['num_trans_features'], hid_dim)
        self.embed_transaction_type = nn.Embedding(len(cis['transaction_type']), hid_dim, padding_idx=0)
        self.embed_province = nn.Embedding(len(cis['province']), hid_dim, padding_idx=0)
        self.embed_city = nn.Embedding(len(cis['city']), hid_dim, padding_idx=0)
        self.embed_county = nn.Embedding(len(cis['county']), hid_dim, padding_idx=0)
        self.embed_town = nn.Embedding(len(cis['town']), hid_dim, padding_idx=0)
        self.embed_building = nn.Embedding(len(cis['building_name']), hid_dim, padding_idx=0)
        self.embed_time = Time2vecEncoding(hid_dim, scale=1000)
        self.embed_price = nn.Linear(1, hid_dim)

        self.embed_poi = nn.Linear(cis['num_poi_features'], hid_dim)
        self.embed_trp = nn.Linear(cis['num_trp_features'], hid_dim)
        
        # transformer blocks
        self.in_norm = nn.LayerNorm(hid_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(n_heads, hid_dim, p_drop) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(p=p_drop)

        # init
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=(1/np.sqrt(self.hid_dim)))
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def get_output_dim(self):
        return self.hid_dim

    def forward(self, batch):
        transaction = batch['transaction'] 
        province = batch['province']
        city = batch['city']
        county = batch['county']
        town = batch['town']
        building = batch['building']
        transaction_type = batch['transaction_type']
        days_passed = batch['days_passed']
        price = batch['price'] 

        poi = batch['poi']
        trp = batch['trp']

        # [item, price, item, price, ...]
        # seq_len = max_seq_len * 2
        B, S, _ = transaction.shape

        ###############
        # Embedding
        transaction_embedding = self.embed_transaction(transaction) 
        transaction_type_embedding = self.embed_transaction_type(transaction_type)
        building_embedding = self.embed_building(building)
        province_embedding = self.embed_province(province)
        city_embedding = self.embed_city(city)
        county_embedding = self.embed_county(county)
        town_embedding = self.embed_town(town)

        price_embedding = self.embed_price(price.unsqueeze(-1))
        time_embedding = self.embed_time(days_passed.unsqueeze(-1))

        transaction_embedding = (transaction_embedding 
                                + transaction_type_embedding
                                + building_embedding
                                + time_embedding
        )

        if self.use_categorical:
            transaction_embedding = (
                transaction_embedding
                + province_embedding
                + city_embedding
                + county_embedding 
                + town_embedding
            )

        price_embedding = (price_embedding
                          + time_embedding 
        )

        poi = self.embed_poi(poi)
        trp = self.embed_trp(trp)
        

        # stack interchangeably
        # (t1, p1, ..., ts, ps)
        # transaction_embedding = torch.zeros_like(transaction_embedding).to(transaction_embedding.device)
        x = torch.stack((transaction_embedding, price_embedding), dim=1) 
        x = x.permute(0,2,1,3).reshape(B, 2*S, self.hid_dim)

        ###############
        # Forward
        # norm & droupout
        x = self.in_norm(x)
        x = self.dropout(x)

        # attn_mask.
        ones = torch.ones((2*S, 2*S)) 
        attn_mask = torch.tril(ones).view(1,1, 2*S, 2*S).to(x.device)

        for block in self.transformer_blocks:
            x = block(x, attn_mask)
        x = self.out_norm(x) 

        # x: (B, T*2, D) -> (B, 2, T, D)
        # x_out[:, 0]: (s0, s1, ..., sT)
        # x_out[:, 1]: (p0, p1, ..., pT)
        x = rearrange(x, 'b (s l) h -> b l s h', b=B, s=S, h=self.hid_dim)

        state = x[:, 0]
        price = x[:, 1]

        x = {
            'state': state,
            'price': price,
            'poi': poi,
            'trp': trp
        }

        return x