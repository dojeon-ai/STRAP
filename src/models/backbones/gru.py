import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models.layers import TransformerBlock, Time2vecEncoding, PositionalEncoding
from src.models.backbones.base import BaseBackbone
import ipdb
from einops import rearrange


class GRU(BaseBackbone):
    name = 'gru'
    def __init__(self,
                 categorical_ids,
                 hid_dim,
                 batch_size,
                 max_seq_length,
                 mask_input):
        
        super().__init__(categorical_ids)
        self.max_seq_length = max_seq_length
        self.hid_dim = hid_dim
        self.batch_size = batch_size
        self.mask_input = mask_input

        # embedding
        cis = self.categorical_ids
        self.embed_transaction = nn.Linear(cis['num_trans_features'], hid_dim)
        self.embed_transaction_type = nn.Embedding(len(cis['transaction_type']), hid_dim, padding_idx=0)        
        self.embed_time = Time2vecEncoding(hid_dim, scale=1000)
        self.embed_price = nn.Linear(1, hid_dim)
        
        self.embed_province = nn.Embedding(len(cis['province']), hid_dim, padding_idx=0)
        self.embed_city = nn.Embedding(len(cis['city']), hid_dim, padding_idx=0)
        self.embed_county = nn.Embedding(len(cis['county']), hid_dim, padding_idx=0)
        self.embed_town = nn.Embedding(len(cis['town']), hid_dim, padding_idx=0)
        self.embed_building = nn.Embedding(len(cis['building_name']), hid_dim, padding_idx=0)

        self.embed_poi = nn.Linear(cis['num_poi_features'], hid_dim)
        self.embed_trp = nn.Linear(cis['num_trp_features'], hid_dim)

        ## GRU
        self.in_norm = nn.LayerNorm(hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim, num_layers=2, batch_first=True)
        self.out_norm = nn.LayerNorm(hid_dim)
        
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
        transaction = batch['transaction']  # torch.Size([1484, 50, 24]): 1484 from neighbor residents (5 per house)
        transaction_type = batch['transaction_type']    # torch.Size([1484, 50])        
        days_passed = batch['days_passed']
        price = batch['price'] 
        
        province = batch['province']
        city = batch['city']
        county = batch['county']
        town = batch['town']
        building = batch['building']

        poi = batch['poi']
        trp = batch['trp']
        
        mask = batch['mask'] 

        # [item, price, item, price, ...]
        # seq_len = max_seq_len * 2
        N, S, _ = transaction.shape # torch.Size([1484, 50, 24])

        # embedding
        transaction_embedding = self.embed_transaction(transaction)
        transaction_type_embedding = self.embed_transaction_type(transaction_type)
        province_embedding = self.embed_province(province)
        city_embedding = self.embed_city(city)
        county_embedding = self.embed_county(county)
        town_embedding = self.embed_town(town)
        building_embedding = self.embed_building(building)

        transaction_embedding = (transaction_embedding
                                + transaction_type_embedding
                                + building_embedding)
        """
        transaction_embedding = (
            transaction_embedding
            + province_embedding
            + city_embedding
            + county_embedding 
            + town_embedding)   # torch.Size([1484, 50, 256])
        """        

        price_embedding = self.embed_price(price.unsqueeze(-1))
        time_embedding = self.embed_time(days_passed.unsqueeze(-1))
        transaction_embedding = transaction_embedding + time_embedding 

        # GRU
        x = transaction_embedding + price_embedding
        x = self.in_norm(x)
        if self.mask_input:
            x = x * mask.unsqueeze(-1)
        
        num_layers = 2
        s0 = torch.zeros((num_layers, N, self.hid_dim), device=x.device)
        x, sn = self.gru(x, s0)
        x = self.out_norm(x)

        # resident embedding (for the target transaction, use the features up to t-1)
        B = self.batch_size
        transaction = transaction_embedding[:B, -1] # (B, D)
        
        resident = torch.zeros((N, self.hid_dim), device=x.device)
        resident[:B] = x[:B, -2]
        resident[B:] = x[B:, -1]
        
        poi = self.embed_poi(poi)   
        trp = self.embed_trp(trp)   

        x = {
            'transaction': transaction,
            'res': resident,
            'poi': poi,
            'trp': trp
        }

        return x

