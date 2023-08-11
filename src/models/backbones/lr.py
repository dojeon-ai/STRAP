import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import Time2vecEncoding
from src.models.backbones.base import BaseBackbone
import ipdb
from einops import rearrange    # tensor operations

class LinearRegression(BaseBackbone):
    name='lr'
    def __init__(self,
                 categorical_ids,
                 hid_dim,
                 max_seq_length,
                 use_categorical):
        
        super().__init__(categorical_ids)
        self.max_seq_length = max_seq_length
        self.hid_dim = hid_dim
        self.use_categorical = use_categorical

        # embedding
        cis = self.categorical_ids
        self.embed_transaction = nn.Linear(cis['num_trans_features'], hid_dim)   # (24, 256)
        self.embed_transaction_type = nn.Embedding(len(cis['transaction_type']), hid_dim, padding_idx = 0)  # (5, 256)
        self.embed_province = nn.Embedding(len(cis['province']), hid_dim, padding_idx=0)  # (19, 256)
        self.embed_city = nn.Embedding(len(cis['city']), hid_dim, padding_idx=0)  # (101, 256)
        self.embed_county = nn.Embedding(len(cis['county']), hid_dim, padding_idx=0)    # (251, 256)
        self.embed_town = nn.Embedding(len(cis['town']), hid_dim, padding_idx=0)  # (3571, 256)
        self.embed_building = nn.Embedding(len(cis['building_name']), hid_dim, padding_idx=0)  # (28543, 256)
        self.embed_time = Time2vecEncoding(hid_dim, scale=1000) # time / scale
        self.embed_price = nn.Linear(1, hid_dim) # (1, 256)

        self.embed_poi = nn.Linear(cis['num_poi_features'], hid_dim)   # (16, 256)
        self.embed_trp = nn.Linear(cis['num_trp_features'], hid_dim)   # (3, 256)

        # init
        self.apply(self._init_weights)


    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=(1/np.sqrt(self.hid_dim)))
            nn.init.constant_(m.bias.data, 0.0)


    def get_output_dim(self):

        return self.hid_dim


    def forward(self, batch):
        transaction = batch['transaction']  # torch.Size([1484, 50, 24]): 1484 from neighbor residents (5 per house)
        transaction_type = batch['transaction_type']    # torch.Size([1484, 50])
        province = batch['province']
        city = batch['city']
        county = batch['county']
        town = batch['town']
        building = batch['building']
        
        
        days_passed = batch['days_passed']
        price = batch['price'] 

        poi = batch['poi']
        trp = batch['trp'] 

        # [item, price, item, price, ...]
        # seq_len = max_seq_len * 2
        B, S, _ = transaction.shape # torch.Size([1484, 50, 24])

        # embedding
        transaction_embedding = self.embed_transaction(transaction)
        transaction_type_embedding = self.embed_transaction_type(transaction_type)
        province_embedding = self.embed_province(province)
        city_embedding = self.embed_city(city)
        county_embedding = self.embed_county(county)
        town_embedding = self.embed_town(town)
        building_embedding = self.embed_building(building)

        price_embedding = self.embed_price(price.unsqueeze(-1))
        time_embedding = self.embed_time(days_passed.unsqueeze(-1))

        transaction_embedding = (transaction_embedding
                                + transaction_type_embedding
                                + building_embedding
                                + time_embedding)

        if self.use_categorical:
            transaction_embedding = (
                transaction_embedding
                + province_embedding
                + city_embedding
                + county_embedding 
                + town_embedding)   # torch.Size([1484, 50, 256])
        

        price_embedding = (price_embedding
                          + time_embedding) # torch.Size([1484, 50, 256])
        
        poi = self.embed_poi(poi)   # torch.Size([3493, 256])
        trp = self.embed_trp(trp)   # torch.Size([6224, 256])



        # stack
        x = torch.stack((transaction_embedding, price_embedding), dim = 1)  # torch.Size([1484, 2, 50, 256])
        x = x.permute(0, 2, 1, 3).reshape(B, 2 * S, self.hid_dim)   # torch.Size([1484, 100, 256])

        x = rearrange(x, 'b (s l) h -> b l s h', b=B, s=S, h=self.hid_dim)  # torch.Size([1484, 2, 50, 256])

        state = x[:, 0] # torch.Size([1484, 50, 256])
        price = x[:, 1] # torch.Size([1484, 50, 256])

        x = {
            'state': state,
            'price': price,
            'poi': poi,
            'trp': trp
        }

        return x
