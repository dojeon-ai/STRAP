import gzip
import os
import ipdb
from pathlib import Path
from typing import List, Tuple
import random 
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, Dataset
from .base import BaseLoader
from einops import rearrange
from src.common.data_utils import *
import time


class TransactionResidentDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 device: str,
                 max_seq_length: int,
                 tvt_split: int):

        self.data_path = data_path
        self.device = device
        self.max_seq_length = max_seq_length
        self.tvt_split = tvt_split

        ########################
        # 1. load resident info
        resident_path = Path(data_path + '/resident.csv')
        res_df = pd.read_csv(resident_path, index_col=0).set_index('pnu')

        ############################
        # 2. load transaction info
        transaction_path = Path(data_path + '/transaction.csv')
        trans_df = pd.read_csv(transaction_path, index_col=0, low_memory=False)
        trans_df = trans_df[trans_df['floor']!='UNK']  # delete around (~7 transactions)
        trans_df['floor'] = trans_df['floor'].astype(float)
        trans_df = trans_df.drop(labels=['province','city','county','town', 'latitude','longitude'], axis=1) 

        # building_name is identical for the all transactions in the resident
        # building_name should be part of resident not transaction
        pnu2building_name = trans_df.groupby('pnu')['building_name'].first().values
        pnu2building_name = np.concatenate((np.array([0,0]), pnu2building_name)) # PAD, MASK

        trans_df = trans_df[(trans_df['transaction_type'] == 2)]

        ############################
        # 3. load poi info (poi = amenity)
        poi_path = Path(data_path + '/poi.csv')
        poi_df = pd.read_csv(poi_path, index_col=0, low_memory=False)

        # generate poi-type for training
        # poi_type=2, poi_sub_type=2: elementary school
        # poi_type=3, poi_sub_type=2: local market
        # network will get confusion if poi_sub_type is equal
        # 2 is for [PAD, MASK]
        poi_type = poi_df['poi_type'] - 2
        poi_sub_type = poi_df['poi_sub_type'] - 2
        poi_df['poi_model_type'] = poi_type * poi_sub_type + poi_sub_type + 2

        # convert to one_hot encoding
        def encode_and_bind(df, feature):
            dummies = pd.get_dummies(df[feature])
            df = pd.concat([df, dummies], axis=1)
            return(df)
            
        poi_df = encode_and_bind(poi_df, 'poi_model_type')
        poi_df = poi_df.drop(labels=['province','city','county','town', 
                                     'latitude','longitude',
                                     'poi_type', 'poi_sub_type', 'poi_model_type'], axis=1) 

        ############################
        # 4. load transportation info
        trp_path = Path(data_path + '/transportation.csv')
        trp_df = pd.read_csv(trp_path, index_col=0, low_memory=False)
        
        # convert to one_hot encoding
        trp_df = encode_and_bind(trp_df, 'trp_type')
        trp_df = trp_df.drop(labels=['province','city','county','town', 
                                     'latitude','longitude','trp_type'], axis=1) 

        #############################
        # 5. load categorical ids
        with open(data_path + '/categorical_ids.json', 'r') as f:
            categorical_ids = json.load(f)

        ###########################
        # 6. Standardization
        # categorical column should not be normalized
        # trans_df: transaction type, building_name, days_passed
        # res_df: province, city, county, town
        trans_categorical_cols = ['pnu', 'transaction_type', 'building_name', 'days_passed']
        trans_cols_to_norm = [c for c in trans_df.keys().tolist() if c not in trans_categorical_cols]
        trans_df_mean = trans_df.loc[:,trans_cols_to_norm].mean(axis=0)
        trans_df_std = trans_df.loc[:,trans_cols_to_norm].std(axis=0)
        trans_df.loc[:,trans_cols_to_norm] = \
            (trans_df.loc[:,trans_cols_to_norm] - trans_df_mean) / (trans_df_std + 1e-5)

        res_categorical_cols = ['province','city','county','town']
        res_cols_to_norm = [e for e in res_df.keys().tolist() if e not in res_categorical_cols]
        res_df_mean = res_df.loc[:,res_cols_to_norm].mean(axis=0)
        res_df_std = res_df.loc[:,res_cols_to_norm].std(axis=0)
        res_df.loc[:,res_cols_to_norm] = (res_df.loc[:,res_cols_to_norm] - res_df_mean) / res_df_std

        standardize_dict = {}
        for col in trans_cols_to_norm:
            mean = trans_df_mean.loc[col] 
            std = trans_df_std.loc[col]
            standardize_dict[col] = (mean, std)

        for col in res_cols_to_norm:
            mean = res_df_mean.loc[col] 
            std = res_df_std.loc[col]
            standardize_dict[col] = (mean, std)

        ###############################
        # 7. Sort transaction dataset
        # transaction per pnu:  min:2, max:8547, mean:192.75, >100:16643, >50: 21514
        # {pnu->days_passed}
        # pnu2start_idx : dict (pnu -> start of dataloader idx)
        trans_df = trans_df.sort_values(by=['pnu', 'days_passed']).reset_index(drop=True)

        pnu2startidx = {}
        pnu2endidx = {}
        start_idx_list = trans_df[trans_df.groupby('pnu').agg('cumcount')==0].index.tolist()
        start_idx_list.append(len(trans_df))
        pnu_list = trans_df['pnu'].unique()

        for idx in range(len(pnu_list)):
            pnu = pnu_list[idx]
            start_idx = start_idx_list[idx]
            end_idx = start_idx_list[idx+1] - 1

            pnu2startidx[pnu] = start_idx
            pnu2endidx[pnu] = end_idx

        ############################################
        # 8. Get categorical features for each pnu
        PAD = pd.DataFrame(0, index=[0], columns=res_df.columns)
        MASK = pd.DataFrame(0, index=[1], columns=res_df.columns)
        res_df = pd.concat([PAD, MASK, res_df])

        pnu2categorical = {
            'building': pnu2building_name,
            'province': res_df['province'].values,
            'city': res_df['city'].values,
            'county': res_df['county'].values,
            'town': res_df['town'].values
        }
        trans_df = trans_df.drop(labels=['building_name'], axis=1)
        res_df = res_df.drop(labels=['province', 'city', 'county', 'town'], axis=1)

        ############################################
        # 9. Summing up
        self.trans_df = trans_df
        self.res_df = res_df
        self.poi_df = poi_df
        self.trp_df = trp_df
        
        self.trans_arr = trans_df.values
        self.res_arr = res_df.values
        self.poi_arr = poi_df.values
        self.trp_arr = trp_df.values
        
        self.pnu2startidx = pnu2startidx
        self.pnu2endidx = pnu2endidx
        self.pnu2categorical = pnu2categorical
        
        # 4: transaction_type, days_passed, pnu, price
        num_trans_features = len(trans_df.columns) + len(res_df.columns) - 4
        categorical_ids['num_trans_features'] = num_trans_features
        categorical_ids['num_poi_features'] = len(poi_df.columns)
        categorical_ids['num_trp_features'] = len(trp_df.columns)
        self.categorical_ids = categorical_ids

        self.standardize_dict = standardize_dict

    def __len__(self) -> int:
        return len(self.trans_df)

    def get_tvt_indices(self):
        # split (train, test, val) by the days
        trans_df = self.trans_df
        train_size = int(len(trans_df) * self.tvt_split)
        val_size = (len(trans_df) - train_size) // 2
        test_size = len(trans_df) - train_size - val_size
        print(f'train/val/test: {train_size}:{val_size}:{test_size}')

        trans_df_by_days = trans_df.sort_values(by=['days_passed'])
        train_indices = trans_df_by_days.iloc[0:train_size].index.values
        val_indices = trans_df_by_days.iloc[train_size:train_size+val_size].index.values
        test_indices = trans_df_by_days.iloc[train_size+val_size:].index.values

        return train_indices, val_indices, test_indices  

    def getitem(self, idx):
        """
        # mask
        masks: (S,)

        # numerical input
        # (n_t: no.of transaction cols, n_r: no.of resident cols)
        # (n_c: no.of categorical cols, n_l: no.of label cols)
        transactions: (S, n_t + n_r - n_c - n_l) 
        
        # categorical input
        building_name, province, city, county, town: (S, )

        # time2vec
        days_passed: (S,)

        # target
        prices: (S,)
        """
        pnu = self.trans_df['pnu'][idx]
        pnu_start = self.pnu2startidx[pnu]
        S = self.max_seq_length

        ########################################
        # retrieve transaction history
        num_history = idx - pnu_start + 1  
        if num_history>= S:
            pnu_start = idx - S + 1
            padding_len = 0
            mask = np.ones(S)

        else:
            # pad if number of past transactions < max_seq_length
            pnu_start = pnu_start
            padding_len = S - num_history
            mask = np.array([0]*padding_len + [1]*num_history)
        
        indices = np.arange(pnu_start, idx+1)
        
        ######################################
        # construct transaction features

        # version 1: slow but readable code
        # pandas iloc and loc is slow!!! use numpy for faster indexing
        """
        history = self.trans_df.loc[indices] 
        
        days_passed = history['days_passed'].values
        transaction_type = history['transaction_type'].values
        price = history['price'].values
        history = history.drop(labels=['pnu', 'days_passed', 'transaction_type', 'price'], axis=1)
        
        trans_feat = history.values        
        res_feat = self.res_df.loc[pnu].values
        res_feat = np.tile(res_feat.reshape(1,-1), (min(num_history, S),1))
        trans_feat = np.concatenate((trans_feat, res_feat), 1)
        """

        # version 2: fast but unreadable code
        # numpy indexing is hard to recognize
        history = self.trans_arr[indices]
        days_passed = history[:, 1]
        transaction_type = history[:, 2]
        price = history[:, 3]

        trans_feat = history[:, 4:]
        res_feat = self.res_arr[pnu]
        res_feat = np.tile(res_feat.reshape(1,-1), (min(num_history, S),1))
        trans_feat = np.concatenate((trans_feat, res_feat), 1)

        # padding
        _, t_d = trans_feat.shape
        pad_feat = np.zeros((padding_len, t_d))
        trans_feat = np.concatenate((pad_feat, trans_feat), 0)

        #####################################
        # construct categorical & target features
        pad_feat = np.zeros(padding_len)
        price = np.concatenate((pad_feat, price))
        days_passed = np.concatenate((pad_feat, days_passed))
        transaction_type = np.concatenate((pad_feat, transaction_type))

        building = np.array(self.pnu2categorical['building'][pnu]).repeat(min(num_history, S))
        province = np.array(self.pnu2categorical['province'][pnu]).repeat(min(num_history, S))
        city = np.array(self.pnu2categorical['city'][pnu]).repeat(min(num_history, S))
        county = np.array(self.pnu2categorical['county'][pnu]).repeat(min(num_history, S))
        town= np.array(self.pnu2categorical['town'][pnu]).repeat(min(num_history, S))

        building = np.concatenate((pad_feat, building))
        province = np.concatenate((pad_feat, province))
        city = np.concatenate((pad_feat, city))
        county = np.concatenate((pad_feat, county))
        town = np.concatenate((pad_feat, town))

        #########################################
        # numpy -> torch
        pnu = torch.LongTensor([pnu])
        idx = torch.LongTensor([idx])

        transaction = torch.FloatTensor(trans_feat)
        price = torch.FloatTensor(price)
        days_passed = torch.FloatTensor(days_passed)

        transaction_type = torch.LongTensor(transaction_type)
        building = torch.LongTensor(building)
        province = torch.LongTensor(province)
        city = torch.LongTensor(city)
        county = torch.LongTensor(county)
        town = torch.LongTensor(town)

        mask = torch.BoolTensor(mask)

        data = {'idx': idx,
                'pnu': pnu,
                'transaction': transaction,
                'price': price,
                'days_passed': days_passed,
                'transaction_type': transaction_type,
                'building': building,
                'province': province, 
                'city': city, 
                'county': county, 
                'town': town,
                'mask': mask}
                
        return data        

    def __getitem__(self, idx: int) -> dict():
        return self.getitem(idx)


class SequentialGraphDataLoader(BaseLoader):
    name = 'sequential_graph'
    def __init__(self,
                 data_path: Path,
                 num_workers: int,
                 pin_memory: bool,
                 device: str,
                 batch_size: int,
                 seed: int,
                 max_seq_length: int,
                 tvt_split: int,
                 max_resident_neighbors: int,
                 distance_threshold: float):
        
        super().__init__()
        self.data_path = data_path
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.sampler_rng = random.Random(seed)
        self.tvt_split = tvt_split

        self.max_resident_neighbors = max_resident_neighbors
        self.distance_threshold = distance_threshold
        
        # nodes
        self.dataset = self._get_dataset()   

        # edges
        # for resdient <-> resdient, we use 'pnu' as a key
        # for resdient <-> poi, we use 'pnu' for resdient, idx of df for poi
        # for resdient <-> trp, we use 'pnu' for resdient, idx of df for trp
        resident_edge_path = Path(data_path + '/resident2resident.csv')
        resident_edge_df = pd.read_csv(resident_edge_path, index_col=0)
        resident_edge_df = resident_edge_df[resident_edge_df['distance'] < self.distance_threshold]

        poi_edge_path = Path(data_path + '/resident2poi.csv')
        poi_edge_df = pd.read_csv(poi_edge_path, index_col=0)

        trp_edge_path = Path(data_path + '/resident2transportation.csv')
        trp_edge_df = pd.read_csv(trp_edge_path, index_col=0)

        self.resident_edge_df = resident_edge_df
        self.poi_edge_df = poi_edge_df
        self.trp_edge_df = trp_edge_df

        self.resident_edge_arr = resident_edge_df.values
        self.poi_edge_arr = poi_edge_df.values
        self.trp_edge_arr = trp_edge_df.values
        
        self.res_src2trg = {}
        self.poi_src2trg = {}
        self.trp_src2trg = {}
        
        res_src_pnus = self.resident_edge_arr[:, -2]
        res_trg_pnus = self.resident_edge_arr[:, -1]
        for src, trg in zip(res_src_pnus, res_trg_pnus):
            if src in self.res_src2trg:
                self.res_src2trg[src].append(trg)
            else:
                self.res_src2trg[src] = [trg]
                
        poi_src_pnus = self.poi_edge_arr[:, -1]
        poi_trg_idxs = self.poi_edge_arr[:,  1]
        for src, trg in zip(poi_src_pnus, poi_trg_idxs):
            if src in self.poi_src2trg:
                self.poi_src2trg[src].append(trg)
            else:
                self.poi_src2trg[src] = [trg]
                
        trp_src_pnus = self.trp_edge_arr[:, -1]
        trp_trg_idxs = self.trp_edge_arr[:,  1]
        for src, trg in zip(trp_src_pnus, trp_trg_idxs):
            if src in self.trp_src2trg:
                self.trp_src2trg[src].append(trg)
            else:
                self.trp_src2trg[src] = [trg]
        

    def _get_dataset(self):
        dataset = TransactionResidentDataset(
                    self.data_path, 
                    self.device,
                    self.max_seq_length,
                    tvt_split = self.tvt_split,
                )
        return dataset

    # function used to preprocess and batch the data in a custom way when loading it into a data loader.
    def collate_fn(self, batch):
        """
        standard graph neural network input for PyTorch geometric

        batch = {
            'transaction': 
                node 1, ..., node n: standard input
                node n+1, ..., node N: neighbors for node 1,...,n
            'edge':
                (1, n+1), (1, n+2), ..., (n, N-1), (n, N) 
        }
        """
        trans_df = self.dataset.trans_df
        max_res_edges = self.max_resident_neighbors

        _batch = {}
        for key in batch[0].keys():
            _batch[key] = torch.stack([item[key] for item in batch])

        pnu_batch = _batch['pnu']
        days_passed_batch = _batch['days_passed'][:, -1]

        ###########################
        # 1. get neighbors
        res_neighbor_pnu_batch = []
        poi_neighbor_idx_batch = []
        trp_neighbor_idx_batch = []
        
        for pnu in pnu_batch:
            # version1: slow, readable
            # res_pnus = self.resident_edge_df['trg_pnu'][self.resident_edge_df['src_pnu']==pnu.item()].tolist()
            # poi_idxs = self.poi_edge_df[self.poi_edge_df['src_pnu']==pnu.item()]['trg_idx'].tolist()
            # trp_idxs = self.trp_edge_df[self.trp_edge_df['src_pnu']==pnu.item()]['trg_idx'].tolist()

            # version2: fast, unreadable
            pnu = pnu.item()

            if pnu in self.res_src2trg:
                res_pnus = np.array(self.res_src2trg[pnu])
            else:
                res_pnus = []
                
            if pnu in self.poi_src2trg:
                poi_idxs = np.array(self.poi_src2trg[pnu])
            else:
                poi_idxs = []
                
            if pnu in self.trp_src2trg:
                trp_idxs = np.array(self.trp_src2trg[pnu])
            else:
                trp_idxs = []
                
            if len(res_pnus) >= max_res_edges:
                res_pnus = np.random.choice(res_pnus, max_res_edges, replace=False)
            
            res_neighbor_pnu_batch.append(res_pnus)
            poi_neighbor_idx_batch.append(poi_idxs)
            trp_neighbor_idx_batch.append(trp_idxs)

        ################################
        # 2. get resident neighbors
        node_src_idx = 0
        node_trg_idx = 0
        res_edges = []
        for res_neighbor_pnus, days_passed in zip(res_neighbor_pnu_batch, days_passed_batch):
            for res_neighbor_pnu in res_neighbor_pnus:
                # there exists pnu that does not exist in trans_df
                if res_neighbor_pnu in self.dataset.pnu2startidx:
                    n_pnu_start = self.dataset.pnu2startidx[res_neighbor_pnu]
                    n_pnu_end = self.dataset.pnu2endidx[res_neighbor_pnu]

                    neighbor_days_passed = trans_df['days_passed'][n_pnu_start:n_pnu_end + 1]
                    candidates = (neighbor_days_passed < days_passed.item())
                else:
                    continue

                num_candidates = sum(candidates)
                # there does not exist any neighbor's history prior to days_passed
                if num_candidates == 0:
                    pass

                # get neighbor's history prior to days_passed
                else:
                    neighbor_trans_idx = n_pnu_start + num_candidates -1
                    item = self.dataset.getitem(neighbor_trans_idx)
                    batch.append(item)
                    res_edges.append((node_src_idx, node_trg_idx))
                    node_trg_idx += 1

            node_src_idx += 1
            
        ################################
        # 3. get poi & trp neighbors
        node_idx = 0
        poi_batch, trp_batch = [], []
        poi_edges, trp_edges = [], []

        for poi_neighbor_idxs, trp_neighbor_idxs in zip(poi_neighbor_idx_batch, trp_neighbor_idx_batch):
            for poi_neighbor_idx in poi_neighbor_idxs:
                poi_edges.append((node_idx, len(poi_edges)))

            for trp_neighbor_idx in trp_neighbor_idxs:
                trp_edges.append((node_idx, len(trp_edges)))
            
            node_idx += 1

        poi_neighbor_idx_batch = np.concatenate(poi_neighbor_idx_batch)
        trp_neighbor_idx_batch = np.concatenate(trp_neighbor_idx_batch)

        poi_batch = self.dataset.poi_arr[poi_neighbor_idx_batch.astype(int)]
        trp_batch = self.dataset.trp_arr[trp_neighbor_idx_batch.astype(int)]

        ##############################
        # 4. encode neighbors
        _batch = {}
        for key in batch[0].keys():
            _batch[key] = torch.stack([item[key] for item in batch])
        batch = _batch
        
        if max_res_edges > 0:   # if you use graph neural network
            res_edges = torch.LongTensor(res_edges) 
            batch['res_edge'] = res_edges

        poi_batch = torch.FloatTensor(poi_batch)
        trp_batch = torch.FloatTensor(trp_batch)
        poi_edges = torch.LongTensor(poi_edges)
        trp_edges = torch.LongTensor(trp_edges)
        
        batch['poi'] = poi_batch
        batch['trp'] = trp_batch
        batch['poi_edge'] = poi_edges
        batch['trp_edge'] = trp_edges
            
        return batch

    def get_dataloader(self):     
        train_indices, val_indices, test_indices = self.dataset.get_tvt_indices()
        collate_fn = self.collate_fn

        # use custom sampler to sample from a fixed interval   
        train_sampler = CustomIndexSampler(indices=train_indices,
                                        rng=self.sampler_rng, 
                                        shuffle=True)

        val_sampler = CustomIndexSampler(indices=val_indices,
                                        rng=self.sampler_rng, 
                                        shuffle=False)

        test_sampler = CustomIndexSampler(indices=test_indices,
                                        rng=self.sampler_rng, 
                                        shuffle=False)

        # return dataloader for train / val / test
        train_dataloader = DataLoader(self.dataset, 
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      sampler=train_sampler,
                                      collate_fn=collate_fn,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory,
                                      drop_last=True)

        val_dataloader = DataLoader(self.dataset, 
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    sampler=val_sampler,
                                    collate_fn=collate_fn,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    drop_last=True)

        test_dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     sampler=test_sampler,
                                     collate_fn=collate_fn,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory,                                      
                                     drop_last=True)

        return train_dataloader, val_dataloader, test_dataloader
