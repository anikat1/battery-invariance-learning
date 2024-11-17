import os
import pdb
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from data_shuffle import init_preprocess

class InvarianceDataset(object):
    def __init__(self, args, list_pair, list_material, eval, inv_type, load=False):
        super().__init__()
        self.dir_data = args.dir_data
        
        self.pair_list = list_pair
        self.insample_size = args.history_size # input sequence length
        self.time_lag = args.time_lag #time lag
        self.horizon = args.out_seq_len #forecast sequence length
        self.ts_len = args.history_size+args.time_lag+args.out_seq_len

        self.mat_one_hot = F.one_hot(torch.arange(0, args.num_class), num_classes = args.num_class)
        ts_names = ['Penetrator Force [mm]', 'vCell [V]', 'TC5 above punch [C]', 'TC6 below punch [C]']
        self.MAX_LENGTH = args.max_len
        self.PAD_VALUE = args.pad_value
        self.jump = args.time_lag # this can be any value, want to set up input seq as disjoint from next data in batch
        num_in_feat = args.input_size # num features to be masked (load, voltage)
        self.data_path = eval #where to save the preprocessed data train/test
        self.inv_type = inv_type
        
        self.input_seq = []
        self.target_seq = []
        self.mask_in = []
        self.invariance=[] #feature name that is different between the pairs, e.g., material name/soc value
        self.all_enc = []
        self.pos_id = [] #start and end timestamp of input to forecast
        
        '''
        force_seg number of segments for cumulative force, we take the 
        chunk length=time_lag to sync with the input-output sequence
        '''
        self.seg_len = int(self.MAX_LENGTH/self.time_lag)
        
        if load:
            with open(self.dir_data+self.data_path+"X", "rb") as fp:   
                self.input_seq = pickle.load(fp)
            with open(self.dir_data+self.data_path+"Y", "rb") as fp: 
                self.target_seq = pickle.load(fp)
            with open(self.dir_data+self.data_path+"mask_seq", "rb") as fp:
                self.mask_in = pickle.load(fp)
            with open(self.dir_data+self.data_path+"metadata", "rb") as fp: 
                self.all_enc = pickle.load(fp)
            with open(self.dir_data+self.data_path+"invariance", "rb") as fp: 
                self.invariance = pickle.load(fp)
            with open(self.dir_data+self.data_path+"pos_idx", "rb") as fp: 
                self.pos_id = pickle.load(fp)
        else:   
            for i in range(len(list_pair)):
                s1 = list_pair[i][0] # file name of the 1st domain type
                s2 = list_pair[i][1] #file name of the 2nd domain type
                print(i, s1, s2)
                s1_ts, force_masks1, vol_masks1, meta_info1 = init_preprocess(args, self.seg_len, list_material, ts_names, s1, inv_type)
                forecast_s1idx = s1_ts.shape[0]-self.horizon
                cp1 = np.arange(self.insample_size, forecast_s1idx-self.time_lag, self.jump, 
                                    dtype= int)
                
                s2_ts, force_masks2, vol_masks2, meta_info2 = init_preprocess(args, self.seg_len, list_material, ts_names, s2, inv_type)
                forecast_s2idx = s2_ts.shape[0]-self.horizon
                cp2 = np.arange(self.insample_size, forecast_s2idx-self.time_lag, self.jump, dtype= int)
                chunks = cp1 if len(cp1)<len(cp2) else cp2
                #print('S1, S2 total temp:', s1, s2, len(cp1), len(cp2), len(chunks))
                #print(np.sum(force_masks1), np.sum(vol_masks1), np.sum(force_masks2), np.sum(vol_masks2))
                #print('chunks:',i, len(chunks)) 
                for idx in chunks:
                    #(here PAD_VALUE = 0 , MAX_LENGTH = length of longest seq)
                    # every input is two times
                    masks = np.ones((self.seg_len, num_in_feat*2))*self.PAD_VALUE
                    tmp_idx = int(idx/self.time_lag) #index until current time to store cum_force and vol_gradient 
                    start_id = idx-self.insample_size
                    
                    masks[:tmp_idx, 0] = force_masks1[:tmp_idx]
                    masks[:tmp_idx, 1] = vol_masks1[:tmp_idx]
                    masks[:tmp_idx, 2] = force_masks2[:tmp_idx]
                    masks[:tmp_idx, 3] = vol_masks2[:tmp_idx]
                    
                    sid_y = idx+self.time_lag
                    eid = sid_y+self.horizon
                    src = torch.stack((torch.tensor(s1_ts[start_id:idx,:]), torch.tensor(s2_ts[start_id:idx,:])), dim =0)
                    trg = torch.stack((torch.tensor(s1_ts[sid_y:eid, 2]), torch.tensor(s2_ts[sid_y:eid, 2])), dim =0)
                    
                    self.target_seq.append(trg)
                    self.input_seq.append(src)
                    self.mask_in.append(torch.FloatTensor(masks))
                    temp_pos = float(idx/len(chunks))
                    meta_info1[2] = temp_pos
                    meta_info2[2] = temp_pos
                    enc = torch.stack((torch.FloatTensor(meta_info1[:4]),torch.FloatTensor(meta_info2[:4])), dim =0)
                    self.all_enc.append(enc)
                    self.invariance.append((meta_info1[4], meta_info2[4])) 
                    self.pos_id.append((sid_y,eid))
                    #self.end_id.append(eid)
                
            '''
            if not os.path.exists(self.dir_data+self.data_path): 
                os.mkdir(self.dir_data+self.data_path)
            with open(self.dir_data+self.data_path+"X", 'wb') as fp:
                pickle.dump(self.input_seq, fp)
            with open(self.dir_data+self.data_path+"Y", 'wb') as fp:
                pickle.dump(self.target_seq, fp)
            with open(self.dir_data+self.data_path+"mask_seq", "wb") as fp:   # Unpickling
                pickle.dump(self.mask_in, fp)
            with open(self.dir_data+self.data_path+"metadata", 'wb') as fp:
                pickle.dump(self.all_enc, fp)
            with open(self.dir_data+self.data_path+"invariance", 'wb') as fp:
                pickle.dump(self.invariance, fp)
            
            with open(self.dir_data+self.data_path+"pos_idx", "wb") as fp:   # Unpickling
                pickle.dump(self.pos_id, fp)
            with open(self.dir_data+self.data_path+"list_pair", "wb") as fp:   # Unpickling
                pickle.dump(self.pair_list, fp)
            '''
        print('custom data preprocess finished..')


             
    def __len__(self): 
        return len(self.input_seq)
        
    def __getitem__(self,i):
        enc = self.all_enc[i]
        mat_enc = torch.stack((self.mat_one_hot[int(enc[0,3].item())], self.mat_one_hot[int(enc[1,3].item())]), dim = 0)
        #print('invariance name:',self.invariance[i])
        #print('src, trg, mask:', self.input_seq[i].shape, self.target_seq[i].shape, self.mask_in[i].shape)
        #print('mat_enc, meta_enc:', mat_enc.size(), self.all_enc[i].size())
        #print('start, end:', self.pos_id[i])
        return {
            'src': self.input_seq[i],#torch.tensor(src), 
            'trg': self.target_seq[i], 
            'start_id': self.pos_id[i][0], 
            'end_id': self.pos_id[i][1],
            'meta_dat': self.all_enc[i][:3],
            'mat_enc': mat_enc,
            'mask_seq': self.mask_in[i],
            'inv1': self.invariance[i][0],
            'inv2': self.invariance[i][1],            
            #'ts_len': ts.shape[0]
            }

'''
#debug data loader
for idx, data in enumerate(train_loader3):
    #inv = data['inv1']
    "access: s1=data['src'][0], s2 = data['src][1]"
    print(idx, len(data['battery_id']))
    print('src, trg, mask:', data['src'].size(), data['trg_y'].size(), data['mask_seq'].size())
    print('mat_enc, meta_enc:', data['mat_enc'].size(), data['meta_dat'].size())
    print('start, end:', data['start_id'][:3], data['end_id'][:3])
    print('battery:', data['battery_id'][:3])
    #print('start, end:', data['start_id'], data['end_id'], data['inv1'], data['inv2'])
    if idx>10:
        break
        
'''
       
       



            
        
            




        
        