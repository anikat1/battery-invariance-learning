import torch 
import torch.nn.functional as F
import pandas as pd 
import numpy as np
import os
import pickle
import pdb
from data_shuffle import init_preprocess

class MaskedORNL:
    def __init__(self, args, list_files, list_material, load = False): 
        super().__init__()
        self.dir_data = args.dir_data
        self.battery_list=list_files
        self.insample_size = args.history_size
        self.time_lag = args.time_lag
        self.horizon = args.out_seq_len
        self.ts_len = args.history_size+args.time_lag+args.out_seq_len 

        self.input_seq = []
        self.target_seq = []
        self.mask_in = [] #will have cumulative load, voltage masked seq
        self.all_enc = [] #will contain soc, cap, temp_enc, material class 
        self.pos_id = [] #pos id will contain start, end seq, battery file id of list_files
        
        self.mat_one_hot = F.one_hot(torch.arange(0, args.num_class), num_classes = args.num_class)
        self.MAX_LENGTH = args.max_len
        self.PAD_VALUE = args.pad_value
        self.jump = self.time_lag # this can be any value, want to set up input seq as disjoint from next data in batch
        num_feat = args.input_size # num features to be masked (load, voltage)
        self.seg_len = int(self.MAX_LENGTH/self.time_lag)
        num_ts = 0
        ts_names = ['2000 Pounds [Pounds]', 'Voltage [V]', 'Temperature [C]'] 
        data = pd.read_csv(self.dir_data+args.in_file, header = [0,1,2], index_col=0, 
                       low_memory=False)
        
        for fid in list_files:
            file_id, label = fid.split('_') 
            tlabel = int(float(label) *100)
            label = str(tlabel)+' SOC'
            soc = float(label.split(' ')[0])
            cap = float(file_id.split(' ')[0])
            cap = cap/1000
            print(fid, soc, cap)
            material_class = list_material.index('ORNL')
            df_mts = data[[(file_id, label, ts_names[0]), (file_id, label, ts_names[1]), 
                       (file_id, label, ts_names[2])]]
            mts = df_mts.dropna().values
            mts_norm = (mts-mts.min(axis=0))/(mts.max(axis=0)-mts.min(axis=0))      
            seg_cp = np.arange(args.time_lag, args.max_len, args.time_lag, dtype = int)
            force_segs = np.zeros(self.seg_len)
            vol_segs = np.zeros(self.seg_len)
            for idx in range(len(seg_cp)):
                sg = seg_cp[idx]
                force_segs[idx] = np.sum(mts_norm[sg-args.time_lag:sg,0])
                #vol_segs[idx] = np.sum(np.diff(mts_norm[sg-self.time_lag:sg,1]))
                vol_segs[idx] = np.sum(mts_norm[sg-args.time_lag:sg,1])
            
            force_segs = force_segs/force_segs.max()
            vol_segs = vol_segs/vol_segs.max()  
            forecast_sidx = mts_norm.shape[0]-self.horizon
            cp = np.arange(self.insample_size, forecast_sidx-self.time_lag, 
                               self.jump, dtype= int)
            total_cp = len(cp) 
            for idx in cp:
                #(here PAD_VALUE = 0 , MAX_LENGTH = length of longest seq)
                masks = np.ones((self.seg_len, num_feat))*self.PAD_VALUE
                tmp_idx = int(idx/self.time_lag) #index until current time to store cum_force and vol_gradient 
                start_id = idx-self.insample_size
                
                masks[:tmp_idx, 0] = force_segs[:tmp_idx]
                masks[:tmp_idx, 1] = vol_segs[:tmp_idx]
                sid_y = idx+self.time_lag
                eid = sid_y+self.horizon
                src = mts_norm[start_id:idx,:]
                trg = mts_norm[sid_y:eid, 2]
                self.target_seq.append(trg)
                self.input_seq.append(src)
                self.mask_in.append(torch.FloatTensor(masks))
                temp_pos = float(idx/total_cp)
                meta_info = [cap, soc, temp_pos, material_class, fid] 
                enc = torch.FloatTensor(meta_info[:4])
                self.all_enc.append(enc)
                self.pos_id.append((sid_y, eid, meta_info[4]))
                
                num_ts+=1 
        
        print('total ts:', num_ts, len(self.input_seq))     
    
    def __len__(self): 
        return len(self.input_seq)
    
    def __getitem__(self,i):
        enc = self.all_enc[i]
        mat_enc = self.mat_one_hot[int(enc[3].item())]
        
        return {
            'src': self.input_seq[i],#torch.tensor(src), 
            'trg_y': self.target_seq[i], 
            'start_id': self.pos_id[i][0], 
            'end_id': self.pos_id[i][1],
            'battery_id': self.pos_id[i][2],
            'meta_dat': self.all_enc[i][:3],
            'mat_enc': mat_enc,
            'mask_seq': self.mask_in[i]
            }  
        
class MaskedBatteryArchive:
    def __init__(self, args, list_files, list_material, eval, load = False):
        super().__init__()
        self.dir_data = args.dir_data
        
        self.battery_list=list_files
        self.insample_size = args.history_size
        self.time_lag = args.time_lag
        self.horizon = args.out_seq_len
        self.ts_len = args.history_size+args.time_lag+args.out_seq_len
        #self.cutpoints=[]
        #self.ts_index=[]
        #self.timeseries=[]
        self.input_seq = []
        self.target_seq = []
        self.mask_in = [] #will have cumulative load, voltage masked seq
        self.all_enc = [] #will contain soc, cap, temp_enc, material class 
        self.pos_id = [] #pos id will contain start, end seq, battery file id of list_files
        
        self.mat_one_hot = F.one_hot(torch.arange(0, args.num_class), num_classes = args.num_class)
        ts_names = ['Penetrator Force [mm]', 'vCell [V]', 'TC5 above punch [C]', 'TC6 below punch [C]']
        self.MAX_LENGTH = args.max_len
        self.PAD_VALUE = args.pad_value
        self.jump = self.time_lag # this can be any value, want to set up input seq as disjoint from next data in batch
        num_ts = 0
        num_feat = args.input_size # num features to be masked (load, voltage)
        '''
        force_seg number of segments for cumulative force, we take the 
        chunk length=time_lag to sync with the input-output sequence
        '''
        self.seg_len = int(self.MAX_LENGTH/self.time_lag)
        data_path = eval
        
        if load:
            with open(self.dir_data+data_path+"X", "rb") as fp:   
                self.input_seq = pickle.load(fp)
            with open(self.dir_data+data_path+"Y", "rb") as fp:   # Unpickling
                self.target_seq = pickle.load(fp)
            with open(self.dir_data+data_path+"mask_seq", "rb") as fp:   # Unpickling
                self.mask_in = pickle.load(fp)
            with open(self.dir_data+data_path+"metadata", "rb") as fp: 
                self.all_enc = pickle.load(fp)
            with open(self.dir_data+data_path+"pos_idx", "rb") as fp:   # Unpickling
                self.pos_id = pickle.load(fp)
            #with open(self.dir_data+data_path+"end_idx", "rb") as fp:   # Unpickling
            #    self.end_id = pickle.load(fp)
            #with open(self.dir_data+data_path+"battery_idx", "rb") as fp:   # Unpickling
            #    self.battery_id = pickle.load(fp)
        else:    
            for fid in self.battery_list: 
                #print(fid)
                s_ts, force_masks, vol_masks, meta_info = init_preprocess(args, self.seg_len, list_material, ts_names, fid, None)
                
                forecast_sidx = s_ts.shape[0]-self.horizon
                cp = np.arange(self.insample_size, forecast_sidx-self.time_lag, 
                               self.jump, dtype= int)
                total_cp = len(cp) 
                for idx in cp:
                    #(here PAD_VALUE = 0 , MAX_LENGTH = length of longest seq)
                    masks = np.ones((self.seg_len, num_feat))*self.PAD_VALUE
                    tmp_idx = int(idx/self.time_lag) #index until current time to store cum_force and vol_gradient 
                    start_id = idx-self.insample_size
                    
                    masks[:tmp_idx, 0] = force_masks[:tmp_idx]
                    masks[:tmp_idx, 1] = vol_masks[:tmp_idx]
                    sid_y = idx+self.time_lag
                    eid = sid_y+self.horizon
                    src = s_ts[start_id:idx,:]
                    trg = s_ts[sid_y:eid, 2]
                    self.target_seq.append(trg)
                    self.input_seq.append(src)
                    self.mask_in.append(torch.FloatTensor(masks))
                    temp_pos = float(idx/total_cp)
                    meta_info[2] = temp_pos
                    enc = torch.FloatTensor(meta_info[:4])
                    self.all_enc.append(enc)
                    self.pos_id.append((sid_y,eid,meta_info[4]))
                    
                    num_ts+=1
                #break
            '''
            if not os.path.exists(self.dir_data+data_path):     
                os.mkdir(self.dir_data+data_path)
            with open(self.dir_data+data_path+"X", 'wb') as fp:
                pickle.dump(self.input_seq, fp)
            with open(self.dir_data+data_path+"Y", 'wb') as fp:
                pickle.dump(self.target_seq, fp)
            with open(self.dir_data+data_path+"mask_seq", "wb") as fp:   # Unpickling
                pickle.dump(self.mask_in, fp)
            with open(self.dir_data+data_path+"metadata", "wb") as fp: 
                pickle.dump(self.all_enc, fp)
            with open(self.dir_data+data_path+"pos_idx", "wb") as fp:   # Unpickling
                pickle.dump(self.pos_id, fp)
            '''    
        print('total ts:', num_ts, len(self.input_seq))  
        #pdb.set_trace()
            
    def __len__(self): 
        return len(self.input_seq)
    
    def __getitem__(self,i):
        enc = self.all_enc[i]
        mat_enc = self.mat_one_hot[int(enc[3].item())]
        return {
            'src': self.input_seq[i],#torch.tensor(src), 
            'trg_y': self.target_seq[i], 
            'start_id': self.pos_id[i][0], 
            'end_id': self.pos_id[i][1],
            'battery_id': self.pos_id[i][2],
            'meta_dat': self.all_enc[i][:3],
            'mat_enc': mat_enc,
            'mask_seq': self.mask_in[i]
            #'ts_len': ts.shape[0]
            }
