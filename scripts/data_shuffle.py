import numpy as np
import torch
import os
import glob

import numpy as np
import pandas as pd
import itertools
from itertools import cycle
from parse import parse_args

def init_preprocess(args, seg_len, list_material, ts_names, s, inv):
    #s is the file name to preprocess
    s_info = s.split('_')
    s_material = s_info[1]
    s_cap = float(s_info[3][:-2])/100
    s_soc = float(s_info[4][:-3])/100
    s_material_class = list_material.index(s_material)

    data = pd.read_excel(args.dir_data+s, usecols=ts_names, engine="openpyxl")
    data['TC6 below punch [C]'] = data['TC6 below punch [C]'].fillna(0)
    data['Temperature'] = data['TC5 above punch [C]'] + data['TC6 below punch [C]']
    data['Temperature'] = data['Temperature']/2.0
    #select only the columns to use for input/output
    data=data.loc[:, ['Penetrator Force [mm]','vCell [V]','Temperature']] 
    mts = data.dropna().values
    mts_norm = (mts-mts.min(axis=0))/(mts.max(axis=0)-mts.min(axis=0))  

    seg_cp = np.arange(args.time_lag, args.max_len, args.time_lag, dtype = int)
    force_segs = np.zeros(seg_len)
    vol_segs = np.zeros(seg_len)
    for idx in range(len(seg_cp)):
        sg = seg_cp[idx]
        force_segs[idx] = np.sum(mts_norm[sg-args.time_lag:sg,0])
        #vol_segs[idx] = np.sum(np.diff(mts_norm[sg-self.time_lag:sg,1]))
        vol_segs[idx] = np.sum(mts_norm[sg-args.time_lag:sg,1])
    #print('force, vol:', np.sum(force_segs), np.sum(vol_segs))
    force_segs = force_segs/force_segs.max()
    vol_segs = vol_segs/vol_segs.max()   
    if inv=="material":
        invar = s_material 
    elif inv == "soc":
        invar = s_info[4][:-3] #soc 
    else:
        invar = s #for lstm keeping the file name
    meta = [s_cap, s_soc, 0, s_material_class, invar] # add 3rd element 0 to replace with temp encoding
    
    return mts_norm, force_segs, vol_segs, meta 

def read_battery_file_list(args, mat, soc, type="full"):
    dir_list = [f for f in glob.glob(glob.escape(args.dir_data)+f"/SNL_{mat}_*.xlsx")]
    data_list = []
    for file in dir_list:
        fid = os.path.basename(file)
        file_info = fid.split('_')
        fsoc = int(file_info[4][:-3])
        if fsoc==soc:
            data_list.append(fid)
    
    #print('num_files:',len(data_list)) 
    return data_list

def select_shuffle_algorithm(args, list_soc, list_mat, dic_map_idx, algorithm):
    #train ids is a list of properties to select train adversarials
    #dataset are the list of file names
    if algorithm=="material":
        return data_shuffle_material(args, list_soc, list_mat, dic_map_idx)
    elif algorithm=="soc":
        return data_shuffle_soc(args, list_soc, list_mat, dic_map_idx) 
    else: 
        return data_shuffle_reg(args, list_soc, list_mat) 
    
def data_shuffle_reg(args, list_soc, list_mat):
    S = []
    for i in range(len(list_mat)):
        for k in list_soc:
            l1 = read_battery_file_list(args, list_mat[i], k)
            S+=l1
    
    #print('S', len(S))
    #print(*S, sep = "\n")

    return S

def data_shuffle_material(args, list_soc, list_mat, dic_map_idx):
    #for this shuffle between two lists, only material will change, soc and temp encoding remain same
    S = [] #will add pair of list of files for invariance
    ttl = len(list(dic_map_idx.keys()))
    for i in range(len(list_mat)-1):
        for j in range(i+1, len(list_mat)):
            key = f"{list_mat[i]}-{list_mat[j]}"
            dic_map_idx[key] = ttl
            for k in list_soc:
                #collect all the files with given material and soc
                l1 = read_battery_file_list(args, list_mat[i], k)
                l2 = read_battery_file_list(args, list_mat[j], k) 
                #print(list_mat[i], list_mat[j], k, len(l1), len(l2))
                l1.sort()
                l2.sort()
                comb = list(zip(l1, cycle(l2))
                        if len(l1) > len(l2)
                        else zip(cycle(l1), l2))
   
                S+=comb
            ttl+=1
    
    #print('S', len(S))
    #print(*S, sep = "\n")

    return S


def data_shuffle_soc(args, list_soc, list_mat, dic_map_idx):
    #for this shuffle between two lists, only soc will change, material and temp encoding remain same
    S = [] #will add list of files for 1st of pair for invariance
    ttl= len(list(dic_map_idx.keys()))
    for i in range(len(list_soc)-1):
        for j in range(i+1, len(list_soc)):
            key = f"{str(list_soc[i])}-{str(list_soc[j])}"
            dic_map_idx[key] = ttl
            for k in list_mat:
                l1 = read_battery_file_list(args, k, list_soc[i])
                l2 = read_battery_file_list(args, k, list_soc[j]) 
                #print(list_soc[i], list_soc[j], k, len(l1), len(l2))
                l1.sort()
                l2.sort()
                comb = list(zip(l1, cycle(l2))
                        if len(l1) > len(l2)
                        else zip(cycle(l1), l2))
   
                S+=comb
            ttl+=1
    
    #print('S', len(S))
    #print(*S, sep = "\n")

    return S

def post_process_input(args,sids,data_ids,src,predicted_nowcast, count_occ=None):
    processed_src=src.numpy()
    
    for batch_id in range(len(data_ids)):
        if data_ids[batch_id] in predicted_nowcast.keys(): 
            st = sids[batch_id]-args.time_lag-args.history_size
            ed = st+args.history_size
            hs=0
            for ix in range(st, ed):
                if ix in predicted_nowcast[data_ids[batch_id]].keys():
                    processed_src[batch_id,hs,2]=predicted_nowcast[data_ids[batch_id]][ix]
                
                hs+=1
    
    return torch.tensor(processed_src)

def read_thermal_runoff(args):
    file_list = open(args.dir_list+'battery_list2.txt','r')
    Lines = file_list.readlines()
    num = 0
    data_list =[]
    for line in Lines:
        data_list.append(line.strip())
        num+=1
    
    return data_list

def post_process_results(sids, eids, data_ids, model_forecast, predicted_nowcast):
   
    for batch_id in range(len(data_ids)):
        if data_ids[batch_id] not in predicted_nowcast.keys():
            predicted_nowcast[data_ids[batch_id]]={}
            #count_occ[data_ids[batch_id]]={}
        
        st = sids[batch_id]
        ed = eids[batch_id]
        index=0
        for ix in range(st, ed):
            if ix not in predicted_nowcast[data_ids[batch_id]]:
                predicted_nowcast[data_ids[batch_id]][ix]=model_forecast[batch_id,index]
                #count_occ[data_ids[batch_id]][ix]=1
            else:
                predicted_nowcast[data_ids[batch_id]][ix]=predicted_nowcast[data_ids[batch_id]][ix]+model_forecast[batch_id,index]
                #count_occ[data_ids[batch_id]][ix]=count_occ[data_ids[batch_id]][ix]+1
            
            index+=1

def generate_forecast_data(args, forecasts, start_t, end_t, list_did, type):
    import pandas as pd
    forecasts_df = pd.DataFrame(forecasts, columns=[f'V{i+1}' for i in range(args.out_seq_len)])
    forecasts_df.index.name = 'id'
    forecasts_df['start_t']=start_t
    forecasts_df['end_t']=end_t
    forecasts_df['data_id']=list_did
    forecasts_df.to_csv(args.dir_out+args.test_file+f'_{type}.csv')     
    



