import torch
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

#from data import InvarianceDataset
#from regressor_data import MaskedBatteryArchive
from data_shuffle import generate_forecast_data
from data_loader import custom_train_loader, custom_test_loader
from parse import parse_args
from model import DOML

import os
import numpy as np
import pandas as pd

from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import copy

dict_map_idx={}

args, hparams=parse_args()
torch.manual_seed(args.random_seed)
if not os.path.exists(args.dir_out):
    os.mkdir(args.dir_out)
#data object setup

#model setup 
num_classes =[len(args.list_mat_inv), len(args.list_soc_inv)]
num_domains = [0, 0]
for n in range(len(num_classes)):
    num_domains[n] = int((num_classes[n]*(num_classes[n]-1))/2)

num_masks = int(args.max_len/args.time_lag)
n_inputs_soc = [int(args.history_size*args.input_size), args.num_class, args.all_enc, args.input_size]
n_outputs_soc = [int(args.out_seq_len*args.output_size), 1, 1, num_masks, args.out_seq_len]
type_inv =["material", "soc", "reg"]
print('class, domain', num_classes, num_domains)
hparams['num_classes'] = num_classes
hparams['num_domains'] = num_domains
hparams['n_inputs'] = n_inputs_soc
hparams['n_outputs'] = n_outputs_soc

hparams_mat = copy.deepcopy(hparams)
hparams_mat['n_inputs'] = [int(args.history_size*(args.input_size-1)), args.num_class, args.all_enc, args.input_size-1] 
 
torch.cuda.empty_cache()
#device="cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(log_dir='runs/'+args.dir_tbs)
if not args.test_only:
    
    train_loader1, list_pair_mat = custom_train_loader(args, args.list_mat_inv, args.list_soc_inv, dict_map_idx,
                                                 algorithm = "material",  batch_size = args.batch_size_inv)                                                                                     

    train_loader2, list_pair_soc = custom_train_loader(args, args.list_mat_inv, args.list_soc_inv, dict_map_idx,                                      
                                                algorithm = "soc", batch_size = args.batch_size_inv)

    train_loader3, list_all = custom_train_loader(args, args.list_mat_inv, args.list_soc_inv, dict_map_idx,
                                              algorithm = "all", batch_size = args.batch_size)

    print('domain:', dict_map_idx)
    
    model = DOML(hparams, hparams_mat, device, writer)
    
    for n_t in range(args.epochs):
        save = False
        if (n_t+1)%args.save_freq==0:
            save= True
        model.train_reg(args, train_loader1, train_loader2, train_loader3, dict_map_idx, type_inv, 
                  args.list_mat_inv, args.list_soc_inv, save, n_t, update_inv = False)
    
else:
    test_loader_mat, list_all = custom_test_loader(args, args.list_mat_inv, 
                args.list_soc_inv, dict_map_idx,
                algorithm = args.algorithm, batch_size = args.batch_size)
    print('test:', len(test_loader_mat))
    
    model = DOML(hparams, hparams_mat, device, writer, eval=True)
    
    #test regressor
    test_output, start_t, end_t, list_did, rmse, mape = model.test_regressor(args.epochs, test_loader_mat, args.dir_model, args.dir_mat_model, args.dir_soc_model)
    generate_forecast_data(args, test_output, start_t, end_t, list_did, 'test') 
    
writer.flush()
writer.close()

