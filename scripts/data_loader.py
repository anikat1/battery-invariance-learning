import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

from data import InvarianceDataset
from regressor_data import MaskedBatteryArchive, MaskedORNL
from data_shuffle import select_shuffle_algorithm, read_thermal_runoff

def custom_train_loader(args, list_mat, list_soc, dict_map_idx, algorithm, batch_size):
    list_pair = select_shuffle_algorithm(args, list_soc, list_mat, 
                                         dict_map_idx, 
                                         algorithm=algorithm)
    print('S:', len(list_pair), algorithm) 
    #print(dict_map_idx)
    if algorithm =="all":
        dataset = MaskedBatteryArchive(args, list_pair, args.mat_list, 
                                       eval = f'train_{algorithm}/', 
                                       load = False)
    else:
        dataset = InvarianceDataset(args, list_pair, args.mat_list, 
                                eval= f'train_inv_{algorithm}/', 
                                inv_type= algorithm, 
                                load=False)
    dataset_size = len(dataset)
    train_indices = list(range(dataset_size))
    
    #split = int(np.floor(args.test_split * dataset_size))
    '''
    if args.random_seed>=0:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    '''
    #train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    #test_sampler = SubsetRandomSampler(test_indices)
    #train_sampler = SequentialSampler(train_indices)
    #test_sampler = SequentialSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              num_workers=0, sampler=train_sampler, shuffle = False, drop_last=False)
    
    #test_loader = DataLoader(dataset, batch_size=batch_size,
     #                   num_workers =0, sampler=test_sampler, shuffle=False)

    print('loader:', algorithm, len(train_loader))

    return train_loader, list_pair

def custom_test_loader(args, list_mat, list_soc, dict_map_idx, algorithm, batch_size):
    list_pair = select_shuffle_algorithm(args, list_soc, list_mat, 
                                         dict_map_idx, 
                                         algorithm=algorithm)
    print('S:', len(list_pair)) 
    #print(dict_map_idx)
    if algorithm =="all":
        dataset = MaskedBatteryArchive(args, list_pair, list_mat, 
                                       eval = f'test_{algorithm}/', 
                                       load = False)
    elif algorithm =="ornl":
        list_files = read_thermal_runoff(args)
        dataset = MaskedORNL(args, list_files, list_mat)

    else:
        dataset = InvarianceDataset(args, list_pair, args.mat_list, 
                                eval= f'test_inv_{algorithm}/', 
                                inv_type= algorithm, 
                                load=False)
    
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=None, shuffle = False)
    
    print('loader:', algorithm, len(loader))

    return loader, list_pair
