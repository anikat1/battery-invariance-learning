import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn
from pytorch_forecasting.metrics import MAPE, RMSE
    
import numpy as np
import networks
import copy
import pdb
import sys
from utils import *
from data_shuffle import post_process_results
import pickle

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#scipt modifierd from the adapted code MLIR and META domainbed

class DOML(object):
    def __init__(self, hparams, hparams_mat, device, writer=None, eval = False):
        super(DOML, self).__init__()

        #Algorithms
        self.num_domains = hparams['num_domains'] #total num domains
        self.num_classes = hparams['num_classes'] #total num domains
        self.n_outputs = hparams['n_outputs']
        self.hparams = hparams
        self.hparams_mat = hparams_mat
        self.device = device
        self.total_domains = np.sum(self.num_domains)
        self.mat_featurizer = networks.InvariantModel(self.hparams) #f_psi^1
        self.soc_featurizer = networks.InvariantModel(self.hparams) #f_psi^2
        self.regressor = networks.InvariantLSTM( hparams, self.n_outputs[-1]) #g_theta: LSTM regression 
        #self.regressor = networks.InvariantTransformer( hparams, self.n_outputs[-1]) #g_theta: LSTM regression 
        self.writer =writer #tensorboard plot
        
        advclassifier_mat = networks.AdvClassifier( #{g_theta_ij}
            self.mat_featurizer.out_shape,
            self.num_classes[0], #need adversarial for material classes, i.e., 3 [LCO, LFP, NMC]
            False #set to true if training does not do well for 1 linear layer
        )
        advclassifier_soc = networks.AdvClassifier( #{g_theta_ij}
            self.soc_featurizer.out_shape,
            self.num_classes[1], #need adversarial for soc, i.e. 6, [0, 50, 70, 90]
            False #set to true if training does not do well for 1 linear layer
        )

        #optimizer for material invariance
        self.gen_mat_opt = torch.optim.Adam(
            self.mat_featurizer.parameters(),
            lr=self.hparams["lr_g"]*self.hparams["advfactor_gamma"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9)
        )
        #optimizer for soc invariance
        self.gen_soc_opt = torch.optim.Adam(
            self.soc_featurizer.parameters(),
            lr=self.hparams["lr_g"]*self.hparams["advfactor_gamma"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9)
        )
        
        self.advclassifiers = nn.ModuleList([copy.deepcopy(advclassifier_mat) if i<self.num_domains[0] 
                                                    else copy.deepcopy(advclassifier_soc) for i in range(self.total_domains)])
        
        #Optimizers 
        self.reg_opt = torch.optim.Adam(
            self.regressor.parameters(),
            lr=self.hparams["lr_l"],
            weight_decay=self.hparams['weight_decay_l'],
            # betas=(self.hparams['beta1'], 0.9)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        
        if not eval:
            self.mat_featurizer = torch.nn.DataParallel(self.mat_featurizer, dim =0)  
            self.soc_featurizer = torch.nn.DataParallel(self.soc_featurizer, dim =0)  
            self.regressor = torch.nn.DataParallel(self.regressor, dim =0) 
            #num_domains[0]: number of pairs of different material 
            #num_domains[1]: number of pairs of different soc, len(dict_keys)
            self.advclassifiers = nn.ModuleList([torch.nn.DataParallel(copy.deepcopy(advclassifier_mat), dim =0) if i<self.num_domains[0] 
                                                    else torch.nn.DataParallel(copy.deepcopy(advclassifier_soc), dim =0) for i in range(self.total_domains)])
            
            self.mat_featurizer = self.mat_featurizer.to(self.device)
            self.soc_featurizer = self.soc_featurizer.to(self.device)
            self.regressor = self.regressor.to(self.device)
            self.advclassifiers = self.advclassifiers.to(self.device)
            
            self.criterion = self.criterion.to(self.device)
            self.reg_criterion = self.reg_criterion.to(self.device)
        #else:
        #    self.advclassifiers = nn.ModuleList([copy.deepcopy(advclassifier_mat) if i<self.num_domains[0] 
        #                                            else copy.deepcopy(advclassifier_soc) for i in range(self.total_domains)])
        
        #optimizer for material adversarials
        self.disc_opt = torch.optim.Adam(
            self.advclassifiers.parameters(),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9)
        )
        cudnn.benchmark = True
        mat = count_parameters(self.mat_featurizer)
        soc = count_parameters(self.soc_featurizer) 
        reg = count_parameters(self.regressor)

        ttl= mat+soc+reg
        #print('#parameters mat, soc, reg:', mat, soc, reg, ttl)      
        
    def train_invariance(self, dic_domain, list_source, source_loader, 
                         type_inv, iter, num_splits=1, 
                         update = True, save=False, dir_out = None):
        n_source = len(list_source)
        #list_source is the list of classes
        total_domains = len(list(dic_domain.keys()))

        y_disc = 0
        num_disc = 0
        ce_loss1 = 0
        ce_loss2 =0
        cos_sim = 0
        euc = 0
        '''
        we define one adversarial classifier for each domain 
        e.g., all instances with LCO-LFP will be passed though one classifier
        '''
        for idx, source in enumerate(source_loader):
            num_disc+=1
            source1, source2 = source['src'][:,0,:,:2], source['src'][:,1,:,:2] 
            exog1, exog2 = source['meta_dat'][:,0,:3], source['meta_dat'][:,1,:3]
            mat1, mat2 = source['mat_enc'][:,0,:], source['mat_enc'][:,1,:] 
            mask1, mask2 = source['mask_seq'][:,:,:2], source['mask_seq'][:,:,2:] 
            #convert them to device for gpu
            source1, source2 = source1.to(device = self.device, dtype = torch.float32), source2.to(device = self.device, dtype = torch.float32) 
            mat1, mat2 = mat1.to(device = self.device, dtype = torch.float32), mat2.to(device = self.device, dtype = torch.float32)
            mask1, mask2 = mask1.to(device = self.device, dtype = torch.float32), mask2.to(device = self.device, dtype = torch.float32)
            exog1, exog2 = exog1.to(device = self.device, dtype = torch.float32), exog2.to(device = self.device, dtype = torch.float32) 
            
            source1_idx, source2_idx = source['inv1'][0], source['inv2'][0] #assuming batch size 1
            #print('src:', source1.size(), source2.size(), source1_idx, source2_idx)
            #print('exog:', exog1.size(), exog2.size(), mat1.size(), mat2.size(), mask1.size(), mask2.size())
            domain_key = str(source1_idx)+'-'+str(source2_idx)
            classifier_idx = dic_domain[domain_key]
            classifier = self.advclassifiers[classifier_idx]
            #print('classifier idx:', classifier_idx)
            mask1_new = copy.deepcopy(mask1[:,:,0])
            mask1_new = mask1_new[:,:,None] 
            mask2_new = mask2[:,:,1]
            mask2_new = mask2_new[:,:,None] 
            # use source1[:,:,0], mask_new for removing voltage information
            if type_inv=="material": #experimenting to keep only force to material invariance
                source1_z = self.mat_featurizer(source1, mat1, exog1, mask1) 
                source2_z = self.mat_featurizer(source2, mat2, exog2, mask2)
                source1_y, source2_y = list_source.index(source1_idx),list_source.index(source2_idx) 
 
            else: #type_inv=="soc"
                source1_z = self.soc_featurizer(source1, mat1, exog1, mask1) 
                source2_z = self.soc_featurizer(source2, mat2, exog2, mask2)
                #for soc lists are int type [50, 70, ...]
                source1_y, source2_y = list_source.index(int(source1_idx)),list_source.index(int(source2_idx)) 
            
            source1_y, source2_y = torch.tensor([source1_y], dtype=torch.long), torch.tensor([source2_y], dtype=torch.long) 
            source1_y, source2_y = source1_y.to(device = self.device, dtype=torch.long), source2_y.to(device = self.device, dtype=torch.long) 
            
            er_s1 = self.criterion(classifier(source1_z), source1_y)
            er_s2 = self.criterion(classifier(source2_z), source2_y)
            #print('cross_entropy:',idx, er_s1.item(), er_s2.item())
            ce_loss1+= er_s1.item()
            ce_loss2+= er_s2.item()

            cos_loss = F.cosine_similarity(source1_z, source2_z)
            dist = (source1_z - source2_z).pow(2).sum().sqrt()
            cos_sim+=cos_loss.item()
            euc+=dist.item()
            '''
            if save:
                self.writer.add_scalar(f"train_{type_inv}/celoss1", er_s1, idx)
                self.writer.add_scalar(f"train_{type_inv}/celoss2", er_s2, idx)
                
                with torch.no_grad():
                    torch.save(source1_z, dir_out+f'{domain_key}_{idx}_source1_z.pt')
                    torch.save(source2_z, dir_out+f'{domain_key}_{idx}_source2_z.pt') 
                    #save source1_z, source2_z, cross_entropy loss, save all the models
            '''    
            y_disc += torch.abs(er_s1 - er_s2) #y discrepancy
            
        y_disc/= num_disc
        ce_loss1/= num_disc
        ce_loss2/= num_disc
        cos_sim/= num_disc
        euc/= num_disc
        ## update the discriminator
        if update:
            disc_loss = -y_disc/num_splits
            
            #print('adv_loss:', disc_loss.item())
            self.disc_opt.zero_grad()
            self.writer.add_scalar(f"train_{type_inv}/adv", disc_loss, iter)
            self.writer.add_scalar(f"train_{type_inv}/cos", cos_sim, iter)
            self.writer.add_scalar(f"train_{type_inv}/euc", euc, iter)
            disc_loss.backward()
            #see classifier.parameters
            self.disc_opt.step()
            #see if classifier.parameters weights have changed
            # print(num_splits)
            # print(disc_loss)
            # exit(0)
        
        return y_disc, ce_loss1, ce_loss2
    
    def update(self, epochs, dic_domain, type_inv, list_source1, 
               source_loader1, dir_out, list_source2, source_loader2, save, iter):
        
        #num_loader = len(source_loader1)
        num_split =1
        #print('loader:', num_loader)
        #self.gen_soc_opt.zero_grad()
        #training material invariance
        for n_i in range(epochs):
            pos_epoch = iter*epochs+n_i
            adv, _, _ = self.train_invariance(dic_domain, list_source1, source_loader1, type_inv[0], pos_epoch, update = True, save=False)
            inv_disc, ce_loss1, ce_loss2 = self.train_invariance(dic_domain, 
             list_source1, source_loader1,
               type_inv[0], pos_epoch, update=False, save=save, dir_out = dir_out)
            inner_loss = inv_disc/num_split
            print(iter, n_i, inner_loss.item())
            #print(iter, n_i, adv.item())
            self.writer.add_scalar(f"train_{type_inv[0]}/f_mat", inner_loss, pos_epoch)
            self.writer.add_scalar(f"train_{type_inv[0]}/ce_loss1_epoch", ce_loss1, pos_epoch)
            self.writer.add_scalar(f"train_{type_inv[0]}/ce_loss2_epoch", ce_loss2, pos_epoch)
            
            self.disc_opt.zero_grad()
            self.gen_mat_opt.zero_grad()
            inner_loss.backward()
            self.gen_mat_opt.step()
            
            #if (n+1)%10==0:
            #    torch.save(self.mat_featurizer.state_dict(), dir_out+f'fpsi_mat_{n}.pth')
            #    torch.save(self.advclassifiers.state_dict(), dir_out+f'adv_{n}.pth') 
        
        #repeat line 127-134 for soc
        num_loader = len(source_loader2)
        
        for n_i in range(epochs):
            pos_epoch = iter*epochs+n_i
            adv, ce_loss1, ce_loss2 = self.train_invariance(dic_domain, list_source2, source_loader2, type_inv[1], pos_epoch, update = True, save=False)
            inv_disc, ce_loss1, ce_loss2 = self.train_invariance(dic_domain, 
                                            list_source2, source_loader2,
                                            type_inv[1], n_i, update=False, 
                                            save=save, dir_out = dir_out)
            inner_loss = inv_disc/num_split
            print(iter, n_i, inner_loss.item())
            #print(iter, n_i, adv.item())
            self.writer.add_scalar(f"train_{type_inv[1]}/f_mat", inner_loss, pos_epoch)
            self.writer.add_scalar(f"train_{type_inv[1]}/ce_loss1_epoch", ce_loss1, pos_epoch)
            self.writer.add_scalar(f"train_{type_inv[1]}/ce_loss2_epoch", ce_loss2, pos_epoch)
            
            self.disc_opt.zero_grad()
            self.gen_soc_opt.zero_grad()
            inner_loss.backward()
            self.gen_soc_opt.step()

            #if (n+1)%10==0:
            #    torch.save(self.soc_featurizer.state_dict(), dir_out+f'fpsi_soc_{n}.pth')
            #    torch.save(self.advclassifiers.state_dict(), dir_out+f'adv_soc_{n}.pth') 

    def train_reg(self, args, loader1, loader2, loader3, dic_domain, type_inv, 
                  list_source1, list_source2, save, iter, update_inv = False):
        
        '''
        if update_inv:
            self.update(args.epochs_inv, dic_domain, type_inv, list_source1, 
               loader1, args.dir_out, list_source2, loader2)
        else:
            #ckpt_iter=int(args.epochs_inv-1)
            self.mat_featurizer.load_state_dict(torch.load(args.dir_mat_model, map_location = self.device), strict=False)
            self.soc_featurizer.load_state_dict(torch.load(args.dir_soc_model, map_location = self.device), strict=False)
        '''   
        num_epochs = args.epochs_reg
        
        self.regressor.train()
        self.mat_featurizer.train()
        self.soc_featurizer.train()

        self.gen_mat_opt.zero_grad()
        self.gen_soc_opt.zero_grad()
        self.reg_opt.zero_grad() 
        self.update(args.epochs_inv, dic_domain, type_inv, list_source1, 
               loader1, args.dir_out, list_source2, loader2, save, iter)
        
        for n_r in range(num_epochs):
            num_ins = 0
            reg_loss = 0
            pos_epoch = iter*num_epochs+n_r
            for idx, data in enumerate(loader3):
                num_ins+=1
                sourcenn, target = data['src'][:,:,:2], data['trg_y']
                exog, mat = data['meta_dat'], data['mat_enc']
                mask = data['mask_seq'] 
                #convert them to device for gpu
                sourcenn = sourcenn.to(device = self.device, dtype = torch.float32)
                target = target.to(device = self.device, dtype = torch.float32)
                mat = mat.to(device = self.device, dtype = torch.float32)
                mask = mask.to(device = self.device, dtype = torch.float32)
                exog= exog.to(device = self.device, dtype = torch.float32)
                #the next 3 lines modified as to remove voltage from material
                mask_new = copy.deepcopy(mask[:,:,0])
                mask_new = mask_new[:,:,None]
                #use sourcenn[:,:,0] for removing voltage information
                mat_z = self.mat_featurizer(sourcenn, mat, exog, mask) 
                
                soc_z = self.soc_featurizer(sourcenn, mat, exog, mask) 
                logits, output = self.regressor(sourcenn, target, exog, mat, mask, mat_z, soc_z)
                #pdb.set_trace()
                loss = self.reg_criterion(output, target)
                #print('batch:',idx, loss.item())
                if iter==(args.epochs-1):
                    self.writer.add_scalar(f"train_{type_inv[2]}/lstm_batch", loss, idx)
                
                self.gen_mat_opt.zero_grad()
                self.gen_soc_opt.zero_grad() 
                self.reg_opt.zero_grad() 

                loss.backward()

                self.reg_opt.step()
                self.gen_mat_opt.step()
                self.gen_soc_opt.step()
                
                reg_loss+=loss.item()
                
            reg_loss/=num_ins
            print('reg:', iter, n_r, reg_loss)
            self.writer.add_scalar(f"train_{type_inv[2]}/lstm_epoch", reg_loss, pos_epoch)
         
        if save: #(n+1)%10==0
            torch.save(self.soc_featurizer.module.state_dict(), args.dir_out+f'fpsi_soc_{iter}.pth')
            torch.save(self.mat_featurizer.module.state_dict(), args.dir_out+f'fpsi_mat_{iter}.pth')
            torch.save(self.regressor.module.state_dict(), args.dir_out+f'reg_{iter}.pth')
            torch.save(self.advclassifiers.state_dict(), args.dir_out+f'adv_{iter}.pth')   

    def test(self, source_loader, dir_model, iter, type_inv, dic_domain, list_source,
             dir_adv_model = None, dir_out = None, test_file = None, 
             save = False, type="train"):
        
        if type_inv=="material":
            self.mat_featurizer.load_state_dict(torch.load(dir_model+f'fpsi_mat_{iter}.pth', map_location = "cpu"), strict=False)
            self.mat_featurizer.to(self.device)
            self.mat_featurizer.train()
        else:
            self.soc_featurizer.load_state_dict(torch.load(dir_model+f'fpsi_soc_{iter}.pth', map_location = "cpu"), strict=False)
            self.soc_featurizer.to(self.device) 
            self.soc_featurizer.train()
        '''
        if dir_adv_model==None:
            dir_adv_model = dir_model+f'adv_{iter}.pth'
        self.advclassifiers.load_state_dict(torch.load(dir_adv_model, map_location = "cpu"), strict=False)
        
        self.advclassifiers.to(self.device)
        self.advclassifiers.eval()
        '''
        #orig_stdout = sys.stdout
        #f= open(dir_out+f'f{type_inv}_{type}_{iter}.txt', 'w')
        #sys.stdout = f
        #with torch.no_grad():
        num_disc = 0
        cos_sim = 0
        euclidean = 0
        
        y_disc = 0
        ce_loss1 = 0
        ce_loss2 = 0
        jac_list = [] 
        for idx, source in enumerate(source_loader):
            num_disc+=1
            source1, source2 = source['src'][:,0,:,:2], source['src'][:,1,:,:2] 
            exog1, exog2 = source['meta_dat'][:,0,:3], source['meta_dat'][:,1,:3]
            mat1, mat2 = source['mat_enc'][:,0,:], source['mat_enc'][:,1,:] 
            mask1, mask2 = source['mask_seq'][:,:,:2], source['mask_seq'][:,:,2:] 
            #convert them to device for gpu
            source1, source2 = source1.to(device = self.device, dtype = torch.float32), source2.to(device = self.device, dtype = torch.float32) 
            mat1, mat2 = mat1.to(device = self.device, dtype = torch.float32), mat2.to(device = self.device, dtype = torch.float32)
            mask1, mask2 = mask1.to(device = self.device, dtype = torch.float32), mask2.to(device = self.device, dtype = torch.float32)
            exog1, exog2 = exog1.to(device = self.device, dtype = torch.float32), exog2.to(device = self.device, dtype = torch.float32) 
            
            source1_idx, source2_idx = source['inv1'][0], source['inv2'][0] #assuming batch size 1
            #print('src:', source1.size(), source2.size(), source1_idx, source2_idx)
            #print('exog:', exog1.size(), exog2.size(), mat1.size(), mat2.size(), mask1.size(), mask2.size())
            domain_key = str(source1_idx)+'-'+str(source2_idx)
            '''
            classifier_idx = dic_domain[domain_key]
            classifier = self.advclassifiers[classifier_idx]
            '''
            '''
            mask1_new = copy.deepcopy(mask1[:,:,0])
            mask1_new = mask1_new[:,:,None] 
            mask2_new = mask2[:,:,1]
            mask2_new = mask2_new[:,:,None] 
            '''   
                
            if type_inv=="material":
                source1_z, res1 = self.mat_featurizer(source1, mat1, exog1, mask1) 
                source2_z, res2 = self.mat_featurizer(source2, mat2, exog2, mask2)
                source1_y, source2_y = list_source.index(source1_idx),list_source.index(source2_idx) 
                
            else: #type_inv=="soc"
                source1_z, res1 = self.soc_featurizer(source1, mat1, exog1, mask1) 
                source2_z, res2 = self.soc_featurizer(source2, mat2, exog2, mask2)
                source1_y, source2_y = list_source.index(int(source1_idx)),list_source.index(int(source2_idx)) 
                
            
            res1['mat'] = source['inv1'][0]
            res2['mat'] = source['inv2'][0] 
            res1['meta'] = source['meta_dat'][:,0,:3].numpy().tolist()[0]
            res2['meta'] = source['meta_dat'][:,1,:3].numpy().tolist()[0]
            res1['st_id'] = source['start_id'].item()
            res1['ed_id'] = int(source['end_id'].item())
            res2['st_id'] = int(source['start_id'].item())
            res2['ed_id'] = int(source['end_id'].item())
            
            jac_list.append((res1, res2))
            print(idx)
            
            '''    
            source1_y, source2_y = torch.tensor([source1_y], dtype=torch.long), torch.tensor([source2_y], dtype=torch.long) 
            source1_y, source2_y = source1_y.to(device = self.device, dtype=torch.long), source2_y.to(device = self.device, dtype=torch.long) 
            
            er_s1 = self.criterion(classifier(source1_z), source1_y)
            er_s2 = self.criterion(classifier(source2_z), source2_y)
            
            ce_loss1+=er_s1.item()
            ce_loss2+=er_s2.item()
            y_disc+= (torch.abs(er_s1-er_s2)).item()
            
            loss = F.cosine_similarity(source1_z, source2_z)
            dist = (source1_z - source2_z).pow(2).sum().sqrt()
            
            #line = f"{idx},{loss.item()},{dist.item()}\n"
            #f.write(line)
            #print(idx, loss.item(), dist.item())
            cos_sim+= loss.item()
            euclidean+=dist.item()
            '''
            if save:
                torch.save(source1_z, dir_out+f'{domain_key}_{idx}_source1_z.pt')
                torch.save(source2_z, dir_out+f'{domain_key}_{idx}_source2_z.pt')  
                

        cos_sim/=num_disc
        euclidean/=num_disc
        '''
        ce_loss1/=num_disc
        ce_loss2/=num_disc
        y_disc/=num_disc
        '''
    
        print(iter, cos_sim, euclidean)
        '''
        print('adversarial', ce_loss1, ce_loss2, y_disc)
        
        line = f"-1,{cos_sim},{euclidean}\n"
        f.write(line)
        line = f"-2,{ce_loss1},{ce_loss2},{y_disc}\n"
        f.write(line)
        #sys.stdout = orig_stdout
        f.close()
        '''
        with open(dir_out+test_file, 'wb') as fp:
            pickle.dump(jac_list, fp)
        
        return cos_sim, euclidean
    
    def test_regressor(self, iter, data_loader, dir_model, dir_mat_model, dir_soc_model, type="train"):
        
        #pdb.set_trace()
        self.mat_featurizer.load_state_dict(torch.load(dir_mat_model+f'fpsi_mat_{iter}.pth', map_location = 'cuda:0'), strict=False)
        self.soc_featurizer.load_state_dict(torch.load(dir_soc_model+f'fpsi_soc_{iter}.pth', map_location = 'cuda:0'), strict=False)
        self.regressor.load_state_dict(torch.load(dir_model+f'reg_{iter}.pth', map_location = 'cuda:0'), strict=False)
        
        self.mat_featurizer.to(self.device)
        self.soc_featurizer.to(self.device)
        self.regressor.to(self.device)

        self.mat_featurizer.eval()
        self.soc_featurizer.eval()
        self.regressor.eval()

        forecasts=[]
        start_t=[]
        end_t=[]
        list_did=[]
        predicted_nowcast={}
        metric1= RMSE()
        metric2= MAPE()

        with torch.no_grad():
            num_ins = 0
            test_rmse= 0
            test_mape=0
            for idx, data in enumerate(data_loader):
                num_ins+=1
                sourcenn, target = data['src'][:,:,:2], data['trg_y']
                exog, mat = data['meta_dat'], data['mat_enc']
                mask = data['mask_seq'] 

                sids, eids, data_ids = data['start_id'], data['end_id'], data['battery_id']
                sids, eids = sids.tolist(), eids.tolist()
                #convert them to device for gpu
                mask_new = copy.deepcopy(mask[:,:,0])
                mask_new = mask_new[:,:,None] 
                
                sourcenn = sourcenn.to(device = self.device, dtype = torch.float32)
                target = target.to(device = self.device, dtype = torch.float32)
                mat = mat.to(device = self.device, dtype = torch.float32)
                mask = mask.to(device = self.device, dtype = torch.float32)
                exog= exog.to(device = self.device, dtype = torch.float32)
                
                #mask_new = mask_new.to(device=self.device, dtype = torch.float32)
                #pdb.set_trace()
                mat_z = self.mat_featurizer(sourcenn, mat, exog, mask) 
                soc_z = self.soc_featurizer(sourcenn, mat, exog, mask) 
                #mat_z = self.mat_featurizer(sourcenn.float(), mat.float(), exog.float(), mask.float()) 
                #soc_z = self.soc_featurizer(sourcenn.float(), mat.float(), exog.float(), mask.float()) 

                logits, output = self.regressor(sourcenn.float(), target, exog.float(), mat.float(), mask.float(), mat_z.float(), soc_z.float())
            
                rmse_score = metric1(output, target)
                test_rmse+=rmse_score.item()
                self.writer.add_scalar(f'{type}_reg/batch_rmse',rmse_score.item(),idx)
            
                mape_score=metric2(output, target)
                test_mape+=mape_score.item()
                self.writer.add_scalar(f'{type}_reg/batch_mape',mape_score.item(),idx)

                output = output.cpu().numpy()
                post_process_results(sids, eids, data_ids, output, predicted_nowcast)
            
                if len(forecasts) == 0:
                    forecasts = output 
                else:
                    forecasts=np.concatenate([forecasts, output], axis=0)
            
                start_t+=sids
                end_t+=eids
                list_did+=data_ids
            
            test_rmse/=num_ins
            test_mape/=num_ins

            print('rmse,mape:',test_rmse, test_mape)

        return forecasts, start_t, end_t, list_did, test_rmse, test_mape


        

             

        
 

            
            






