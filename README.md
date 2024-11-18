# Battery Safety Invariance Learning
- Please extract data.zip for original datasets
## dependencies
   - tensorboard==2.12.0
   - torch==1.13.1
   - torch-tb-profiler==0.4.1
   - torchmetrics==0.11.3
   - torchvision==0.14.1 

## train the invariance model
 - epochs: #outer iterations
 - epochs_inv: iterations to run invariant models
 - epochs_reg: iterations to run predictor
 - num_class: number of material class (we used 3, one to test on unseen)
 - list_mat_inv: list of material pairs to train f_psi_mat
 - list_soc_inv: list of soc pairs to train f_psi_soc
 - dir_tbs: tensor board directory
 - dir_out: save model output directory
 - dir_data: data directory
 - dim_val: regressor hidden layer size
 - num_layers: lstm layer size
 - learning_rate: regressor learning rate
 - regressor batch_size
 - lr_g, lr_d: learning rate for invariant and adversaril models

## Training
```
python main.py --epochs 70 --epochs_inv 2 --epochs_reg 4 \
--batch_size 120 --batch_size_inv 1  --num_class 3 \
--list_mat_inv LCO LFP --list_soc_inv 50 70 90 \
--mat_list LCO LFP NMC LMO-LNO ORNL NMC-LMO \
--dir_tbs sample --dir_out ../results/ \
--dir_data ../data/snl_battery/ --pad_value 0 \
--save_freq 10 --dim_val 128 --lstm_layers 4 --learning_rate .001 \
--lr_g 0.0002 --lr_d 0.0002
```
## Test invariant model
# epochs: which model checkpoint to load
# epochs_inv: epochs+1
# batch_size 120 
# dir_model: directory of regressor checkpoints 
# dir_mat_model: directory of f_psi_mat
# dir_soc_model: directory of f_psi_soc
# test_file: output file forecasted temperarture of 50 timesteps per instance 

```
python main.py --test_only --epochs 69 --epochs_inv 70 --batch_size 120 \
--dir_model ../results/ \
--dir_mat_model ../results/ \
--dir_soc_model ../results/ \
--dir_out ../results/ \
--dir_tbs  sample --dir_data ../data/snl_battery/ \
--dim_val 128 --save_freq 1 --lstm_layers 4 --test_split 0.1 \
--list_mat_inv LCO LFP --list_soc_inv 50 \
--max_len 40000 --pad_value 0 \
--mat_list LCO LFP NMC LMO-LNO ORNL NMC-LMO \
--test_file forecast_seen_mat_seen_soc_ckpt1 --num_class 3
```

# Citation
- For original citation please use google scholar

@inproceedings{counter2024tabassum,
  author    = {Anika Tabassum, Srikanth Allu, Ramakrishnan Kannan, Nikhil Muralidhar},
  title     = {Counter Data Paucity through Adversarial Invariance Encoding: A Case Study on Modeling Battery Thermal Runaway},
  booktitle = {Proceedings of the IEEE International Conference on Big Data (BigData)},
  year      = {2024},
  publisher = {IEEE},
  address   = {Washington DC, USA},  
  month     = {December}
}
