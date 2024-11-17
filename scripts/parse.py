import argparse

def parse_args():
    parser = argparse.ArgumentParser('argument for invariant learning forecasting model')
    #for ddp training
    parser.add_argument('--nodes', default=1,type=int, metavar='N')
    parser.add_argument('--gpus', default=3, type=int, help='number of gpus')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    
    #training parameters
    parser.add_argument('--batch_size', default=3, type=int, help='batch size')
    parser.add_argument('--batch_size_inv', default=1, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, 
                        help='learning rate for task regressor')
    parser.add_argument('--lr_d', default=0.001, type=float, 
                        help='learning rate for adversarials')
    parser.add_argument('--lr_g', default=0.001, type=float, 
                        help='learning rate for invariance')
    parser.add_argument('--weight_decay_g', default=0, type=float, 
                        help='weight decay for invariance')
    parser.add_argument('--weight_decay_d', default=0, type=float, 
                        help='weight decay for adversarial')                   
    parser.add_argument('--weight_decay_l', default=0, type=float, 
                        help='weight decay for regressor')                     
    parser.add_argument('--advfactor_gamma', default=1.0, type=float, 
                        help='learning rate decay for invariance optimizer')                    
    parser.add_argument('--beta', default=0.5, type=float, 
                        help='beta for adam optimizer')
    parser.add_argument('--epochs', default=2000, type=int, 
                        help='number iterations for entire model')
    parser.add_argument('--epochs_inv', default=1, type=int, 
                        help='number iterations for invariance for alternate training') 
    parser.add_argument('--epochs_reg', default=1, type=int, 
                        help='number iterations for regressor for alternate training')  
    parser.add_argument('--random_seed', default=0, type=int, 
                        help='seed value for random number generation') 
    parser.add_argument('--print_freq', default=20, type=int, 
                        help='frequence to print training loss')
    parser.add_argument('--save_freq', default=10, type=int, 
                        help='frequency to save model')
    
    #parser.add_argument('--device', default='gpu', type=str, help='for test cpu/gpu')
    parser.add_argument('--metric', default='RMSE', type=str, help='loss type')
    parser.add_argument('--dataset', default='thermal-runoff', type=str, 
                        help='select dataset to train (thermal-runoff/batteryarchive)')
    
    #regressor model related
    parser.add_argument('--dim_val', type = int, nargs = '+', required = True, 
                        help='dimension of lstm hidden layers')
    parser.add_argument('--lstm_layers', default=4, type=int, 
                        help='number of lstm layers for multi-lstm')
    #model related parameters
    parser.add_argument('--mlp_width', default=256, type=int, 
                        help='number of neurons for linear layers for invariant')
    parser.add_argument('--mlp_depth', default=4, type=int, 
                        help='number of layers for NN for invariant')  
    parser.add_argument('--mlp_dropout', default=0.2, type=float, 
                        help='dropout prob. for NN for invariant') 
     
    #dataset related parameters
    parser.add_argument('-n', '--mat_list', nargs='+', type = str, 
                        default=['LCO', 'LFP','NMC', 'LMO-LNO', 'ORNL', 'NMC-LMO'], 
                        help="material list available")
    parser.add_argument('--list_mat_inv', nargs='+', type =str, default=['LCO', 'LFP', 'NMC'], 
                        help="material list for training/testing invariant model")
    parser.add_argument('--list_soc_inv', nargs='+', type = int, default=[0, 50, 70, 90], 
                        help="soc list for training/testing invariant model")
    parser.add_argument('--input_size', default=2, type=int, 
                        help='number of features/timeseries for input')
    parser.add_argument('--max_len', default=40000, type=int, 
                        help='max len of the masked input sequence. #snl: 39240, ornl:11574')
    parser.add_argument('--pad_value', default=0, type=int, 
                        help='padded value for masked sequence')
    parser.add_argument('--n_heads', default=4, type=int, 
                        help='#attention head is using transformer')
    parser.add_argument('--input_dim', default=64, type=int, 
                        help='model embedding dimension for transformer')
    parser.add_argument('--all_enc', default=3, type=int, help='dimension of metadata')
    parser.add_argument('--num_class', default=6, type=int, 
                        help='number of materials, original class=6')
    parser.add_argument('--output_size', default=1, type=int, 
                        help='number of features/timeseries in output') 
    parser.add_argument('--time_lag', default=50, type=int,
                        help='time lag from input start seq to forecast start time')
    parser.add_argument('-lhs','--history_size', default=100, type=int, metavar='N',
                        help='input sequence length')
    parser.add_argument('--out_seq_len', default=50, type=int, 
                        help='the length of the models output/target sequence length')
    
    #file/data directory related
    parser.add_argument('--algorithm', type=str, default='all',
                        help='data shuffle algorithm for invariance/snl/ornl')
    
    parser.add_argument('--dir_data', type=str, default = '../data/snl_battery/',
                        help='data directory')
    parser.add_argument('--dir_tbs', type=str, default = 'train_inv',
                        help='tensorboard directory')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--load', action='store_true', default = False,
                        help = "load data from already saved!!")
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='test split') 
    parser.add_argument('--dir_out', type=str, required= True,
                        help='output directory')
    parser.add_argument('--dir_model', type=str, default= 'inv_z/',
                        help='data directory')
    parser.add_argument('--dir_soc_model', type=str, default= 'inv_soc/f_psi_soc_229.pth',
                        help='full file path of the soc model checkpoint')
    parser.add_argument('--dir_mat_model', type=str, default= 'inv_mat/f_psi_mat_299.pth',
                        help='full file path of the mat model checkpoint')
    parser.add_argument('--test_file', type=str, default='regressor_forecast',
                        help='forecast result filename')
    parser.add_argument('--dir_list', default='../data/mechanical_loading_datasets/', 
                        type=str, help='battery list directory')
    parser.add_argument('--in_file', default='raw_mechanical_loading_data.csv', type=str,
                        help='data file name')
    
    
    args = parser.parse_args()
    # hyper parameters for invariant model
    hparams = {}
    hparams['mlp_width'] = args.mlp_width
    hparams['mlp_depth'] = args.mlp_depth 
    hparams['mlp_dropout'] = args.mlp_dropout 
    hparams['resnet18'] = True
    
    # hyper parameters for adversarial optimizer
    hparams["lr_d"] =  args.lr_d
    hparams['weight_decay_d'] = args.weight_decay_d
    hparams['beta1'] = args.beta

    # hyper parameters for invariant optimizer
    hparams["lr_g"] = args.lr_g 
    hparams['weight_decay_g'] = args.weight_decay_g 
    hparams["advfactor_gamma"] =  args.advfactor_gamma # change accordingly to set lr!=10^-3

    #hyper parameters for LSTM regressor
    hparams['lr_l'] = args.learning_rate
    hparams['weight_decay_l'] = args.weight_decay_l
    hparams['dim_hidden'] = args.dim_val
    hparams['num_layers'] = args.lstm_layers
    hparams['checkpnt_invar'] = args.epochs_inv
    hparams['out_seq_len'] = args.out_seq_len
    hparams['out_features'] = args.output_size
    hparams['n_features'] = args.input_size
    hparams['history_size'] = args.history_size
    hparams['mask_dim'] = int(args.max_len/args.time_lag)
    hparams['enc_dim'] = args.all_enc
    hparams['pos_enc'] = args.num_class

    #hyperparameters for transformer
    hparams["n_heads"] = args.n_heads
    hparams["input_dim"] = args.input_dim

    return args, hparams
    
