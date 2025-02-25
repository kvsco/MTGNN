import argparse
import torch
import datetime
from mtgnn import Trainer

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def run_model(config, flag='train'):
    model = Trainer(config)
    print(f'flag : {flag}')
    if flag == 'train':
        setting = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.train(setting)
        model.test(setting)
    elif flag == 'test':
        model.test('2025_02_24_15_02_33',1)
    torch.cuda.empty_cache()
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/dyd9800/dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='machine_train.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='machine_7_8',
                        help='target feature in S or MS task')  # txn_elapse, tps
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model detail
    parser.add_argument('--num_nodes', type=int, default=228, help='number of nodes/variables')
    parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,
                        help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
    parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--subgraph_size', type=int, default=20, help='k')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

    parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=128, help='end channels')

    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')

    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--clip', type=int, default=5, help='clip')
    parser.add_argument('--step_size1', type=int, default=2500, help='step_size')
    parser.add_argument('--step_size2', type=int, default=100, help='step_size')

    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--seed', type=int, default=101, help='random seed')
    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')

    # GPU
    parser.add_argument('--device', type=str, default='cuda:1', help='')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    performances = run_model(args, flag='train')
