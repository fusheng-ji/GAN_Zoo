import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dim of latent')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='path of dataset')
    return parser.parse_args()