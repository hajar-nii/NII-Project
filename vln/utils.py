import json
import os
import random
import shutil
import math

import torch
import yaml
import numpy as np
import warnings
from tensorboardX import SummaryWriter
import networkx as nx
import sentencepiece as spm

import matplotlib.pyplot as plt 

padding_idx = 0


class DotDict(dict):
    __getattr__ = dict.__getitem__


def get_config(config_file, oracle):

    def get_value_or_default(name, default, input_config):
        if name in input_config:
            value = input_config[name]
            del input_config[name]
            return value
        print('value "{}" not found in config; use default: "{}"'.format(name, default))
        return default


    with open(config_file) as f:
        config_yaml = yaml.safe_load(f)
    input_config = DotDict(config_yaml)

    config = DotDict()
    # train
    config['iterations'] = get_value_or_default('iterations', 1, input_config)
    config['max_num_epochs'] = get_value_or_default('max_num_epochs', 150, input_config)

    # optimizer
    config['optimizer'] = get_value_or_default('optimizer', 'adamw', input_config)
    config['weight_decay'] = get_value_or_default('weight_decay', 0.001, input_config)
    config['learning_rate'] = get_value_or_default('learning_rate', 0.0005, input_config)

    # model
    config['hidden_dim'] = get_value_or_default('hidden_dim', 256, input_config)
    config['embedding_dim'] = get_value_or_default('embedding_dim', 32, input_config)
    config['num_heads'] = get_value_or_default('num_heads', 2, input_config)
    config['vocab_size'] = get_value_or_default('vocab_size', 2000, input_config)

    # regularization
    config['use_layer_norm'] = get_value_or_default('use_layer_norm', True, input_config)
    config['dropout'] = get_value_or_default('dropout', 0.3, input_config)
    config['rnn_state_dropout'] = get_value_or_default('rnn_state_dropout', 0.1, input_config)
    config['attn_dropout'] = get_value_or_default('attn_dropout', 0.3, input_config)

    # features
    #   vis
    config['use_image_features'] = get_value_or_default('use_image_features', 'resnet_fourth_layer', input_config)
    #       regularization
    config['img_feature_dropout'] = get_value_or_default('img_feature_dropout', 0.0, input_config)
    #   graph
    config['junction_type_embedding'] = get_value_or_default('junction_type_embedding', True, input_config)
    config['heading_change'] = get_value_or_default('heading_change', True, input_config)
    config['heading_change_noise'] = get_value_or_default('heading_change_noise', 0.1, input_config)

    # ablation
    config['second_rnn'] = get_value_or_default('second_rnn', True, input_config)
    config['do_sample_bpe'] = get_value_or_default('do_sample_bpe', True, input_config)
    config['use_text_attention'] = get_value_or_default('use_text_attention', True, input_config)
    config['use_image_attention'] = get_value_or_default('use_image_attention', True, input_config)

    # oracle
    config['oracle_initial_rotation'] = 'r' in oracle
    config['oracle_directions'] = 'd' in oracle
    config['oracle_stopping'] = 's' in oracle

    assert len(input_config) == 0

    assert config.use_layer_norm is False or config.use_layer_norm is True

    assert 256 % config.num_heads == 0

    assert config.use_image_features in [False, 'resnet_fourth_layer', 'resnet_last_layer', 'segmentation']

    return config

def get_scans():
    with open('datasets/r2r/scans.txt') as f: # works fine 
        scans = [scan.strip() for scan in f.readlines()]
        # print ("scans :", scans)
        # print(len(scans))
    return scans

def get_scan_index(scan):
    scans= get_scans()
    return scans.index(scan)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_tokenizer(opts):
    model_file = "{}/vocab/{}_vocab_2000.model".format(opts.dataset_dir, opts.dataset)
    print(model_file)
    if not os.path.isfile(model_file):
        print('Could not find vocab with 2000, looking for 3000')
        model_file = "{}/vocab/{}_3000.model".format(opts.dataset_dir, opts.dataset)
    if not os.path.isfile(model_file):
        raise ValueError('could not find vocab model file: {}'.format(model_file))
    tokenizer = spm.SentencePieceProcessor(model_file=model_file)
    assert tokenizer.pad_id() == padding_idx
    return tokenizer


def load_datasets(splits, opts=None):
    
    data = []
    for split in splits:
        assert split in ["train", "test", "val_seen", "val_unseen"]
        # with open('%s/data/%s.json' % (opts.dataset_dir, split)) as f:
        with open('datasets/r2r/data/%s.json' %  split) as f:
            # for line in f:
                # print('for line in f: \n')
                # print(type(line) )
                # item = dict(json.loads(line))
            data_list = json.loads(f.read())
            # print (data)
            for item in data_list: #! item is a dict
                # print (item)
                # print ('\n')
                # print (type(item))
                
                item["heading"] = (item["heading"] *180.0) / math.pi
                # for instruction in item["instructions"]:
                    # all_instructions = []
                    # instruction = instruction.lower() #! modified
                    # all_instructions.append(instruction)
                    # item.update(instructions= all_instructions)
                data.append(item)
                # print (item['instructions'])
                # print('\n')
    return data

def scans_in_split_json(split):
        train_data = []
        with open('datasets/r2r/data/%s.json' % split)  as f:
            # for line in f:
                # print('for line in f: \n')
                # print(type(line) )
                # item = dict(json.loads(line))
            data_list = json.loads(f.read())
            # print (data)
            for item in data_list: #! item is a dict
                # print (item)
                # print ('\n')
                # print (type(item))
                for instruction in item["instructions"]:
                    all_instructions = ""
                    instruction = instruction.lower() #! modified
                    all_instructions += instruction
                    item.update(instructions= all_instructions)
                train_data.append(item)
        # train_data = load_datasets(["train"])
        scans_in_train_json = []
        for item in train_data:
            scans_in_train_json.append(item["scan"])
        random.shuffle(scans_in_train_json)
        return scans_in_train_json

 

def set_tb_logger(log_dir, resume):
    """ Set up tensorboard logger"""
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        import shutil
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


def load_nav_graph_old(opts, scan_id):
    with open("datasets/r2r/graph/links_orar/links_%s.txt" % scan_id) as f: #! hard coded the path since we just want r2r
            G = nx.Graph()
            for line in f:
                pano_1, _, pano_2 = line.strip().split(",")
                G.add_edge(pano_1, pano_2)     
            # print ("G :\n", G) 
    return G 


    
def load_nav_graph(opts, scan_id):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    with open('datasets/r2r/connectivity/%s_connectivity.json' % scan_id) as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i,item in enumerate(data): #for each viewpoint
            if item['included']:
                for j,conn in enumerate(item['unobstructed']): #for each neighboring viewpoint (0 is the viewpoint itself)
                    if conn and data[j]['included']: #if the neighbor is also included in the simulator
                        positions[item['image_id']] = np.array([item['pose'][3],
                                item['pose'][7], item['pose'][11]]);
                        # print ("positions :", positions[item['image_id']])
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
        nx.set_node_attributes(G, values=positions, name='position')  
        # nx.draw(G , with_labels=True)
        # plt.show()
    return G

def load_all_nav_graphs_old(opts):
    scans = get_scans()
    graphs = {}
    for scan in scans:
            G = load_nav_graph(opts, scan)
            graphs[scan] = G
    return graphs
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#TODO: CHECK LATER if here are any problems
def resume_training(opts, model, instr_encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opts.resume == 'latest':
        file_extention = '.{}.pth.tar'.format(opts.iteration)
    elif opts.resume == 'SPD_best':
        file_extention = '_model_SPD_best.{}.pth.tar'.format(opts.iteration)
    elif opts.resume == 'TC_best':
        file_extention = '_model_TC_best.{}.pth.tar'.format(opts.iteration)
    else:
        raise ValueError('Unknown resume option: {}'.format(opts.resume))

    weights_dir = os.path.join(opts.output_dir, 'weights')
    opts.resume = os.path.join(weights_dir, 'ckpt' + file_extention)
    if os.path.isfile(opts.resume):
        checkpoint = torch.load(opts.resume, map_location=device)
        opts.start_epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
        instr_encoder.load_state_dict(checkpoint['instr_encoder_state_dict'])

        try:
            SPD = checkpoint['SPD']
        except KeyError:
            print('SPD not provided in ckpt, set to inf.')
            SPD = float('inf')
        try:
            TC = checkpoint['TC']
        except KeyError:
            print('TC not provided in ckpt, set to 0.0.')
            TC = 0.0
        print("=> loaded checkpoint '{}' (iteration {}, epoch {})".format(opts.resume, opts.iteration, checkpoint['epoch']-1))
    else:
        raise ValueError("=> no checkpoint found at '{}' in iteration {}".format(opts.resume, opts.iteration))
    return model, instr_encoder, SPD, TC


def save_checkpoint(state, is_best_TC, epoch=-1):
    opts = state['opts']
    weights_dir = os.path.join(opts.output_dir, 'weights')
    filename = os.path.join(weights_dir, 'ckpt.{}.pth.tar'.format(opts.iteration))
    if opts.store_ckpt_every_epoch:
        filename = os.path.join(weights_dir, 'ckpt.{}.{}.pth.tar'.format(opts.iteration, epoch))

    dotdict_config = state['opts'].config
    state['opts'].config = dict(dotdict_config)
    torch.save(state, filename)
    state['opts'].config = dotdict_config

    if is_best_TC:
        best_filename = os.path.join(weights_dir, 'ckpt_model_TC_best.{}.pth.tar'.format(opts.iteration))
        shutil.copyfile(filename, best_filename)



if __name__ == "__main__":
    scans = get_scans()
    # graph = load_nav_graph('gZ6f7yhEvPG')
    # load_tokenizer(opts)
    
    load_datasets(["train"])