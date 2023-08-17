import os
import pickle
import random

import numpy as np
import networkx as nx
import torch

from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dis

from base_navigator import BaseNavigator
from utils import load_datasets, load_nav_graph, get_scans, get_scan_index, scans_in_split_json

_SUCCESS_THRESHOLD = 2


scans_in_train_json = scans_in_split_json('train')
scans_in_val_json = scans_in_split_json('val_seen')



def load_features_scan(features_dir, features_name, scan_id):
    print("=================================")
    print("=====Loading image features for scan %s ======" % scan_id)
    assert type(features_name) == str
    feature_file = os.path.join(features_dir, str(scan_id) + '_'+ features_name + '.pickle')
    print('feature_file', feature_file)
    if os.path.isfile(feature_file):
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
            # print(features.keys())
            # print('\n')
    else:
        raise ValueError('could not read image features')
    assert len(features) > 0
    any_pano = list(features.keys())[0]
    # any_features = list(features.values())[0]
    any_features = list(features[any_pano].values())[0]
    feature_shape = any_features.shape
    assert 'feature_shape' not in features
    features['feature_shape'] = feature_shape
    return features

def load_features (features_dir, features_name):
    features_list = []
    scans = get_scans()
    for scan in scans:
        features = load_features_scan(features_dir, features_name, scan)
        features_list.append(features)
    return features_list




class EnvBatch:
    def __init__(self, opts, splits, image_features, batch_size=10, name=None, tokenizer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.name = name
        self.image_features = image_features
        self.tokenizer = tokenizer
        self.scans = get_scans()
        self.batch_scans = []
        # self.all_img_features = []

        self.split = splits[0]

        self.navs = []
        print("=====Initializing %s navigators=====" % self.name)
        for i in range(batch_size):  # tqdm(range(batch_size)):
            # rand_scan = random.randint(0, 89)
            #TODO: when doing random, check that there are no doubles in navs
            
            if self.split == 'train':
                self.batch_scans.append(scans_in_train_json[i])
                nav = BaseNavigator(self.opts.dataset_dir, scans_in_train_json[i])
                self.navs.append(nav)

            elif self.split == 'val_seen':
                self.batch_scans.append(scans_in_val_json[i])
                nav = BaseNavigator(self.opts.dataset_dir, scans_in_val_json[i])
                self.navs.append(nav)

        print("=====================================")

        # self.all_img_features.append(self.image_features[0])
        # tmp = np.concatenate(self.all_img_features, self.image_features[1])
        # for i in range (1,len(image_features)):
        #     self.all_img_features= np.concatenate(self.all_img_features, self.image_features[i])

    #! The panos ids and heading are all in the batch
    #! but panoId and heading doesn't exactly correspond to the scan of navs[i]
    def new_episodes(self, pano_ids, headings):
        """ Iteratively initialize the simulators for # of batchsize"""
        print ("Starting a new episode \n")
        for i, (panoId, heading) in enumerate(zip(pano_ids, headings)):
            # print ("i \n", i)
            # print ("panoId \n", panoId)
            # print ("heading \n", heading)
            # print ("navs[i].scan_id \n", self.navs[i].scan_id)
            self.navs[i].graph_state = (panoId, heading)
            self.navs[i].initial_pano_id = panoId
        # print ("finished loading graph_state \n")

    def get_nearest_heading(self, nav, pano,  heading):
        """ Get the nearest heading in the graph given the heading"""
        # pano, _ = nav.graph_state
        # print ("inside get_nearest_heading \n")
        # print ("actual nav is for scan \n", nav.scan_id)
        # print ("nav.graph.nodes[pano]: \n", nav.graph.nodes[pano])
        neighbors = nav.graph.nodes[pano].neighbors
        headings = [n for n in neighbors.keys()]
        nearest_heading = min(headings, key=lambda x: abs(x - heading))
        return nearest_heading
            
    def _get_imgs(self, batch_size):
        imgs = []
        for i in range(batch_size):
            nav = self.navs[i]
            pano, heading_exact = nav.graph_state
            feature_for_scan = self.image_features[get_scan_index(nav.scan_id)]
            #TODO:  l'algo ne trouve pas le heading exact dans le dictionnaire
            #TODO:  il faut donc trouver le heading le plus proche
            #! Done !
            heading = self.get_nearest_heading(nav, pano ,heading_exact)
            image_feature = feature_for_scan[pano][heading] #! for the actual scan
            imgs.append(image_feature)
        imgs = np.array(imgs, dtype=np.float32)
        #  print ("imgs shape \n", imgs.shape)
        if self.opts.config.use_image_features == 'resnet_fourth_layer':
            assert imgs.shape[-1] == 100  # (batch_size, 100, 100)
            # print ('Getting images from 4th layer\n')
        elif self.opts.config.use_image_features == 'resnet_last_layer':
            assert imgs.shape[-1] == 2048  # (batch_size, 5, 2048)
        elif self.opts.config.use_image_features == 'segmentation':
            assert imgs.shape[-1] == 25  # (batch_size, 5, 25)
        else:
            ValueError('image features not processed')
        return torch.from_numpy(imgs).to(self.device)

    def _get_junction_type(self, batch_size):
        junction_types = []
        for i in range(batch_size):
            nav = self.navs[i]
            pano, _ = nav.graph_state

            num_neighbors = len(nav.graph.nodes[pano].neighbors) #! only one graph
            if num_neighbors == 3:
                junction_type = 1
            elif num_neighbors == 4:
                junction_type = 2
            elif num_neighbors > 4:
                junction_type = 3
            else:
                junction_type = 0

            junction_types.append(junction_type)
        junction_types = np.array(junction_types, dtype=np.int64)  # (batch_size)
        return torch.from_numpy(junction_types).to(self.device)

    def _get_gt_action(self, batch):
        gt_action = []
        for i, item in enumerate(batch):
            gt_action.append(self._get_gt_action_i(batch, i))

        gt_action = np.array(gt_action, dtype=np.int64)
        return torch.from_numpy(gt_action).to(self.device)

    def _get_gt_action_i(self, batch, i):
        nav = self.navs[i]
        print ("Start _get_gt_action_i \n")
        print ("nav.scan_id \n", nav.scan_id)
        print ("nav.graph_state \n", nav.graph_state)
        gt_path = batch[i]['path']
        panoid, heading = nav.graph_state
        if panoid not in gt_path:
            return None
        pano_index = gt_path.index(panoid)
        if pano_index < len(gt_path) - 1:
            gt_next_panoid = gt_path[pano_index + 1]
        else:
            gt_action = 3  # STOP
            return gt_action
        pano_neighbors = nav.graph.nodes[panoid].neighbors
        print ("pano neighbors \n", pano_neighbors)
        neighbors_id = [neighbor.panoid for neighbor in pano_neighbors.values()]
        print ("neighbors id \n", neighbors_id)
        print ("gt next panoid \n", gt_next_panoid)
        print ("pano neighbors keys \n", list(pano_neighbors.keys()))
        gt_next_heading = list(pano_neighbors.keys())[neighbors_id.index(gt_next_panoid)]
        delta_heading = (gt_next_heading - heading) % 360
        if delta_heading == 0:
            gt_action = 0  # FORWARD
        elif delta_heading < 180:
            gt_action = 2  # RIGHT
        else:
            gt_action = 1  # LEFT
        return gt_action

    def _action_select(self, a_prob, ended, num_act_nav, trajs, total_steps, batch):
        """Called during testing."""
        a = []
        heading_changes = []
        action_list = ["forward", "left", "right", "stop"]
        for i in range(len(batch)):
            nav = self.navs[i]
            if ended[i].item():
                a.append([3])
                heading_changes.append([[0]])
                continue

            action_index = a_prob[i].argmax()
            action = action_list[action_index]

            if self.opts.config.oracle_initial_rotation:
                if len(trajs[i]) == 1:
                    gt_action_index = self._get_gt_action_i(batch, i)
                    action = action_list[gt_action_index]
            if self.opts.config.oracle_directions:
                pano, _ = nav.graph_state
                if len(nav.graph.nodes[pano].neighbors) > 2:
                    gt_action_index = self._get_gt_action_i(batch, i)
                    if gt_action_index is not None: # agent is still on the gold path
                        if gt_action_index != 3: # if intersection is also stopping; don't tell the model
                            action = action_list[gt_action_index]
            if self.opts.config.oracle_stopping:
                gt_action_index = self._get_gt_action_i(batch, i)
                if gt_action_index is not None:  # agent is still on the gold path
                    if gt_action_index == 3 or action == 'stop':  # oracle if gold is stop or agent incorrectly stops
                        action = action_list[gt_action_index]

            if action == "stop":
                ended[i] = 1
                num_act_nav[0] -= 1

            nav.step(action)
            a.append([action_list.index(action)])
            heading_change = nav.get_heading_change()
            heading_changes.append([[heading_change]])

            new_pano, new_heading = nav.graph_state
            if not nav.prev_graph_state[0] == nav.graph_state[0]:
                trajs[i].append(new_pano)

            total_steps[0] += 1
        a = np.asarray(a, dtype=np.int64)
        heading_changes = np.asarray(heading_changes, dtype=np.float32)
        return torch.from_numpy(a).to(self.device), torch.from_numpy(heading_changes).to(self.device)

    def cal_cls(self, graph, traj, gt_traj):
        PC = np.mean(np.exp([-np.min(
                [nx.dijkstra_path_length(graph, traj_point, gt_traj_point)
                for traj_point in traj])
                for gt_traj_point in gt_traj]))
        EPL = PC * len(gt_traj)
        LS = EPL / (EPL + np.abs(EPL - len(traj)))
        return LS * PC

    def cal_dtw(self, graph, prediction, reference, success):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction)+1):
            for j in range(1, len(reference)+1):
                best_previous_cost = min(
                    dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
                cost = nx.dijkstra_path_length(graph, prediction[i-1], reference[j-1])
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        dtw_group = [dtw]
        ndtw = np.exp(-dtw/(_SUCCESS_THRESHOLD * np.sqrt(len(reference))))
        dtw_group += [ndtw, success * ndtw]
        return dtw_group

    def _eva_metrics(self, trajs, batch, graph, metrics):
        for i, item in enumerate(batch):
            success = 0
            traj = trajs[i]
            gt_traj = item["path"]
            ed = edit_dis(traj, gt_traj)
            ed = 1 - ed / max(len(traj), len(gt_traj))
            # print ("gt_traj[-1] is :\n ", gt_traj[-1])
            # print ("item is :\n ", item["scan"])
            # print ("graph is :\n ", graph[item["scan"]])
            target_list = list(nx.all_neighbors(graph[item["scan"]], gt_traj[-1])) + [gt_traj[-1]]
            if traj[-1] in target_list:
                success = 1
                metrics[0] += 1
                metrics[2] += ed

            metrics[1] += nx.dijkstra_path_length(graph[item["scan"]], traj[-1], gt_traj[-1])
            if self.opts.CLS:
                metrics[3] += self.cal_cls(graph[item["scan"]], traj, gt_traj)
            if self.opts.DTW:
                dtw_group = self.cal_dtw(graph[item["scan"]], traj, gt_traj, success)
                for j in range(-3, 0):
                    metrics[j] += dtw_group[j]

    def action_step(self, target, ended, num_act_nav, trajs, total_steps):
        action_list = ["forward", "left", "right", "stop"]
        heading_changes = list()
        for i in range(len(ended)):
            nav = self.navs[i]
            if ended[i].item():
                heading_changes.append([[0.0]])
                continue
            action = action_list[target[i]]
            if action == "stop":
                ended[i] = 1
                num_act_nav[0] -= 1
            nav.step(action)
            heading_change = nav.get_heading_change()
            if self.opts.config.heading_change_noise > 0:
                noise = np.random.normal(0.0, abs(heading_change * self.opts.config.heading_change_noise))
                if action == 'left' and (heading_change + noise) >= 0:
                    noise = 0
                if action == 'right' and (heading_change + noise) <= 0:
                    noise = 0
                heading_change += noise
                heading_change = max(-1.0, heading_change)
                heading_change = min(1.0, heading_change)
            heading_changes.append([[heading_change]])

            new_pano, new_heading = nav.graph_state
            if not nav.prev_graph_state[0] == nav.graph_state[0]:
                trajs[i].append(new_pano)

            total_steps[0] += 1
        heading_changes = np.asarray(heading_changes, dtype=np.float32)
        return torch.from_numpy(heading_changes).to(self.device)


class OutdoorVlnBatch:
    def __init__(self, opts, image_features, batch_size=10, splits=["train"], tokenizer=None, name=None, sample_bpe=False):
        self.env = EnvBatch(opts, splits, image_features, batch_size, name, tokenizer)
        #! self.env already has the scan ids (self.env.batch_scans)
        self.opts = opts

        self.batch_size = batch_size
        self.splits = splits

        self.tokenizer = tokenizer
        self.json_data = load_datasets(splits, opts)
        self.sample_bpe = sample_bpe

        self.data = None

        self.graph = {}
        self.minibatch_scans = []

        self.split = splits[0]

        # print ("Iniitializing Outdoor VLN Batch:\n ")
        # print (self.env.batch_scans)
      
        self.reset_epoch()

        self._load_nav_graph()

    # ! we only need to get necessary data to our batch
    def _get_data(self):
        data = []
        tokenizer = self.tokenizer
        for i, item in enumerate(self.json_data):
            if self.split == "train":
                if item["scan"] in self.env.batch_scans:
                    instr = item["instructions"]
                    if self.sample_bpe:
                        _encoder_input = tokenizer.encode(instr, enable_sampling=True, alpha=0.3, nbest_size=-1)
                    else:
                        _encoder_input = tokenizer.encode(instr)
                    _encoder_input.append(tokenizer.eos_id())
                    _encoder_input.insert(0, tokenizer.bos_id())
                    item["instr_encoding"] = _encoder_input
                    data.append(item)
            else:
                instr = item["instructions"]
                if self.sample_bpe:
                    _encoder_input = tokenizer.encode(instr, enable_sampling=True, alpha=0.3, nbest_size=-1)
                else:
                    _encoder_input = tokenizer.encode(instr)
                _encoder_input.append(tokenizer.eos_id())
                _encoder_input.insert(0, tokenizer.bos_id())
                item["instr_encoding"] = _encoder_input
                data.append(item)

        # print("******************************************\n")
        # print ("data length is :\n ", len(data)) #! data length is 265
        return data

    def _load_nav_graph(self):
        for scan_id in self.env.batch_scans:
            graph_tmp = load_nav_graph(self.opts, scan_id) #! loads only necessary graphs
            self.graph[scan_id] = graph_tmp
        print("Loading navigation graphs done.")

    #! Train.json doesn't have all scans 
    def _next_minibatch(self):
        batch = []
        for scan_id in self.env.batch_scans: #! scans may repeat
            # print ("len(self.env.batch_scans) is :\n ", len(self.env.batch_scans)) #! 64
            items_for_scan = []
            for item in self.data:
                if item["scan"] == scan_id:
                    # print ("TROUVE\n")
                    # print ("item scan:\n ", item["scan"])
                    items_for_scan.append(item)
                    random.shuffle(items_for_scan)
            batch.append(items_for_scan[0])
            self.minibatch_scans.append(items_for_scan[0]["scan"])
            # print ("items for items_for_scan[0] :\n ", items_for_scan[0]['scan'])
                # else :
                #     print ("PAS TROUVE \n")
            # print ("scan id:\n ", scan_id)
            # print ("self.data:\n ", self.data)
            # items_for_scan = [item for item in self.data if item["scan"] == scan_id]
            # print ("items for scan:\n ", items_for_scan)
            # random.shuffle(items_for_scan)
            # batch.append(items_for_scan[0])
        # print ("len(batch) :\n ", len(batch))
        
        # print ("Scans of the mini batch :\n ", batch)
        assert len(batch) == self.batch_size
        assert self.env.batch_scans== self.minibatch_scans

        # print ("Both navs-scans and minibatch_scans are equal\n ")
        # batch = self.data[self.ix:self.ix+self.batch_size]
        # batch = self.data[0:self.batch_size] not working either
        # self.ix += self.batch_size
        self.batch = batch
        self.minibatch_scans = []
        # print ("batch size:\n ", len(self.batch))
        # print ("batch :\n ", self.batch)
        
    def get_imgs(self):
        if self.opts.config.use_image_features:
            return self.env._get_imgs(len(self.batch))
        else:
            return None

    def get_junction_type(self):
        return self.env._get_junction_type(len(self.batch))

    def reset(self):
        self._next_minibatch() # we have all the scans of the batch
        pano_ids = []
        headings = []
        trajs = []
        scans = []
        for item in self.batch:
            scans.append(item["scan"])
            pano_ids.append(item["path"][0])
            headings.append(int(item["heading"]))
            trajs.append([pano_ids[-1]])
        # scans.sort()
        # print ("mini batch scans:\n ", scans)
        # self.env.batch_scans.sort()
        # print ("full batch scans:\n ", self.env.batch_scans)
        self.env.new_episodes(pano_ids, headings)
        return trajs  # returned a batch of the first panoid for each route_panoids
        
    def get_gt_action(self):
        return self.env._get_gt_action(self.batch)

    def action_select(self, a_t, ended, num_act_nav, trajs, total_steps):
        return self.env._action_select(a_t, ended, num_act_nav, trajs, total_steps, self.batch)

    #! we get data that is not in the batch
    #! then we get instructions to scans we don't have in our batch either
    def reset_epoch(self):
        self.ix = 0
        if self.sample_bpe or self.data is None:
            self.data = self._get_data()
        random.shuffle(self.data)
            
    def eva_metrics(self, trajs, metrics):
        # print("Evaluating metrics...")
        # print ("self.graph.keys is :\n ", self.graph.keys())
        # for graph_key in self.graph.keys():
        #     print ("graph's id  is :\n ", graph_key)
        #     print("graph value is \n", self.graph[graph_key])
        self.env._eva_metrics(trajs, self.batch, self.graph, metrics)
