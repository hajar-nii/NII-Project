import os

#! Hard coded the path to the R2R dataset
#! Code is working correctly
#todo: change the path to be adressed in the config file later
#todo: Not pressing as of now, since the code is working correctly

dataset_dir = 'datasets/r2r'
# Function is working correctly
def get_scans():
    with open(str(dataset_dir)+ '/scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        # print ("scans :", scans)
        # print(len(scans))
        nodes_list = {}
        links_list = {}
        for scan in scans:
            nodes_list.update({scan : dataset_dir + '/graph/nodes_orar/nodes_' + scan + '.txt' })
            links_list.update({scan : dataset_dir + '/graph/links_orar/links_' + scan + '.txt'})
        # print ("nodes_list :", nodes_list)
        # print ("links_list :", links_list)
    return scans, nodes_list, links_list

def get_scan_index(scan):
    scans, _, _ = get_scans()
    return scans.index(scan)

class Node:
    def __init__(self, panoid, pano_yaw_angle, lat, lng):
        self.panoid = panoid
        self.pano_yaw_angle = pano_yaw_angle
        self.neighbors = {}
        self.coordinate = (lat, lng)


class Graph:
    def __init__(self):
        self.nodes = {}
        
    def add_node(self, panoid, pano_yaw_angle, lat, lng):
        self.nodes[panoid] = Node(panoid, pano_yaw_angle, lat, lng)

    def add_edge(self, start_panoid, end_panoid, heading):
        start_node = self.nodes[start_panoid]
        end_node = self.nodes[end_panoid]
        start_node.neighbors[heading] = end_node

    # def add_edge(self, start_panoid, end_panoid, heading):
    #     start_node = self.nodes[start_panoid]
    #     end_node = self.nodes[end_panoid]

    #     if not heading in start_node.neighbors:
    #         start_node.neighbors[heading] = [end_node]
    #     else:
    #         start_node.neighbors[heading].append(end_node)
    #     # start_node.neighbors.setdefault(heading, []).append(end_node)

    # def get_node_neighbors(self, node):
    #     neighbors_nodes = sum (node.neighbors.values(), [])
    #     # neighbors_panoid = [neighbor.panoid for neighbor in neighbors_nodes]
    #     return neighbors_nodes

        

class GraphLoader:
    def __init__(self, dataset_dir):
        self.graph_list = []
        self.scans_list, self.node_file_list, self.link_file_list  = get_scans()
        # self.graph = Graph()
        # self.node_file = os.path.join(dataset_dir, 'nodes.txt')
        # self.link_file = os.path.join(dataset_dir, 'links.txt')

    def construct_graphs(self):
        
        for scan in self.scans_list:
            temp_graph = Graph()
            with open(self.node_file_list[scan]) as f:
                for line in f:
                    panoid, pano_yaw_angle, lat, lng = line.strip().split(',')
                    temp_graph.add_node(panoid, int(pano_yaw_angle), float(lat), float(lng))

            with open(self.link_file_list[scan]) as f:
                for line in f:
                    start_panoid, heading, end_panoid = line.strip().split(',')
                    temp_graph.add_edge(start_panoid, end_panoid, float(heading))

            num_edges = 0
            for panoid in temp_graph.nodes.keys():
                num_edges += len(temp_graph.nodes[panoid].neighbors)
            
            # print ("Graph constructed is :", temp_graph)
            self.graph_list.append(temp_graph)

        print('graphs constructed')
        # print('Graph list is :', self.graph_list)
        print ("Length of graph list is :", len(self.graph_list)) #prints 90

        return self.graph_list
    
    #TODO : Rewrite the Construct graphs as to use construct_single_graph
    def construct_single_graph(self, scan_id):
        graph = Graph()
        with open(self.node_file_list[scan_id]) as f:
            for line in f:
                panoid, pano_yaw_angle, lat, lng = line.strip().split(',')
                graph.add_node(panoid, int(pano_yaw_angle), float(lat), float(lng))

        with open(self.link_file_list[scan_id]) as f:
            for line in f:
                start_panoid, heading, end_panoid = line.strip().split(',')
                graph.add_edge(start_panoid, end_panoid, float(heading))

        num_edges = 0
        for panoid in graph.nodes.keys():
            num_edges += len(graph.nodes[panoid].neighbors)
        return graph



# if __name__ == '__main__':
    # get_scans()
    # graph_loader = GraphLoader(dataset_dir)
    # # graph_loader.construct_graphs()
    # # print ("Graph index of b8cTxDM8gDG  is :", get_scan_index('b8cTxDM8gDG'))
    # # print ("Graph index of X7HyMhZNoso  is :", get_scan_index('X7HyMhZNoso'))
    # # print ("Graph index of zsNo4HB9uLZ  is :", get_scan_index('zsNo4HB9uLZ'))

   
    # graph  = graph_loader.construct_single_graph('JmbYfDe2QKZ')
    # pano_id = "63333423642f49caac4871521e93cc45"
    # pano_neigbors  = graph.nodes[pano_id].neighbors
    
    # # sum = sum (pano_neigbors.values(), [])

    # # neighbors_id = graph.get_node_neighbors_panoid(graph.nodes[pano_id])
    # # neighbors_id = [neighbor.panoid for neighbor in sum]
    # # print ("pano_neigbors :", pano_neigbors)
    # # print ("sum of them is :", sum)
    # print ("neighbors_id :", neighbors_id)

