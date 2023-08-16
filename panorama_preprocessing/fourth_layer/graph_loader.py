import os
dataset_dir = 'dataset'
# Function is working correctly
def get_scans():
    with open('dataset/scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        # print ("scans :", scans)
        # print(len(scans))
        nodes_list = {}
        links_list = {}
        for scan in scans:
            nodes_list.update({scan : dataset_dir + '/nodes_orar/nodes_' + scan + '.txt' })
            links_list.update({scan : dataset_dir + '/links_orar/links_' + scan + '.txt'})
        # print ("nodes_list :", nodes_list)
        # print ("links_list :", links_list)
    return scans, nodes_list, links_list

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
            
            print ("Graph constructed is :", temp_graph)
            self.graph_list.append(temp_graph)

        print('graphs constructed')
        # print('Graph list is :', self.graph_list)
        # print ("Length of graph list is :", len(self.graph_list)) #prints 90

        return self.graph_list



if __name__ == '__main__':
    # get_scans()
    graph_loader = GraphLoader(dataset_dir)
    graph_list = graph_loader.construct_graphs()
    for item in graph_list[33].nodes.values():
        for key, value in item.neighbors.items():
            print ("key is :", key)
            print ("value is :", value)
        # print ("item is \n:", item)