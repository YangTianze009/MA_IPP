import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

from classes.Graph import *

class RRTGraph:
    def __init__(self, step_size):
        self.step_size = step_size

    def create_graph(self, latest_coords):
        self.nu_node_coords = latest_coords
        self.graph = Graph()
        Radius = self.step_size*1.1
        X = latest_coords
        radius_nn = spatial.cKDTree(X)

        for centre in X:
            from_node = self.findNodeIndex(centre)
            self.graph.add_node(str(from_node))
            # index of all points within radius distance
            to_nodes = radius_nn.query_ball_point(centre, Radius)
            for each_node in to_nodes:
                distance = np.linalg.norm(centre - latest_coords[each_node])
                self.graph.add_node(str(each_node))
                self.graph.add_edge(str(from_node), str(each_node), distance)
        
#        self.visualize_graph(self.graph, path = None, j=0, type='visual', node_coords=latest_coords)
        return self.graph

    def findNodeIndex(self, p):
        return np.where((self.nu_node_coords == p).all(axis=1))[0][0]

    def visualize_graph(self, graph, path, j, gp = None, true_distrib = None, type = 'ENV', node_coords = None, budget = 0, cov_trace = 999, time=0):
        self.gp_ipp = gp
        if type == 'visual':
            plt.figure(2)
        else:
            plt.figure(3)
        if type == 'ENV':
            colour = 'grey'
            node_color = 'blue'
        elif type == 'Tree':
            colour = 'black'
            node_color = 'pink'
        else:
            colour = 'grey'
            node_color = 'black'
        x_vals = node_coords[:,0]
        y_vals = node_coords[:,1]

        # Plot info distribution
        if type != 'visual':
            self.gp_ipp.plot(true_distrib)


        edge_dict = graph.edges

        if type:# != 'ENV':
            for each_node in graph.nodes:
                connected_edges = edge_dict[str(each_node)]
                for node, edge in connected_edges.items():

                    if node != each_node:
                        x_e = [x_vals[int(each_node)], x_vals[int(node)]]
                        y_e = [y_vals[int(each_node)], y_vals[int(node)]]
                        if type != 'visual':
                            plt.subplot(1, 3, 1) # mean
                        plt.plot(x_e, y_e, color='black')

        # Plot the nodes
        if type != 'visual':
            plt.subplot(1, 3, 1) # mean
        plt.scatter(x_vals[1:], y_vals[1:], color = node_color) # All sampled nodes, in blue
        plt.scatter(x_vals[0], y_vals[0], color = 'orange') # Start node, in orange


        plt.suptitle('Budget: {:.4g}/{:.4g},  cov_trace: {:.4g},  time: {:.4g}'.format(budget, 8.0, cov_trace, time))
        if type != 'visual':
            plt.savefig(path + type + '_' + 'RRT_res_' + str(j) + '.png')
        else:
            plt.show()