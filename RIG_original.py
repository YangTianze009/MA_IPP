'''
Original planner for RIG_tree
This program generates an asymptotically optimal informative path planning, rapidly exploring random tree RRT*

ADAPTED FROM - https://www.linkedin.com/pulse/motion-planning-algorithm-rrt-star-python-code-md-mahbubur-rahman/
'''

import os
import numpy as np
from math import atan2, cos, sin
import matplotlib.pyplot as plt

from classes.Gaussian2D import *
from gp_ipp import GaussianProcessForIPP


class Node:
    def __init__(self, xcoord, ycoord):
        self.x = xcoord
        self.y = ycoord
        self.cost = 0.0
        self.info = 0.0
        self.std = 1.0
        self.parent = None


class RRT:
    def __init__(self, num_nodes=50, XDIM=1.0, YDIM=1.0, radius=0.5, sample_size=0.05, gp_func=None,
                 gaussian_distrib=None):
        self.path = f'RRT_results/RRT_star_trees/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.num_nodes = num_nodes
        self.XDIM = XDIM
        self.YDIM = YDIM
        self.radius = radius  # To look for parent & rewiring
        self.sample_size = sample_size  # distance of newly sampled node from tree
        self.node_coords = np.array([])
        self.gp_ipp = gp_func  # GaussianProcessForIPP(self.node_coords)
        self.underlying_distribution = gaussian_distrib
        self.ground_truth = self.get_ground_truth()

    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth

    # Utilities
    def distance(self, node1, node2):  # Pass coordinates
        return np.linalg.norm(np.array(node1) - np.array(node2))

    def step_from_to(self, from_node, to_node):  # Pass coordinates
        if self.distance(from_node, to_node) < self.sample_size:
            return to_node
        else:
            theta = atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
            return from_node[0] + self.sample_size * cos(theta), from_node[1] + self.sample_size * sin(theta)

    def chooseParent(self, nn, newnode):
        for p in self.nodes:
            if self.distance([p.x, p.y], [newnode.x, newnode.y]) < self.radius and p.cost + self.distance([p.x, p.y],
                                                                                                          [newnode.x,
                                                                                                           newnode.y]) < \
                    nn.cost + self.distance([nn.x, nn.y], [newnode.x, newnode.y]):
                nn = p
        newnode.cost = nn.cost + self.distance([nn.x, nn.y], [newnode.x, newnode.y])
        newnode.parent = nn
        return newnode, nn

    def reWire(self, newnode):
        for i in range(len(self.nodes)):
            p = self.nodes[i]
            if p != newnode.parent and self.distance([p.x, p.y], [newnode.x,
                                                                  newnode.y]) < self.radius and newnode.cost + self.distance(
                [p.x, p.y], [newnode.x, newnode.y]) < p.cost:
                # Show old lines here ->
                p.parent = newnode
                p.cost = newnode.cost + self.distance([p.x, p.y], [newnode.x, newnode.y])
                self.nodes[i] = p
                # Show new lines here ->

    def findNodeIndex(self, p):
        return np.where((self.nodes == p).all(axis=1))[0][0]

    def prune(self, newnode):
        if newnode.cost > 8.0:
            return True
        for p in self.nodes:
            if p.std < newnode.std and p.cost < newnode.cost:
                return True
        return False

    def draw_stuff(self):  # , num, start_node):
        x_vals = []
        y_vals = []
        for each_node in self.nodes:
            x_vals.append(each_node.x)
            y_vals.append(each_node.y)

        plt.figure(1)
        plt.scatter(x_vals[1:], y_vals[1:], color='blue')  # All sampled nodes, in blue
        plt.scatter(x_vals[0], y_vals[0], color='orange')  # Start node, in orange
        plt.show()

    def RRT_planner(self, start_node, iterations=500, info=None):
        counts = 0
        self.nodes = []
        start = Node(start_node[0], start_node[1])
        self.nodes.append(start)

        node_C = np.array([[start.x, start.y]])
        start.info = self.underlying_distribution.distribution_function(node_C.reshape(-1, 2)) + np.random.normal(0,
                                                                                                                  1e-10)

        goal = Node(1.0, 1.0)  # Destination

        while counts < iterations:
            for i in range(self.num_nodes):
                rand = Node(np.random.rand() * self.XDIM, np.random.rand() * self.YDIM)
                nn = self.nodes[0]
                for p in self.nodes:
                    if self.distance([p.x, p.y], [rand.x, rand.y]) < self.distance([nn.x, nn.y], [rand.x, rand.y]):
                        nn = p
            interpolatedNode = self.step_from_to([nn.x, nn.y], [rand.x, rand.y])
            newnode = Node(interpolatedNode[0], interpolatedNode[1])
            node_C = np.array([[newnode.x, newnode.y]])
            newnode.info, newnode.std = self.gp_ipp.flexi_updates(node_C)

            if not self.prune(newnode):
                [newnode, nn] = self.chooseParent(nn, newnode)
                self.nodes.append(newnode)
                self.reWire(newnode)

            counts += 1
        #            if counts == iterations:
        #                print('Tree constructed')
        #                self.draw_stuff()
        return self.nodes


if __name__ == '__main__':
    rrt_tree = RRT()
    nodes = rrt_tree.RRT_planner(iterations=100)  # 500 iterations
    print(len(nodes))
