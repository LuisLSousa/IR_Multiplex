import networkx as nx
import math
import random
import matplotlib.pyplot as plt
from os.path import join
from itertools import count


class MultiplexNetwork(nx.Graph):
    # Generate a random graph with a given average degree and size.
    def __init__(self, numNodes, avgDegree):
        super().__init__()
        self.numNodes = numNodes
        self.avgDegree = avgDegree
        self.add_nodes_from(range(self.numNodes))
        while self.avg_deg() < self.avgDegree:
            n1, n2 = random.sample(self.nodes(), 2)
            self.add_edge(n1, n2, weight=1)

    def avg_deg(self):
        return self.number_of_edges() * 2 / self.numNodes

def wattsStrogatz(numNodes, avgDegree, prob, rndSeed=None):
    G = nx.watts_strogatz_graph(numNodes,avgDegree, prob, seed=rndSeed)
    return G

def getNeighborPairs(G, nodeInfo, pos):
    # The index of each node in nodeInfo corresponds to the node with the same index in G.nodes
    # For each node, get all of its neighbors
    pairs = []
    for it, n in enumerate(nodeInfo):
        neighbors = G.neighbors(n['pos'])
        for neighbor in neighbors:
            neighborIt = pos.index(neighbor)
            pairs.append([n, nodeInfo[neighborIt]])
    random.shuffle(pairs)
    return pairs

def getRecipientReputation(donor, recipient):
    return donor['perception'][recipient['id']]['reputation']

def getGossiper(G, donor, nodeInfo, pos):

    # Choose one of the donor's neighbors who will witness and gossip about the interaction
    neighbors = G.neighbors(donor['pos'])
    snitch = []
    for neighbor in neighbors:
        neighborIt = pos.index(neighbor)
        snitch.append(nodeInfo[neighborIt])

    gossiper = random.choice(snitch)

    return gossiper

def updatePerception(socialNorm, witness, donor, recipient, action):
    # print('Before: ')
    # print('Norm: ', socialNorm,'Donor:',witness['perception'][donor['pos']]['reputation'],'Action: ', action)

    # Stern Judging (Coop with Good = G; Defect with Bad = G; else B)
    if socialNorm == 'SternJudging':
        if action == 'Cooperate' and witness['perception'][recipient['pos']]['reputation'] == 'Good':
            witness['perception'][donor['pos']]['reputation'] = 'Good'
        elif action == 'Defect' and witness['perception'][recipient['pos']]['reputation'] == 'Bad':
            witness['perception'][donor['pos']]['reputation'] = 'Good'
        else:
            witness['perception'][donor['pos']]['reputation'] = 'Bad'


    # Simple Standing (Defect with Good = B; else G)
    elif socialNorm == 'SimpleStanding':
        if action == 'Defect' and witness['perception'][recipient['pos']]['reputation'] == 'Good':
            witness['perception'][donor['pos']]['reputation'] = 'Bad'
        else:
            witness['perception'][donor['pos']]['reputation'] = 'Good'

    # Shunning (Coop with Good = G; else B)
    elif socialNorm == 'Shunning':
        if action == 'Cooperate' and witness['perception'][recipient['pos']]['reputation'] == 'Good':
            witness['perception'][donor['pos']]['reputation'] = 'Good'
        else:
            witness['perception'][donor['pos']]['reputation'] = 'Bad'

    # Image Scoring (Coop = Good; Defect = Bad)
    elif socialNorm == 'ImageScoring':
        if action == 'Cooperate':
            witness['perception'][donor['pos']]['reputation'] = 'Good'
        elif action == 'Defect':
            witness['perception'][donor['pos']]['reputation'] = 'Bad'
        else:
            print('Error: Action is neither "Cooperate" nor "Defect" ' )
            exit()

    else:
        print('Wrong socialNorm, check initial parameters')
        exit()

    # print('After: ')
    # print('Donor:', witness['perception'][donor['pos']]['reputation'])

    # witness['perception'][donor['pos']]['reputation'] =


def drawGraph(G, nodeInfo, dir, it):

    # Group nodes according to their strategy
    arr = ['AllC', 'AllD', 'Disc', 'pDisc']
    groups = set(arr)
    mapping = dict(zip(sorted(groups), count()))
    nodes = G.nodes()
    colors = [mapping[str(n['strategy'])] for n in nodeInfo]

    # Drawing nodes and edges separately so we can capture collection for colorbar
    #pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, with_labels=False, node_size=50, cmap = plt.jet())
    cbar = plt.colorbar(nc, ticks=[0, 1, 2, 3], orientation='vertical')
    cbar.ax.set_yticklabels(['AllC', 'AllD', 'Disc', 'pDisc'])
    # print(mapping) # Check if colorbar label is correct
    plt.axis('off')
    plt.savefig(join(dir, 'graph{}.png'.format(it)))
    plt.close()

def countFreq(arr):
    mp = dict()

    # Traverse through array elements and count frequencies
    for i in range(len(arr)):
        if arr[i] in mp.keys():
            mp[arr[i]] += 1
        else:
            mp[arr[i]] = 1

    # Normalize
    for k in mp.keys():
        mp[k] = mp[k]/len(arr)
    return mp

def probability(percentage):
    return (random.random() < percentage);
