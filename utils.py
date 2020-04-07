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

def barabasiAlbert(numNodes, avgDegree,rndSeed=None):
    G = nx.barabasi_albert_graph(numNodes, avgDegree, seed=rndSeed)
    return G

def randomizedNeighborhoods(layer1, fractionNodes, numNodes,rndSeed=None):
    G = layer1
    nx.double_edge_swap(G, nswap=fractionNodes*numNodes, max_tries=numNodes*numNodes, seed=rndSeed)

    return G

def totalRandomization(layer1, numNodes):
    G = layer1
    print(G.nodes())
    nodes = list(G)
    random.shuffle(nodes)
    mapping={}
    for n in range(numNodes):
        mapping.update({n: nodes[n]})
    G=nx.relabel_nodes(G,mapping)
    return G


def getNeighborPairs(G, nodeInfo, pos):
    # The index of each node in nodeInfo corresponds to the node with the same index in G.nodes
    # For each node, get all of its neighbors
    pairs = []
    done = [] # to make sure nodes each link A-B or B-A is only used once
    for it, n in enumerate(nodeInfo):
        neighbors = G.neighbors(n['pos'])
        for neighbor in neighbors:
            neighborIt = pos.index(neighbor)
            if neighborIt not in done:
                pairs.append([n, nodeInfo[neighborIt]])
        done.append(n['pos'])

    random.shuffle(pairs)
    return pairs

def getNeighborsAsynchronous(G, node, neighbor, arr, nodeInfo, pos):
    # The index of each node in nodeInfo corresponds to the node with the same index in G.nodes
    # For each node, get all of its neighbors
    pairs = []
    neighborsA = G.neighbors(node['pos'])
    for n in neighborsA:
        neighborIt = pos.index(n)
        if ([node['pos'],nodeInfo[neighborIt]['pos']] not in arr) and ([nodeInfo[neighborIt]['pos'],node['pos']] not in arr):
            pairs.append([node, nodeInfo[neighborIt]])
    neighborsB = G.neighbors(neighbor['pos'])
    for n in neighborsB:
        neighborIt = pos.index(n)
        if ([neighbor['pos'], nodeInfo[neighborIt]['pos']] not in arr) and ([nodeInfo[neighborIt]['pos'], neighbor['pos']] not in arr):
            pairs.append([neighbor, nodeInfo[neighborIt]])

    random.shuffle(pairs)
    return pairs

def getRecipientReputation(donor, recipient):
    return donor['perception'][recipient['id']]['reputation']

def pickNeighbor(G, donor, nodeInfo, pos):

    # Choose one of the donor's neighbors who will witness and gossip about the interaction
    neighbors = G.neighbors(donor['pos'])
    arr = []
    for neighbor in neighbors:
        neighborIt = pos.index(neighbor)
        arr.append(nodeInfo[neighborIt])

    chosen = random.choice(arr)

    return chosen

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
    #print(mapping)
    nodes = G.nodes()
    colors = [mapping[str(n['strategy'])] for n in nodeInfo]
    #print(colors)
    # Drawing nodes and edges separately in order to capture collection for colorbar
    #pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, with_labels=False, node_size=50, cmap = plt.jet())
    cbar = plt.colorbar(nc, ticks=[0, 1, 2, 3], orientation='vertical')
    cbar.ax.set_yticklabels(['AllC', 'AllD', 'Disc', 'pDisc']) # Alphabetic order
    # fixme - when there is only one strategy remaining, it defaults to green
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
    return (random.random() < percentage)

def calculateAverage(arr,variable, total):
    sum = 0
    for item in arr:
        sum += item[variable]

    return sum/total
