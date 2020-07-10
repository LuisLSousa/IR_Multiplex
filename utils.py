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
    G = nx.watts_strogatz_graph(numNodes, avgDegree, prob, seed=rndSeed)
    return G


def barabasiAlbert(numNodes, avgDegree, rndSeed=None):
    G = nx.barabasi_albert_graph(numNodes, avgDegree, seed=rndSeed)
    return G


def randomizedNeighborhoods(layer1, fractionNodes, numNodes, rndSeed=None):
    G = layer1
    nx.double_edge_swap(G, nswap=fractionNodes * numNodes, max_tries=numNodes * numNodes, seed=rndSeed)

    return G


def totalRandomization(layer1, numNodes):
    G = layer1
    nodes = list(G)
    random.shuffle(nodes)
    mapping = {}
    for n in range(numNodes):
        mapping.update({n: nodes[n]})
    G = nx.relabel_nodes(G, mapping)
    return G


def calculateInitialReputation():  # Randomly attribute a good or bad reputation
    if random.choice([0, 1]) == 0:
        initialReputation = 'Good'
    else:
        initialReputation = 'Bad'

    return initialReputation


def calculateInitialStrategy():  # Randomly attribute an initial strategy
    rand = random.randrange(4)
    if rand == 0:
        initialStrategy = 'AllC'
    elif rand == 1:
        initialStrategy = 'AllD'
    elif rand == 2:
        initialStrategy = 'Disc'
    else:
        initialStrategy = 'pDisc'
    return initialStrategy


def getNeighborPairs(G, nodeInfo, pos, numInteractions):
    # The index of each node in nodeInfo corresponds to the node with the same index in G.nodes
    # For each node, get all of its neighbors
    pairs = []
    done = []  # to make sure each link A-B or B-A is only used once
    for it, n in enumerate(nodeInfo):
        neighbors = G.neighbors(n['pos'])
        for neighbor in neighbors:
            neighborIt = pos.index(neighbor)
            if neighborIt not in done:
                for i in range(numInteractions):  # Number of times the node plays with each neighbor
                    # 50/50 chance of playing as a donor or a recipient
                    if probability(0.5):
                        pairs.append([n, nodeInfo[neighborIt]])
                    else:
                        pairs.append([nodeInfo[neighborIt], n])

        done.append(n['pos'])

    random.shuffle(pairs)

    return pairs


def getNeighborsAsynchronous(G, chosen_nodes, nodeInfo, pos, numInteractions):
    # The index of each node in nodeInfo corresponds to the node with the same index in G.nodes
    # Get all neighbors of both nodes (donor and recipient)
    pairs = []
    for node in chosen_nodes:
        neighbors = G.neighbors(node['pos'])
        for n in neighbors:
            neighborIt = pos.index(n)
            for i in range(numInteractions):
                # 50/50 chance of playing as a donor or a recipient
                if probability(0.5):
                    pairs.append([node, nodeInfo[neighborIt]])
                else:
                    pairs.append([nodeInfo[neighborIt], node])

    random.shuffle(pairs)  # fixme - This may be unnecessary

    return pairs


def getRecipientReputation(donor, recipient):
    return donor['perception'][recipient['id']]['reputation']


def pickNeighbor(G, node, nodeInfo, pos):
    # Pick a neighbor
    neighbors = G.neighbors(node['pos'])
    arr = []
    for neighbor in neighbors:
        neighborIt = pos.index(neighbor)
        arr.append(nodeInfo[neighborIt])
    if arr:
        chosen = random.choice(arr)
        return chosen
    else:
        return None


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
            print('Error: Action is neither "Cooperate" nor "Defect" ')
            exit()

    else:
        print('Wrong socialNorm, check initial parameters')
        exit()

    # print('After: ')
    # print('Donor:', witness['perception'][donor['pos']]['reputation'])

    # witness['perception'][donor['pos']]['reputation'] =


def drawGraph(G, nodeInfo, directory, it):
    # Group nodes according to their strategy
    arr = ['AllC', 'AllD', 'Disc', 'pDisc']
    groups = set(arr)
    mapping = dict(zip(sorted(groups), count()))
    # print(mapping)
    nodes = G.nodes()
    colors = [mapping[str(n['strategy'])] for n in nodeInfo]
    # print(colors)
    # Drawing nodes and edges separately in order to capture collection for colorbar
    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, with_labels=False, node_size=50,
                                cmap=plt.jet())
    cbar = plt.colorbar(nc, ticks=[0, 1, 2, 3], orientation='vertical')
    cbar.ax.set_yticklabels(['AllC', 'AllD', 'Disc', 'pDisc'])  # Alphabetic order
    # fixme - when there is only one strategy remaining, it defaults to green
    plt.axis('off')
    plt.savefig(join(directory, 'graph{}.png'.format(it)))
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
        mp[k] = mp[k] / len(arr)
    return mp


def stationaryFraction(nodes):
    good = 0
    bad = 0
    repAllC = [0, 0]  # [no. of good, no. of bad] reputations with strategy AllC
    repAllD = [0, 0]
    repDisc = [0, 0]
    repPDisc = [0, 0]
    for i in nodes:
        for j in range(len(nodes)):
            if i['perception'][j]['reputation'] == 'Good':
                good += 1
                if nodes[j]['strategy'] == 'AllC':
                    repAllC[0] += 1
                elif nodes[j]['strategy'] == 'AllD':
                    repAllD[0] += 1
                elif nodes[j]['strategy'] == 'Disc':
                    repDisc[0] += 1
                elif nodes[j]['strategy'] == 'pDisc':
                    repPDisc[0] += 1
            else:
                bad += 1
                if nodes[j]['strategy'] == 'AllC':
                    repAllC[1] += 1
                elif nodes[j]['strategy'] == 'AllD':
                    repAllD[1] += 1
                elif nodes[j]['strategy'] == 'Disc':
                    repDisc[1] += 1
                elif nodes[j]['strategy'] == 'pDisc':
                    repPDisc[1] += 1

    statGood = good / (good + bad)
    statBad = bad / (good + bad)
    numRep = [repAllC, repAllD, repDisc, repPDisc]

    return [statGood, statBad], numRep
    # print(self.nodes[3]['perception'][7]['reputation'])


def probability(chance):
    return random.random() <= chance


def calculateAverage(arr, variable, total):
    sum = 0
    for item in arr:
        sum += item[variable]

    return sum / total