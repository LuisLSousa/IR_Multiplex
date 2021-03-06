import networkx as nx
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os.path
from os.path import join
from itertools import count
import copy
import time
import re
from collections import Counter

ringAPL = 0  # Global variable used to normalize the average path length for different values of "p-watts-strogatz"
graph = None  # Used to ensure the graph remains the same when testing several social norms with the same initial state
graph2 = None
AllG = []
SJ = []
SS = []
SH = []
IS = []
CC = []
APL = []
x_axis = [] # The variable that will be used in the plot. Can be pWattsStrogatz, avgDegree or explorationRate

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


def completeGraph(numNodes):
    G = nx.complete_graph(numNodes)
    return G


def randomizedNeighborhoods(layer1, numSwap, numNodes, rndSeed=None):
    G = copy.deepcopy(layer1)
    Gf = nx.double_edge_swap(G, nswap=numSwap, max_tries=numNodes * numNodes, seed=rndSeed)
    return Gf


def totalRandomization(layer1, numNodes):
    G = copy.deepcopy(layer1)
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
        neighbors = G[n['pos']] # this used to be G.neighbors(n['pos'})
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
    done = []  # to make sure each link A-B or B-A is only used once
    gamesPlayed = [0, numInteractions] # Games played by the node and the chosen neighbor respectively
    # gamesPlayed[1] starts at numInteractions because the node and the neighbor interact numInteractions times on the node's turn
    g = 0
    for node in chosen_nodes: # Each node has its turn
        neighbors = G[node['pos']] # this used to be G.neighbors()
        for n in neighbors:
            neighborIt = pos.index(n)
            if neighborIt not in done: # make sure nodes don't play more than numInteractions times
                for i in range(numInteractions):
                    # 50/50 chance of playing as a donor or a recipient
                    if probability(0.5):
                        pairs.append([node, nodeInfo[neighborIt]])
                    else:
                        pairs.append([nodeInfo[neighborIt], node])
                    gamesPlayed[g] += 1
        g += 1
        done.append(node['pos'])

    # random.shuffle(pairs)  # This may be unnecessary

    return pairs, gamesPlayed

def pickNeighbor(G, node, nodeInfo, pos):
    # Pick a neighbor
    # neighbors = G.neighbors(node['pos']) # this method is slower than the method below
    neighbors = G[node['pos']]
    arr = []
    for neighbor in neighbors:
        arr.append(nodeInfo[pos.index(neighbor)])
    if arr:
        chosen = random.choice(arr)
        return chosen
    else:
        return None


'''def drawGraph(G, nodeInfo, directory, it):
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
    plt.close()'''


def plotValues(coopRatio, socialNorm):
    global AllG
    global SJ
    global SS
    global IS
    global SH
    # Is the above code needed?

    if socialNorm == 'AllGood':
        AllG.append(coopRatio)
    elif socialNorm == 'SternJudging':
        SJ.append(coopRatio)
    elif socialNorm == 'Shunning':
        SH.append(coopRatio)
    elif socialNorm == 'SimpleStanding':
        SS.append(coopRatio)
    elif socialNorm == 'ImageScoring':
        IS.append(coopRatio)
    else:
        print("Error, wrong social norm name!")
        exit()

def runLogs(AllG, SJ, SH, IS, SS, CC, APL, x_axis, x_label, filename):
    if SJ:
        plt.plot(x_axis, SJ, '^-r', label='SJ')
    if SH:
        plt.plot(x_axis, SH, '-py', label='SH')
    if AllG:
        plt.plot(x_axis, AllG, '-sb', label='AllG')
    if SS:
        plt.plot(x_axis, SS, '<-g', label='SS')
    if IS:
        plt.plot(x_axis, IS, '-or', label='IS')
    if CC:
        plt.plot(x_axis, CC, '-Hg', label='CC')
    if APL:
        plt.plot(x_axis, APL, '-D', label='APL')

    if x_label == 'avgDegree1' or x_label == 'avgDegree2':
        plt.xscale('linear')
    else:
        plt.xscale('symlog', linthreshx=0.00001) # Use a linthreshx equal to the lowest probability after 0
    #plt.xscale('log') # linear, log, symlog
    plt.xlabel(x_label)
    plt.ylabel("Cooperation Ratio")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def barPlot(coopRatio, filename):
    norms = ['SJ', 'SS', 'IS', 'SH']
    for i in range(len(coopRatio)):
        plt.text(norms[i], coopRatio[i], "{:.3f}".format(coopRatio[i]))

    plt.bar(norms, coopRatio, color=['#0000fa', '#db1a1a', '#fe7e07', '#199116'])
    plt.ylabel("Coop Ratio")
    # plt.ticks(norms, coopRatio)
    plt.savefig(filename)
    plt.show()


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


def stationaryFraction(nodes, perceptions):
    good = 0
    bad = 0
    repAllC = [0, 0]  # [no. of good, no. of bad] reputations with strategy AllC
    repAllD = [0, 0]
    repDisc = [0, 0]
    repPDisc = [0, 0]
    # res = dict(sum(map(Counter, perceptions), Counter())) # Count number of good and bad reputations (faster).

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if perceptions[i][j] == 'Good':
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
                    print("Wrong strategy!")
                    exit()
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
                else:
                    print("Wrong strategy!")
                    exit()

    statGood = good / (good + bad)
    statBad = bad / (good + bad)
    numRep = [repAllC, repAllD, repDisc, repPDisc]
    return [statGood, statBad], numRep


def probability(chance):
    return random.random() < chance


def calculateAverage(arr, variable):
    sum = 0
    for item in arr:
        sum += item[variable]

    return sum / len(arr)

def writeFile(coopBar, initialValues, filename):

    f = open('pltOutput.txt', "a")
    f.write("\n######################################")
    if AllG:
        f.write("\nAllG = {}".format(AllG))
    f.write("\nSJ = {}".format(SJ))
    f.write("\nSH = {}".format(SH))
    f.write("\nIS = {}".format(IS))
    f.write("\nSS = {}".format(SS))
    if APL:
        f.write("\nAPL = {}".format(APL))
    if CC:
        f.write("\nCC = {}".format(CC))
    if coopBar:
        f.write("\ncoopRatio = {}".format(coopBar))
        print('Coop Bar =  {}'.format(coopBar))

    f.write("\nx_axis = {}".format(x_axis))
    f.write("\ntypeOfSimulation = \'{}\'".format(initialValues['typeOfSimulation']))
    f.write("\nfilename = \'{}\' ".format(filename))
    f.close()