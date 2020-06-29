import numpy as np
import random
import os
from os import mkdir
from os.path import join
from utils import *
import json


class IndirectReciprocityMultiplexNetworks:

    nodes = []
    def __init__(self, numNodes=100, prob1=0.5, prob2=0.5, avgDegree=2, numGenerations=100, numInteractions=1, logFreq=1, cost=0.1, benefit=1, transError=0.01,beta=10, update='Synchronous', explorationRate=0.01,rndSeed=None, gephiFileName='test.gexf', layer1=None, layer2=None, socialNorm = 'SternJudging', fractionNodes = 0.2):

        self.numNodes = numNodes # Number of nodes
        self.nodes = []
        self.prob1 = prob1 # Rewire probability for Watts-Strogatz - L1
        self.prob2 = prob2 # Rewire probability for Watts-Strogatz - L2
        self.avgDegree = avgDegree
        self.numGenerations = numGenerations
        self.numInteractions = numInteractions
        self.logFreq = logFreq # Generate graphs at every X simulations
        self.cost = cost # Donation Game
        self.benefit = benefit # Donation Game
        self.beta = beta
        self.rndSeed = rndSeed
        self.transError = transError
        self.explorationRate = explorationRate
        self.socialNorm = socialNorm # Global social norm (the entire population follows this)
        self.gephi = gephiFileName # File name for the gephi export
        self.layer1 = layer1 # Layer1 topology
        self.layer2 = layer2 # Layer2 topology
        self.update = update
        self.fractionNodes = fractionNodes

        if self.numInteractions == 0:
            print("numInteractions must be > 0")
            exit()

        self.idIterator = 0
        self.idToIndex = {}  # id:index
        self.initiateNodes()

    def initiateNodes(self):

        if self.layer1 == 'Random':
            self.layer1 = MultiplexNetwork(self.numNodes, self.avgDegree)

        elif self.layer1 == 'WattsStrogatz':
            self.layer1 = wattsStrogatz(self.numNodes, self.avgDegree, self.prob1, self.rndSeed)

        elif self.layer1 == 'BarabasiAlbert':
            self.layer1 = barabasiAlbert(self.numNodes, self.avgDegree, self.rndSeed)

        else:
            print('Wrong layer1 parameter!')
            exit()

        if self.layer2 == 'Random':
            self.layer2 = MultiplexNetwork(self.numNodes, self.avgDegree)

        elif self.layer2 == 'WattsStrogatz':
            self.layer2 = wattsStrogatz(self.numNodes, self.avgDegree, self.prob2, self.rndSeed)

        elif self.layer2 == 'BarabasiAlbert':
            self.layer2 = barabasiAlbert(self.numNodes, self.avgDegree, self.rndSeed)

        elif self.layer2 == 'PerfectOverlap':
            self.layer2 = self.layer1

        elif self.layer2 == 'RandomizedNeighborhoods':
            self.layer2 = randomizedNeighborhoods(self.layer1, self.fractionNodes, self.numNodes, self.rndSeed)

        elif self.layer2 == 'TotalRandomization':
            self.layer2 = totalRandomization(self.layer1, self.numNodes)

        else:
            print('Wrong layer2 parameter!')
            exit()

        self.nodePos = list(self.layer1.nodes())
        for i in range(self.numNodes):
            self.nodes.append({
                'pos': self.nodePos[i],  # Redundante?
                'id': self.idIterator,
                'payoff': 0,
                'strategy': self.calculateInitialStrategy(),
                'viz': None
            })

            self.idToIndex[self.idIterator] = len(self.nodes) - 1
            self.idIterator += 1

        # Check nodes
        # for i in range(self.numNodes):
        #	print('Node:', self.nodes[i]['id'], '| Strategy:', self.nodes[i]['strategy'])

        # Each node has its own perception of all others
        for node in self.nodes:
            node['perception'] = [{'reputation': self.calculateInitialReputation(), 'id': i['id']} for i in self.nodes]
            # Example: Node number 3's perception of node number 7
            # print(self.nodes[3]['perception'][7]['reputation'])

    def calculateInitialReputation(self): # Randomly attribute a good or bad reputation
        if random.choice([0,1]) == 0:
            initialReputation = 'Good'
        else:
            initialReputation = 'Bad'

        return initialReputation

    def calculateInitialStrategy(self): # Randomly attribute an initial strategy

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

    def runSimulation(self):
        print('=====    Initiating simulation   ======')
        LogsPerGen = []
        numRep = [] # Number of AllC, AllD, Disc, and pDisc with G and B reputation
        # Stationary fraction of 'Good' and 'Bad' reputations of each gen. statFrac[ [good,bad] , [good,bad], ... ]
        statFrac = []
        if self.update == 'Synchronous':
            for i in range(self.numGenerations):
                lg = self.runGeneration()
                lg['generation'] = i
                if i % self.logFreq == 0:
                    print('== Logging {} =='.format(i))
                    #drawGraph(self.layer1, self.nodes, dir, i)

                self.socialLearning()
                LogsPerGen.append(lg)
                stat, numRep = stationaryFraction(self.nodes)
                statFrac.append(stat)
                # Reset payoffs after each generation
                for node in self.nodes:
                    node['payoff'] = 0

        elif self.update == 'Asynchronous':
            for i in range(self.numGenerations):
                lg = self.runGenerationAsynchronous()
                lg['generation'] = i
                if i % self.logFreq == 0:
                    print('== Logging {} =='.format(i))
                # drawGraph(self.layer1, self.nodes, dir, i)

                LogsPerGen.append(lg)
                stat, numRep = stationaryFraction(self.nodes)
                statFrac.append(stat)

                for node in self.nodes:
                    node['payoff'] = 0

        else:
            print('Wrong update method.')
            exit()

        self.runVisualization()
        coopRatio = calculateAverage(LogsPerGen, 'cooperationRatio', self.numGenerations)

        print(statFrac[-1])
        print(numRep) # [ repAllC, repAllD, repDisc, repPDisc]; repAllC = [no. good, no. bad], AllD = ...
        print("CoopRatio:", coopRatio)
        print("Last gen: ", LogsPerGen[-1])

    def runGeneration(self):
        interactionPairs = getNeighborPairs(self.layer1, self.nodes, self.nodePos, self.numInteractions)
        actions = []
        for n, pair in enumerate(interactionPairs):
            actions.append(self.runInteraction(pair))
            self.runGossip(pair, actions[-1])  # Update perceptions of the gossiper's neighbors in L2

        actionFreq = countFreq(actions)
        cooperationRatio = actionFreq['Cooperate'] if 'Cooperate' in actionFreq.keys() else 0

        return {'cooperationRatio': cooperationRatio}

    def runInteraction(self, pair):  # Donation Game
        donor = pair[0]
        recipient = pair[1]
        if (donor['strategy'] == 'AllC') or \
                (donor['strategy'] == 'Disc' and getRecipientReputation(donor, recipient) == 'Good')\
                or donor['strategy'] == 'pDisc' and getRecipientReputation(donor, recipient) == 'Bad':
            action = 'Cooperate'
            donor['payoff'] -= self.cost
            recipient['payoff'] += self.benefit

        else:
            action = 'Defect'

        '''
        # Verify if it's working properly
        print(donor['strategy'])
        print(getRecipientReputation(donor, recipient))
        print(action)
        '''
        return action

    def runGossip(self, pair, action):
        # The neighbors of the gossiper on layer 2 will update their perception of the donor
        # Choose one of the donor's neighbors who will witness and gossip about the interaction
        gossiper = pickNeighbor(self.layer1, pair[0], self.nodes, self.nodePos)
        updatePerception(self.socialNorm, gossiper, pair[0], pair[1], action)
        neighbors = self.layer2.neighbors(gossiper['pos'])
        for neighbor in neighbors:
            if probability(self.transError):  # Transmission Error
                if gossiper['perception'][pair[0]['pos']]['reputation'] == 'Good':
                    self.nodes[neighbor]['perception'][pair[0]['pos']]['reputation'] = 'Bad'
                else:
                    self.nodes[neighbor]['perception'][pair[0]['pos']]['reputation'] = 'Good'
            else:
                self.nodes[neighbor]['perception'][pair[0]['pos']]['reputation'] = gossiper['perception'][pair[0]['pos']]['reputation']
            # todo - to repeat this for the neighbors' neighbors, maybe making this function recursive

    def socialLearning(self):
        # social learning where nodes copy another node's strategy with a given probability if
        # that node's payoff is higher

        for node in self.nodes:
            # Probability of exploring a random strategy
            if probability(self.explorationRate):
                node['strategy'] = self.calculateInitialStrategy()

            # Fitness comparison
            else:
                neighbor = pickNeighbor(self.layer1, node, self.nodes, self.nodePos)
                # If the node's payoff is negative, the probability is increased. To combat this:
                if node['payoff'] < 0:
                    node['payoff'] = 0

                prob = 1 / (1 + math.exp(-self.beta * (neighbor['payoff'] - node['payoff'])))
                if probability(prob):
                    node['strategy'] = neighbor['strategy']

        # Code below is to compare a node's fitness with all neighbors
        '''
        interactionPairs = getNeighborPairs(self.layer1, self.nodes, self.nodePos)

        for pair in interactionPairs:
            mine = pair[0]
            partner = pair[1]
            if probability(self.explorationRate):
                mine['strategy'] = self.calculateInitialStrategy()
            else:
                if mine['payoff'] < 0:
                    mine['payoff'] = 0
                    
                prob = 1 / (1 + math.exp(-beta * (partner['payoff'] - mine['payoff'])))
                if probability(prob):
                    mine['strategy'] = partner['strategy']
        '''

    def runVisualization(self):
        # Add node colors for Gephi
        for item in self.nodes:
            if item['strategy'] == 'AllC':  # Blue
                self.nodes[item['pos']]['viz'] = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 0}}
            elif item['strategy'] == 'AllD':  # Red
                self.nodes[item['pos']]['viz'] = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 0}}
            elif item['strategy'] == 'Disc':  # Cyan
                self.nodes[item['pos']]['viz'] = {'color': {'r': 0, 'g': 255, 'b': 255, 'a': 0}}
            elif item['strategy'] == 'pDisc':  # Orange
                self.nodes[item['pos']]['viz'] = {'color': {'r': 255, 'g': 165, 'b': 0, 'a': 0}}

        # fixme - find a simpler way of doing this
        for i in range(self.numNodes):
            self.layer1.nodes[i]['pos'] = self.nodes[i]['pos']
            self.layer1.nodes[i]['id'] =  self.nodes[i]['id']
            self.layer1.nodes[i]['payoff'] = self.nodes[i]['payoff']
            self.layer1.nodes[i]['strategy'] = self.nodes[i]['strategy']
            self.layer1.nodes[i]['viz'] = self.nodes[i]['viz']

        # print(self.layer1.nodes(data=True))
        nx.write_gexf(self.layer1, self.gephi)

    def runGenerationAsynchronous(self):
        cooperationRatio = 0
        #for node in self.nodes: # For all nodes
        for i in range(self.numNodes): # For a random node numNodes times
            node = random.choice(self.nodes)
            if probability(self.explorationRate):
                node['strategy'] = self.calculateInitialStrategy()
            else:
                neighbor = pickNeighbor(self.layer1, node, self.nodes, self.nodePos)
                if neighbor:  # Only if the node has neighbors
                    chosen = [node, neighbor]
                    interactionPairs = getNeighborsAsynchronous(self.layer1, chosen, self.nodes, self.nodePos, self.numInteractions)
                    actions = []
                    for n, pair in enumerate(interactionPairs):
                        actions.append(self.runInteraction(pair))
                        self.runGossip(pair, actions[-1])  # Update perceptions of the gossiper's neighbors in L2

                    self.socialLearningAsynchronous(node, neighbor)

                    actionFreq = countFreq(actions)
                    cooperationRatio = actionFreq['Cooperate'] if 'Cooperate' in actionFreq.keys() else 0

        return {'cooperationRatio': cooperationRatio}

    def socialLearningAsynchronous(self, node, neighbor):
        # Social learning where nodes copy another node's strategy with a probability proportionate to their fitness
        if node['payoff'] < 0:
            node['payoff'] = 0
        prob = 1 / (1 + math.exp(-self.beta * (neighbor['payoff'] - node['payoff'])))
        if probability(prob):
            node['strategy'] = neighbor['strategy']

if __name__ == "__main__":
    # Variables used
    initialValues = {
        'numNodes': 100,  # Number of nodes
        'prob1': 0.25,  # Probability of rewiring links (WattsStrogatz) for Layer 1
        'prob2': 0.25,  # Probability of rewiring links (WattsStrogatz) for Layer 2
        'avgDegree': 4,
        'numGenerations': 5000,
        'numInteractions': 3,  # Number of times nodes play with each of their neighbors. Must be > 0
        'logFreq': 1000,  # How frequently should the model take logs of the simulation (in generations)
        'cost': 0.1,  # Cost of cooperation
        'benefit': 1,  # Benefit of receiving cooperation
        'explorationRate': 0.01,  # Probability of a node adopting a random strategy during Social Learning
        'transError': 0.01,  # Transmission error, in which case an individual gossips wrong information (contrary to his beliefs)
        'beta': 1,  # Pairwise comparison function: p = 1 / (1 + math.exp(-beta * (Fb - Fa)))
        'rndSeed': None,  # Indicator of random number generation state
        'gephiFileName': 'test.gexf',  # File name used for the gephi export. Must include '.gexf'
        'layer1': 'WattsStrogatz',  # Graph topology: 'WattsStrogatz', 'Random', 'BarabasiAlbert',
        'layer2': 'RandomizedNeighborhoods',  # Graph topology: 'WattsStrogatz', 'Random', 'PerfectOverlap' (Layers are equal), 'RandomizedNeighborhoods'
        # (same degree, different neighborhoods), 'TotalRandomization' (degree and neighborhoods are different)
        'fractionNodes': 0.5,  # Fraction of nodes randomized (switch edges) for Randomized Neighborhoods
        'update': 'Asynchronous',  # 'Synchronous' or 'Asynchronous'
        'socialNorm': 'SternJudging',  # SimpleStanding, ImageScoring, Shunning or SternJudging

    }

    changes = [{'socialNorm': 'SimpleStanding',}, {'socialNorm': 'ImageScoring',}, {'socialNorm': 'Shunning',}, {'socialNorm': 'SternJudging',}]
    #changes = [{}]
    '''
    changes = [
               #{'gephiFileName':  'SJ.gexf', 'socialNorm': 'SternJudging',}, {'gephiFileName':  'SS.gexf', 'socialNorm': 'SimpleStanding',}, \
               {'gephiFileName':  'IS.gexf', 'socialNorm': 'ImageScoring',}, {'gephiFileName':  'SH.gexf', 'socialNorm': 'Shunning',},
               ]
    '''
    #changes = [{'update': 'Synchronous', }, {'update': 'Asynchronous', }]

    for j, c in enumerate(changes):
        config = initialValues.copy()
        config.update(c)
        dir = join('output', 'testrun{}'.format(j))

        if not os.path.exists(dir):
            mkdir(dir)

        sim = IndirectReciprocityMultiplexNetworks(**config)
        sim.runSimulation()

        with open(join(dir, 'config.json'), 'w') as fp:
            json.dump(config, fp)

    # todo - calculate clustering coefficient and include it in the results
    # todo - add more graph topologies
    # todo - fix plot labels
    # todo - Histograms
    # todo - learn gephi
    # todo - writing thesis
    # todo - test the entire program
