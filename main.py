import json
import os
from os import mkdir
from utils import *
from mpmath import *

class IndirectReciprocityMultiplexNetworks:

    def __init__(self, numNodes=100, prob1=0.5, prob2=0.5, avgDegree=2, numGenerations=100, numInteractions=1,
                 logFreq=1, cost=0.1, benefit=1, transError=0.01, executionError=0.01, beta=10, update='Synchronous', explorationRate=0.01,
                 rndSeed=None, gephiFileName='test.gexf', layer1=None, layer2=None, socialNorm='SternJudging',
                 fractionNodes=0.2, logsFileName='logs'):

        self.numNodes = numNodes  # Number of nodes
        self.nodes = []
        self.prob1 = prob1  # Rewire probability for Watts-Strogatz - L1
        self.prob2 = prob2  # Rewire probability for Watts-Strogatz - L2
        self.avgDegree = avgDegree
        self.numGenerations = numGenerations
        self.numInteractions = numInteractions
        self.logFreq = logFreq  # Generate graphs at every X simulations
        self.cost = cost  # Donation Game
        self.benefit = benefit  # Donation Game
        self.beta = beta
        self.rndSeed = rndSeed
        self.transError = transError
        self.explorationRate = explorationRate
        self.executionError = executionError
        self.socialNorm = socialNorm  # Global social norm (the entire population follows this)
        self.gephi = gephiFileName  # File name for the gephi export
        self.layer1 = layer1  # Layer1 topology
        self.layer2 = layer2  # Layer2 topology
        self.update = update
        self.fractionNodes = fractionNodes
        self.clusteringCoef1 = 0
        self.APL = 0
        self.logsFileName = logsFileName

        if self.numInteractions <= 0:
            print("numInteractions must be > 0")
            exit()

        self.idIterator = 0
        self.idToIndex = {}  # id:index
        self.initiateGraph()
        self.nodePos = list(self.layer1.nodes())
        self.initiateNodes()

    def initiateNodes(self):
        append = self.nodes.append # Faster than calling self.nodes.append every time
        for i in range(self.numNodes):
            append({
                'pos': self.nodePos[i],  # Redundant?
                'id': self.idIterator,
                'payoff': 0,
                'strategy': calculateInitialStrategy(),
                'viz': None  # Visualization for gephi
            })

            self.idToIndex[self.idIterator] = len(self.nodes) - 1
            self.idIterator += 1

        # Check nodes
        #for i in range(self.numNodes):
        #   print('Node:', self.nodes[i]['id'], '| Strategy:', self.nodes[i]['strategy'])

        # Each node has its own perception of all others
        #for node in self.nodes:
        #    node['perception'] = [calculateInitialReputation() for i in self.nodes]
        self.perceptions = [[calculateInitialReputation() for x in range(len(self.nodes))] for y in range(len(self.nodes))]
        # Each node also has a random perception of himself.

        # Example: Node number 3's perception of node number 7
        # print(self.nodes[3]['perception'][7]) # old

    def initiateGraph(self):
        global ringAPL
        global prevProbWS
        global graph
        global graph2

        if self.layer1 == 'Random':
            self.layer1 = MultiplexNetwork(self.numNodes, self.avgDegree)

        elif self.layer1 == 'WattsStrogatz' and prevProbWS != self.prob1:
            self.layer1 = wattsStrogatz(self.numNodes, self.avgDegree, self.prob1, self.rndSeed)
            prevProbWS = self.prob1
            flagLayer2 = True # When layer1 changes, layer2 should change as well
            graph = self.layer1
            self.clusteringCoef1 = nx.transitivity(self.layer1)
            self.APL = nx.average_shortest_path_length(self.layer1)

            if self.prob1 == 0:  # Used to normalize the APL for different values of "p-watts-strogatz"
                ringAPL = self.APL

            pWattsStrogatz.append(self.prob1)
            CC.append(self.clusteringCoef1)
            APL.append(self.APL / ringAPL)

        elif self.layer1 == 'WattsStrogatz' and prevProbWS == self.prob1:
            self.layer1 = graph
            flagLayer2 = False # If layer1 doesn't change, neither should layer2

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

        elif self.layer2 == 'PO':
            self.layer2 = self.layer1

        elif self.layer2 == 'RN' and flagLayer2:
            self.layer2 = randomizedNeighborhoods(self.layer1, self.fractionNodes, self.numNodes, self.rndSeed)
            graph2 = self.layer2

        elif self.layer2 == 'RN' and not flagLayer2:
            self.layer2 = graph2

        elif self.layer2 == 'TR' and flagLayer2:
            self.layer2 = totalRandomization(self.layer1, self.numNodes)

        elif self.layer2 == 'TR' and not flagLayer2:
            self.layer2 = graph2

        else:
            print('Wrong layer2 parameter!')
            exit()

    def runSimulation(self):
        print('=====    Initiating simulation   ======')
        LogsPerGen = []
        numRep = []  # Number of AllC, AllD, Disc, and pDisc with G and B reputation
        # Stationary fraction of 'Good' and 'Bad' reputations of each gen. statFrac[ [good,bad] , [good,bad], ... ]
        statFrac = []

        if self.update == 'Synchronous':
            for i in range(self.numGenerations):
                lg = self.runGeneration()
                lg['generation'] = i
                if i % self.logFreq == 0:
                    print('== Logging {} =='.format(i))
                    # drawGraph(self.layer1, self.nodes, dir, i)

                self.socialLearning()
                LogsPerGen.append(lg)
                # stat, numRep = stationaryFraction(self.nodes, self.perceptions)
                # statFrac.append(stat)
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
                # stat, numRep = stationaryFraction(self.nodes, self.perceptions) # To see the evolution
                # statFrac.append(stat)
                # Reset payoffs after each generation
                for node in self.nodes:
                    node['payoff'] = 0
        else:
            print('Wrong update method.')
            exit()

        # self.runVisualization()
        coopRatio = calculateAverage(LogsPerGen[-100:], 'cooperationRatio') # Use last 100 generations for the average cooperation ratio
        plotValues(coopRatio, self.socialNorm)
        stat, numRep = stationaryFraction(self.nodes, self.perceptions)
        print("Stationary fraction ( [G, B] ): ", statFrac)
        print("Number of reputations: ", numRep)
        # [ repAllC, repAllD, repDisc, repPDisc]; repAllC = [no. good, no. bad], AllD = ...
        print("CoopRatio:", coopRatio)
        print("Last gen: ", LogsPerGen[-1])
        # print(LogsPerGen)

        # If file doesn't exist, create it
        if not os.path.isfile(self.logsFileName):
            f = open(self.logsFileName, "x")

        f = open(self.logsFileName, "a")
        f.write("######################################")
        f.write(f"\nSocial Norm: {self.socialNorm}")
        f.write(f"\nProbability: {self.prob1}")
        f.write(f"\nStationary fraction ( [G, B] ): {statFrac}")
        f.write(f"\nNumber of reputations [repAllC, repAllD, repDisc, repPDisc]:{numRep}")
        f.write(f"\nAverage Cooperation Ratio:" + str(coopRatio))
        for item in LogsPerGen:
            f.write(f"\n {item}")
        f.write("\n\n")
        f.close()

    def runGeneration(self):
        interactionPairs = getNeighborPairs(self.layer1, self.nodes, self.nodePos, self.numInteractions)
        actions = []
        append = actions.append
        for n, pair in enumerate(interactionPairs):
            append(self.runInteraction(pair))
            self.runGossip(pair, actions[-1])  # Update perceptions of the gossiper's neighbors in L2

        actionFreq = countFreq(actions)
        cooperationRatio = actionFreq['Cooperate'] if 'Cooperate' in actionFreq.keys() else 0

        return {'cooperationRatio': cooperationRatio}

    def runInteraction(self, pair):  # Donation Game
        donor = pair[0]
        recipient = pair[1]
        if (donor['strategy'] == 'AllC') or \
                (donor['strategy'] == 'Disc' and getRecipientReputation(donor, recipient, self.perceptions) == 'Good') \
                or donor['strategy'] == 'pDisc' and getRecipientReputation(donor, recipient, self.perceptions) == 'Bad':
            if probability(self.executionError):
                action = 'Defect'
            else:
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
        gossiper = pickNeighbor(self.layer1, pair[0], self.nodes, self.nodePos)  # Pick one of the donor's neighbors
        updatePerception(self.socialNorm, gossiper, pair[0], pair[1], action, self.perceptions)  # Update that node's opinion of the donor
        neighbors = self.layer2.neighbors(gossiper['pos'])
        for neighbor in neighbors:
            if probability(self.transError):  # Transmission Error
                if self.perceptions[gossiper['pos']][pair[0]['pos']] == 'Good':
                    self.perceptions[neighbor][pair[0]['pos']] = 'Bad'
                else:
                    self.perceptions[neighbor][pair[0]['pos']] = 'Good'
            else:
                self.perceptions[neighbor][pair[0]['pos']] = self.perceptions[gossiper['pos']][pair[0]['pos']]

            # todo - to repeat this for the neighbors' neighbors, maybe making this function recursive

    def socialLearning(self):
        # social learning where nodes copy another node's strategy with a given probability if
        # that node's payoff is higher

        for node in self.nodes:
            # Probability of exploring a random strategy
            if probability(self.explorationRate):
                node['strategy'] = calculateInitialStrategy()

            # Fitness comparison
            else:
                neighbor = pickNeighbor(self.layer1, node, self.nodes, self.nodePos)

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
                mine['strategy'] = calculateInitialStrategy()
            else:
                                  
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
            self.layer1.nodes[i]['id'] = self.nodes[i]['id']
            self.layer1.nodes[i]['payoff'] = self.nodes[i]['payoff']
            self.layer1.nodes[i]['strategy'] = self.nodes[i]['strategy']
            self.layer1.nodes[i]['viz'] = self.nodes[i]['viz']

        # print(self.layer1.nodes(data=True))
        nx.write_gexf(self.layer1, self.gephi)

    def runGenerationAsynchronous(self):
        cooperationRatio = 0
        # for node in self.nodes: # For all nodes
        for i in range(self.numNodes):  # For a random node numNodes times
            node = random.choice(self.nodes)
            if probability(self.explorationRate):
                node['strategy'] = calculateInitialStrategy()
            else:
                neighbor = pickNeighbor(self.layer1, node, self.nodes, self.nodePos)
                if neighbor:  # Only if the node has neighbors
                    chosen = [node, neighbor]
                    interactionPairs = getNeighborsAsynchronous(self.layer1, chosen, self.nodes, self.nodePos,
                                                                self.numInteractions)
                    actions = []
                    append = actions.append
                    for n, pair in enumerate(interactionPairs):
                        append(self.runInteraction(pair))
                        self.runGossip(pair, actions[-1])  # Update perceptions of the gossiper's neighbors in L2

                    self.socialLearningAsynchronous(node, neighbor)
                    actionFreq = countFreq(actions)
                    cooperationRatio = actionFreq['Cooperate'] if 'Cooperate' in actionFreq.keys() else 0

        return {'cooperationRatio': cooperationRatio}

    def socialLearningAsynchronous(self, node, neighbor):
        # Social learning where nodes copy another node's strategy with a probability proportionate to their fitness
        prob = 1 / (1 + mp.exp(-self.beta * (neighbor['payoff'] - node['payoff'])))
        if probability(prob):
            node['strategy'] = neighbor['strategy']


if __name__ == "__main__":
    # Variables used
    initialValues = {
        'numNodes': 500,  # Number of nodes
        'prob1': 0,  # Probability of rewiring links (WattsStrogatz) for Layer 1
        'prob2': 0,  # Probability of rewiring links (WattsStrogatz) for Layer 2
        'avgDegree': 8,
        'numGenerations': 100,
        'numInteractions': 2,  # Number of times nodes play with each of their neighbors. Must be > 0
        'logFreq': 250,  # How frequently should the model take logs of the simulation (in generations)
        'cost': 1,  # Cost of cooperation
        'benefit': 5,  # Benefit of receiving cooperation
        'explorationRate': 0.001,  # Probability of a node adopting a random strategy during Social Learning
        'transError': 0.01, # Probability of a node gossiping wrong information
        'executionError': 0.01, # Probability of a donor attempting to cooperate failing to do so
        # Transmission error, in which case an individual gossips wrong information (contrary to his beliefs)
        'beta': 10,  # Pairwise comparison function: p = 1 / (1 + math.exp(-beta * (Fb - Fa)))
        'rndSeed': None,  # Indicator of random number generation state
        'gephiFileName': 'test.gexf',  # File name used for the gephi export. Must include '.gexf'
        'layer1': 'WattsStrogatz',  # Graph topology: 'WattsStrogatz', 'Random', 'BarabasiAlbert',
        'layer2': 'PO',  # Graph topology: 'WattsStrogatz', 'Random', 'BarabasiAlbert',
        # 'PO' - Perfect Overlap(Layers are equal),
        # 'RN' - Randomized Neighborhoods (same degree, different neighborhoods),
        # 'TR' - Total Randomization (degree and neighborhoods are different)
        'fractionNodes': 0.1,  # Fraction of nodes randomized (switch edges) for Randomized Neighborhoods
        'update': 'Asynchronous',  # 'Synchronous' or 'Asynchronous'
        'socialNorm': 'AllGood',  # SimpleStanding, ImageScoring, Shunning, SternJudging or AllGood (baseline)
        'logsFileName': 'plots/logs.txt',
    }

    #changes = [{}] # Default for a single simulation

    changes = [{'prob1': 0, 'socialNorm': 'ImageScoring', },
               {'prob1': 0.00001, 'socialNorm': 'ImageScoring', },
               {'prob1': 0.0001, 'socialNorm': 'ImageScoring', },
               {'prob1': 0.001, 'socialNorm': 'ImageScoring', },
               {'prob1': 0.01, 'socialNorm': 'ImageScoring', },
               {'prob1': 0.1, 'socialNorm': 'ImageScoring', },
               {'prob1': 1, 'socialNorm': 'ImageScoring', }, ]

    '''
    changes = [{'prob1': 0, 'socialNorm': 'AllGood', },
               {'prob1': 0, 'socialNorm': 'SimpleStanding', },
               {'prob1': 0, 'socialNorm': 'SternJudging', },
               {'prob1': 0, 'socialNorm': 'Shunning', },
               {'prob1': 0, 'socialNorm': 'ImageScoring', },

               {'prob1': 0.00001, 'socialNorm': 'AllGood', },
               {'prob1': 0.00001, 'socialNorm': 'SimpleStanding', },
               {'prob1': 0.00001, 'socialNorm': 'SternJudging', },
               {'prob1': 0.00001, 'socialNorm': 'Shunning', },
               {'prob1': 0.00001, 'socialNorm': 'ImageScoring', },

               {'prob1': 0.0001, 'socialNorm': 'AllGood', },
               {'prob1': 0.0001, 'socialNorm': 'SimpleStanding', },
               {'prob1': 0.0001, 'socialNorm': 'SternJudging', },
               {'prob1': 0.0001, 'socialNorm': 'Shunning', },
               {'prob1': 0.0001, 'socialNorm': 'ImageScoring', },

               {'prob1': 0.001, 'socialNorm': 'AllGood', },
               {'prob1': 0.001, 'socialNorm': 'SimpleStanding', },
               {'prob1': 0.001, 'socialNorm': 'SternJudging', },
               {'prob1': 0.001, 'socialNorm': 'Shunning', },
               {'prob1': 0.001, 'socialNorm': 'ImageScoring', },

               {'prob1': 0.01, 'socialNorm': 'AllGood', },
               {'prob1': 0.01, 'socialNorm': 'SimpleStanding', },
               {'prob1': 0.01, 'socialNorm': 'SternJudging', },
               {'prob1': 0.01, 'socialNorm': 'Shunning', },
               {'prob1': 0.01, 'socialNorm': 'ImageScoring', },

               {'prob1': 0.1, 'socialNorm': 'AllGood', },
               {'prob1': 0.1, 'socialNorm': 'SimpleStanding', },
               {'prob1': 0.1, 'socialNorm': 'SternJudging', },
               {'prob1': 0.1, 'socialNorm': 'Shunning', },
               {'prob1': 0.1, 'socialNorm': 'ImageScoring', },

               {'prob1': 1, 'socialNorm': 'AllGood', },
               {'prob1': 1, 'socialNorm': 'SimpleStanding', },
               {'prob1': 1, 'socialNorm': 'SternJudging', },
               {'prob1': 1, 'socialNorm': 'Shunning', },
               {'prob1': 1, 'socialNorm': 'ImageScoring', }, ]
    '''

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

    # print("p: ", pWattsStrogatz)
    # print("SJ: ", SJ)
    # print("SS: ", SS)
    # print("SH: ", SH)
    # print('IS: ', IS)
    # print("CC: ", CC)
    # print("APL: ", APL)

    runLogs(AllG, SJ, SH, IS, SS, CC, APL, pWattsStrogatz, filename='plots/test.png')

    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))
    exit()
