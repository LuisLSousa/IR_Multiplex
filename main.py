#! python3
import json
import os
from os import mkdir
from utils import *
from mpmath import *


class IndirectReciprocityMultiplexNetworks:

    def __init__(self, numNodes=100, prob1=0.5, prob2=0.5, avgDegree1=4, avgDegree2=4, numGenerations=100,
                 numInteractions=1,
                 logFreq=1, cost=0.1, benefit=1, transError=0.01, executionError=0.01, assessmentError=0.01,
                 assignmentError=0.01, beta=10, update='Synchronous', explorationRate=0.001,
                 rndSeed=None, gephiFileName='test.gexf', layer1=None, layer2=None, socialNorm='SternJudging',
                 numSwap=1000, logsFileName='logs', typeOfSimulation=None, outputDirectory=None):

        self.numNodes = numNodes  # Number of nodes
        self.nodes = []
        self.prob1 = prob1  # Rewire probability for Watts-Strogatz - L1
        self.prob2 = prob2  # Rewire probability for Watts-Strogatz - L2
        self.avgDegree1 = avgDegree1
        self.avgDegree2 = avgDegree2
        self.numGenerations = numGenerations
        self.numInteractions = numInteractions  # The number of interactions on each generation
        self.logFreq = logFreq  # Generate graphs at every X simulations (Currently unused, just prints the iteration to show progress)
        self.cost = cost  # Donation Game
        self.benefit = benefit  # Donation Game
        self.beta = beta
        self.rndSeed = rndSeed
        self.transError = transError  # Probability of the gossiper transmitting wrong information to each neighbor
        self.explorationRate = explorationRate  # Probability of a node adopting a random strategy
        self.executionError = executionError  # Probability of a node attempting to cooperate failing to do so
        self.assessmentError = assessmentError
        self.assignmentError = assignmentError
        self.socialNorm = socialNorm  # Global social norm (the entire population follows this)
        self.gephi = gephiFileName  # File name for the gephi export
        self.layer1 = layer1  # Layer1 topology
        self.layer2 = layer2  # Layer2 topology
        self.update = update  # Synchronous or Asynchronous
        self.numSwap = numSwap
        self.clusteringCoef1 = 0  # Clustering Coefficient used in the plots
        self.APL = 0  # Average path length
        self.logsFileName = 'output/' + outputDirectory + '/' + logsFileName
        self.typeOfSim = typeOfSimulation

        # Variable used for the x_axis in the final plot
        if self.typeOfSim == 'pWattsStrogatz':
            self.x_var = self.prob1
        elif self.typeOfSim == 'explorationRate':
            self.x_var = self.explorationRate
        elif self.typeOfSim == 'avgDegree1':
            self.x_var = self.avgDegree1
        elif self.typeOfSim == 'avgDegree2':
            self.x_var = self.avgDegree2
        elif not self.typeOfSim:
            self.x_var = None
        else:
            print('Wrong typeOfSimulation!')
            exit()

        if self.numInteractions <= 0:
            print('numInteractions must be > 0')
            exit()

        if self.explorationRate == 1:
            print('For explorationRate = 1, nodes always adopt a random strategy, and the donation game is never played. \
            Hence, there will be no cooperation and coopRatio = 0')

        self.idIterator = 0
        self.idToIndex = {}  # id:index
        self.initiateGraph()
        self.nodePos = list(self.layer1.nodes())
        self.initiateNodes()

    def initiateNodes(self):
        append = self.nodes.append  # Faster than calling self.nodes.append every time
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
        # for i in range(self.numNodes):
        #   print('Node:', self.nodes[i]['id'], '| Strategy:', self.nodes[i]['strategy'])

        # Each node has its own perception of all others
        # for node in self.nodes:
        #    node['perception'] = [calculateInitialReputation() for i in self.nodes]
        self.perceptions = [[calculateInitialReputation() for x in range(len(self.nodes))] for y in
                            range(len(self.nodes))]

        # Each node also has a random perception of himself.
        # Example: Node number 3's perception of node number 7
        # print(self.nodes[3]['perception'][7]) # old

    def initiateGraph(self):
        global ringAPL
        global graph
        global graph2
        global x_axis
        flagLayer2 = False

        if self.layer1 == 'Random':
            self.layer1 = MultiplexNetwork(self.numNodes, self.avgDegree1)

        elif self.layer1 == 'WattsStrogatz' and not self.typeOfSim:  # If it's just one simulation without a plot
            self.layer1 = wattsStrogatz(self.numNodes, self.avgDegree1, self.prob1, self.rndSeed)

        elif self.layer1 == 'WattsStrogatz' and self.x_var not in x_axis and self.typeOfSim != 'avgDegree2':  # if the simulation is not in function of the avgDegree2
            self.layer1 = wattsStrogatz(self.numNodes, self.avgDegree1, self.prob1, self.rndSeed)
            flagLayer2 = True  # When layer1 changes, layer2 should change as well
            graph = self.layer1
            self.clusteringCoef1 = nx.transitivity(self.layer1)
            self.APL = nx.average_shortest_path_length(self.layer1)

            if self.prob1 == 0:  # Used to normalize the APL for different values of "p-watts-strogatz"
                ringAPL = self.APL

            # pWattsStrogatz.append(self.prob1)
            x_axis.append(self.x_var)
            CC.append(self.clusteringCoef1)
            APL.append(self.APL / ringAPL)

        elif self.layer1 == 'WattsStrogatz' and self.x_var in x_axis:  # if the graph for the current simulation has already been generated (e.g., when testing different norms with the same graph)
            self.layer1 = graph
            flagLayer2 = False  # If layer1 doesn't change, neither should layer2

        elif self.layer1 == 'WattsStrogatz' and self.typeOfSim == 'avgDegree2':
            self.layer1 = wattsStrogatz(self.numNodes, self.avgDegree1, self.prob1, self.rndSeed)
            graph = self.layer1

        elif self.layer1 == 'BarabasiAlbert':
            self.layer1 = barabasiAlbert(self.numNodes, self.avgDegree1, self.rndSeed)

        elif self.layer1 == 'Complete':
            if self.x_var not in x_axis:
                x_axis.append(self.x_var)
                self.layer1 = completeGraph(self.numNodes)
                graph = self.layer1
            else:
                self.layer1 = graph  # faster than generating a new graph each time

        else:
            print('Wrong layer1 parameter!')
            exit()

        if self.layer2 == 'Random':
            self.layer2 = MultiplexNetwork(self.numNodes, self.avgDegree2)

        elif self.layer2 == 'WattsStrogatz' and self.x_var not in x_axis:  # this is only true when the simulation is in function of avgDegree2 and a new value is tested
            self.layer2 = wattsStrogatz(self.numNodes, self.avgDegree2, self.prob2, self.rndSeed)
            graph2 = self.layer2
            x_axis.append(self.x_var)

        elif self.layer2 == 'WattsStrogatz' and self.x_var in x_axis:
            self.layer2 = graph2

        elif self.layer2 == 'BarabasiAlbert':
            self.layer2 = barabasiAlbert(self.numNodes, self.avgDegree2, self.rndSeed)

        elif self.layer2 == 'PO':
            self.layer2 = copy.deepcopy(self.layer1)

        elif self.layer2 == 'RN' and flagLayer2:
            self.layer2 = randomizedNeighborhoods(self.layer1, self.numSwap, self.numNodes, self.rndSeed)
            graph2 = self.layer2

        elif self.layer2 == 'RN' and not flagLayer2:
            self.layer2 = graph2

        elif self.layer2 == 'TR' and flagLayer2:
            self.layer2 = totalRandomization(self.layer1, self.numNodes)
            graph2 = self.layer2

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

        else:
            print('Wrong update method.')
            exit()

        # self.runVisualization()
        if self.numGenerations >= 5000:  # If there are enough generations for the simulation to stabilize, count 90% of the results after
            logsGen = int(self.numGenerations * 0.8)
            coopRatio = calculateAverage(LogsPerGen[-logsGen:],
                                         'cooperationRatio')  # Use last 100 generations for the average cooperation ratio
        else:
            coopRatio = calculateAverage(LogsPerGen[-100:], 'cooperationRatio')
        plotValues(coopRatio, self.socialNorm)
        stat, numRep = stationaryFraction(self.nodes, self.perceptions)
        statFrac.append(stat)
        print("Social Norm: {}".format(self.socialNorm))
        print("Stationary fraction ( [G, B] ): ", stat)
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
        f.write("\nSocial Norm: {}".format(self.socialNorm))
        f.write("\n{}: {}".format(self.typeOfSim, self.x_var))
        f.write("\nStationary fraction ( [G, B] ): {}".format(statFrac))
        f.write("\nNumber of reputations [repAllC, repAllD, repDisc, repPDisc]:{}".format(numRep))
        f.write("\nAverage Cooperation Ratio: " + str(coopRatio))
        for item in LogsPerGen:
            f.write("\n {}".format(item))
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
                (donor['strategy'] == 'Disc' and self.assessReputation(donor, recipient) == 'Good') \
                or donor['strategy'] == 'pDisc' and self.assessReputation(donor, recipient) == 'Bad':
            if probability(self.executionError):
                action = 'Defect'
            else:
                action = 'Cooperate'
                donor['payoff'] -= self.cost
                recipient['payoff'] += self.benefit
        else:
            action = 'Defect'

        # Verify if it's working properly
        '''
        print('----------')
        print(donor['strategy'])
        print(self.perceptions[donor['pos']][recipient['pos']])
        print(action)
        '''

        return action

    def runGossip(self, pair, action):
        # The neighbors of the gossiper on layer 2 will update their perception of the donor
        # Choose one of the donor's neighbors who will witness and gossip about the interaction
        gossiper = pickNeighbor(self.layer1, pair[0], self.nodes, self.nodePos)  # Pick one of the donor's neighbors
        self.updatePerception(gossiper, pair[0], pair[1], action)  # Update that node's opinion of the donor
        neighbors = self.layer2.neighbors(gossiper['pos'])
        for neighbor in neighbors:
            if probability(self.transError):  # Transmission Error
                if self.assessReputation(gossiper, pair[0]) == 'Good':
                    self.assignReputation(neighbor, pair[0], 'Bad')
                else:
                    self.assignReputation(neighbor, pair[0], 'Good')
            else:
                self.perceptions[neighbor][pair[0]['pos']] = self.perceptions[gossiper['pos']][pair[0]['pos']]
                # todo - self.assignReputation(neighbor, pair[0], self.perceptions[gossiper['pos']][pair[0]['pos']])

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
                prob = 1 / (1 + mp.exp(-self.beta * (neighbor['payoff'] - node['payoff'])))
                if probability(prob):
                    node['strategy'] = neighbor['strategy']

        # Code below is to compare a node's fitness with all neighbors instead of comparing with one only
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
        '''

    def runGenerationAsynchronous(self):
        cooperationRatio = 0
        # for node in self.nodes: # For all nodes
        for i in range(self.numNodes):  # For a random node numNodes times
            node = random.choice(self.nodes)
            if probability(self.explorationRate):
                oldStrategy = node['strategy']
                while node['strategy'] == oldStrategy:
                    node['strategy'] = calculateInitialStrategy()
            else:
                neighbor = pickNeighbor(self.layer1, node, self.nodes, self.nodePos)
                if neighbor:  # Only if the node has neighbors
                    chosen = [node, neighbor]
                    interactionPairs, gamesPlayed = getNeighborsAsynchronous(self.layer1, chosen, self.nodes, self.nodePos, self.numInteractions)
                    actions = []
                    append = actions.append

                    for n, pair in enumerate(interactionPairs):
                        append(self.runInteraction(pair))
                        self.runGossip(pair, actions[-1])  # Update perceptions of the gossiper's neighbors in L2

                    # Calculate the average payoff for both individuals
                    chosen[0]['payoff'] /= gamesPlayed[0]
                    chosen[1]['payoff'] /= gamesPlayed[1]

                    self.socialLearningAsynchronous(node, neighbor)
                    actionFreq = countFreq(actions)
                    cooperationRatio = actionFreq['Cooperate'] if 'Cooperate' in actionFreq.keys() else 0
                    if 'Cooperate' not in actionFreq.keys() and 'Defect' not in actionFreq.keys():
                        print('--------- No donation games were played, cooperationRatio = 0 ---------')

                else:
                    print("------ No neighbors ------")
            # Reset payoffs after each strategy update
            for node in self.nodes:
                node['payoff'] = 0
        return {'cooperationRatio': cooperationRatio}

    def socialLearningAsynchronous(self, node, neighbor):
        # Social learning where nodes copy another node's strategy with a probability proportionate to their fitness
        prob = 1 / (1 + mp.exp(-self.beta * (neighbor['payoff'] - node['payoff'])))
        if probability(prob):
            node['strategy'] = neighbor['strategy']

    def updatePerception(self, witness, donor, recipient, action):
        # Stern Judging (Coop with Good = G; Defect with Bad = G; else B)
        if self.socialNorm == 'SternJudging':
            if action == 'Cooperate' and self.assessReputation(witness, recipient) == 'Good':
                self.assignReputation(witness, donor, 'Good')
            elif action == 'Defect' and self.assessReputation(witness, recipient) == 'Bad':
                self.assignReputation(witness, donor, 'Good')
            else:
                self.assignReputation(witness, donor, 'Bad')

        # Simple Standing (Defect with Good = B; else G)
        elif self.socialNorm == 'SimpleStanding':
            if action == 'Defect' and self.assessReputation(witness, recipient) == 'Good':
                self.assignReputation(witness, donor, 'Bad')
            else:
                self.assignReputation(witness, donor, 'Good')

        # Shunning (Coop with Good = G; else B)
        elif self.socialNorm == 'Shunning':
            if action == 'Cooperate' and self.assessReputation(witness, recipient) == 'Good':
                self.assignReputation(witness, donor, 'Good')
            else:
                self.assignReputation(witness, donor, 'Bad')

        # Image Scoring (Coop = Good; Defect = Bad)
        elif self.socialNorm == 'ImageScoring':
            if action == 'Cooperate':
                self.assignReputation(witness, donor, 'Good')
            elif action == 'Defect':
                self.assignReputation(witness, donor, 'Bad')
            else:
                print('Error: Action is neither "Cooperate" nor "Defect" ')
                exit()

        # All Good (All actions are deemed good - used to establish a baseline)
        elif self.socialNorm == 'AllGood':
            self.assignReputation(witness, donor, 'Good')

        else:
            print('Wrong socialNorm, check initial parameters')
            exit()

    def assessReputation(self, subject, target):
        # The subject is assessing the target's reputation
        if probability(self.assessmentError):
            if self.perceptions[subject['pos']][target['pos']] == 'Good':
                return 'Bad'
            else:
                return 'Good'
        else:
            return self.perceptions[subject['pos']][target['pos']]

    def assignReputation(self, subject, target, value):
        # Update the subject's perception of the target
        if not isinstance(subject, int):
            subject = subject['pos']
        if not isinstance(target, int):
            target = target['pos']
        assert (value == 'Good' or value == 'Bad'), 'Wrong value: use Good or Bad'
        if probability(self.assignmentError):  # probability of assigning the wrong reputation
            if value == 'Good':
                self.perceptions[subject][target] = 'Bad'
            elif value == 'Bad':
                self.perceptions[subject][target] = 'Good'
        else:
            self.perceptions[subject][target] = value


if __name__ == "__main__":
    # Variables used
    start_time = time.time()
    initialValues = {
        'numNodes': 50,  # Number of nodes
        'prob1': 0.5,  # Probability of rewiring links (WattsStrogatz) for Layer 1
        'prob2': 0.5,  # Probability of rewiring links (WattsStrogatz) for Layer 2
        'avgDegree1': 4,  # Layer 1
        'avgDegree2': 8,  # Layer 2
        'numGenerations': 1000,  # 5000
        'numInteractions': 2,  # Number of times nodes play with each of their neighbors. Must be > 0
        'logFreq': 1000,
        # How frequently should the model take logs of the simulation (in generations) (unused, now just prints iterations)
        'cost': 1,  # Cost of cooperation
        'benefit': 5,  # Benefit of receiving cooperation
        'explorationRate': 0.02,  # (0.01) Probability of a node adopting a random strategy instead of Social Learning
        'transError': 0.0,  # Probability of a node gossiping wrong information (0.01)
        'executionError': 0.01,  # Probability of a donor attempting to cooperate failing to do so (0.01)
        'assignmentError': 0.01,
        # Probability of the assigned reputation being the opposite of the prescribed by the social norm (extra)
        'assessmentError': 0.01,  # Probability of assessing a reputation opposite to the one actually owned (extra)
        'beta': 1,  # Pairwise comparison function: p = 1 / (1 + math.exp(-beta * (Fb - Fa)))
        'rndSeed': None,  # Indicator of random number generation state
        'gephiFileName': 'test.gexf',
        # File name used for the gephi export. Must include '.gexf' (Currently unused as the visualization is not needed)
        'layer1': 'Complete',  # Graph topology: 'WattsStrogatz', 'Random', 'BarabasiAlbert', 'Complete'
        'layer2': 'PO',  # Graph topology: 'WattsStrogatz', 'Random', 'BarabasiAlbert',
        # 'PO' - Perfect Overlap (Layers are equal),
        # 'RN' - Randomized Neighborhoods (same degree, different neighborhoods),
        # 'TR' - Total Randomization (degree and neighborhoods are different)
        'numSwap': 4000,  # Number of edges swapped for Randomized Neighborhoods
        'update': 'Asynchronous',  # 'Synchronous' or 'Asynchronous'
        'socialNorm': 'SimpleStanding',  # SimpleStanding, ImageScoring, Shunning, SternJudging or AllGood (baseline)
        'logsFileName': 'logs.txt',
        'typeOfSimulation': 'explorationRate',
        # 'pWattsStrogatz', 'avgDegree1', 'avgDegree2', 'explorationRate', None - for just one simulation (no plot)
        'outputDirectory': 'explorationRate',  # Name of the output directory
    }
    runs = 1  # How many times should each simulation be repeated

    # changes = [{}] # Default for a single simulation
    '''changes = [{'socialNorm': 'SternJudging', },
               {'socialNorm': 'SimpleStanding', },
               {'socialNorm': 'Shunning', },
               {'socialNorm': 'ImageScoring', }, ]'''

    '''changes = [{'avgDegree2': 2, 'socialNorm': 'SternJudging', },
               {'avgDegree2': 2, 'socialNorm': 'Shunning', },
               {'avgDegree2': 2, 'socialNorm': 'ImageScoring', },
               {'avgDegree2': 2, 'socialNorm': 'SimpleStanding', },
               {'avgDegree2': 2, 'socialNorm': 'AllGood', },

               {'avgDegree2': 4, 'socialNorm': 'SternJudging', },
               {'avgDegree2': 4, 'socialNorm': 'Shunning', },
               {'avgDegree2': 4, 'socialNorm': 'ImageScoring', },
               {'avgDegree2': 4, 'socialNorm': 'SimpleStanding', },
               {'avgDegree2': 4, 'socialNorm': 'AllGood', },

               {'avgDegree2': 6, 'socialNorm': 'SternJudging', },
               {'avgDegree2': 6, 'socialNorm': 'Shunning', },
               {'avgDegree2': 6, 'socialNorm': 'ImageScoring', },
               {'avgDegree2': 6, 'socialNorm': 'SimpleStanding', },
               {'avgDegree2': 6, 'socialNorm': 'AllGood', },

               {'avgDegree2': 8, 'socialNorm': 'SternJudging', },
               {'avgDegree2': 8, 'socialNorm': 'Shunning', },
               {'avgDegree2': 8, 'socialNorm': 'ImageScoring', },
               {'avgDegree2': 8, 'socialNorm': 'SimpleStanding', },
               {'avgDegree2': 8, 'socialNorm': 'AllGood', }, ]'''

    changes = [{'explorationRate': 0.01, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.01, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.01, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.01, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.03, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.03, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.03, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.03, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.05, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.05, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.05, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.05, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.08, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.08, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.08, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.08, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.15, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.15, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.15, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.15, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.3, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.3, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.3, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.3, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.5, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.5, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.5, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.5, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.7, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.7, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.7, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.7, 'socialNorm': 'SimpleStanding', },

               {'explorationRate': 0.9, 'socialNorm': 'SternJudging', },
               {'explorationRate': 0.9, 'socialNorm': 'Shunning', },
               {'explorationRate': 0.9, 'socialNorm': 'ImageScoring', },
               {'explorationRate': 0.9, 'socialNorm': 'SimpleStanding', },
               ]

    '''changes = [{'prob1': 0, 'socialNorm': 'AllGood', },
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
               {'prob1': 1, 'socialNorm': 'ImageScoring', }, ]'''

    for j, c in enumerate(changes):
        print("--- %s seconds ---" % (time.time() - start_time))
        config = initialValues.copy()
        config.update(c)
        # dir = join('output', 'test{}'.format(j))
        # if not os.path.exists(dir):
        #    mkdir(dir)
        for i in range(runs):
            sim = IndirectReciprocityMultiplexNetworks(**config)
            sim.runSimulation()
            print('SJ = {}'.format(SJ))
            print('SS = {}'.format(SS))
            print('SH = {}'.format(SH))
            print('IS = {}'.format(IS))

    config = initialValues.copy()
    ch = changes.copy()
    log = [config, ch]  # one file with the initial values and all changes

    if initialValues['outputDirectory']:  # if outputDirectory is None, directory will be called 'test'
        dir = join('output', initialValues['outputDirectory'])
    else:
        dir = join('output', 'test')

    if not os.path.exists(dir):
        mkdir(dir)

    # todo: while os.path.exists(dir): add _x to dir (x is an int)

    with open(join(dir, 'config.json'), 'w') as fp:
        json.dump(log, fp)

    if initialValues['typeOfSimulation']:  # if typeOfSimulation is None, there will be no plot
        filename = 'output/' + initialValues['outputDirectory'] + '/' + initialValues['typeOfSimulation'] + '.png'
        i = 0
        while os.path.isfile(filename):
            i += 1
            filename = 'output/' + initialValues['outputDirectory'] + '/' + initialValues[
                'typeOfSimulation'] + '_{}'.format(i) + '.png'

        f = open('pltOutput.txt', "a")
        f.write("\n######################################")
        # f.write("\nAllG: {}".format(AllG))
        f.write("\nSJ = {}".format(SJ))
        f.write("\nSH = {}".format(SH))
        f.write("\nIS = {}".format(IS))
        f.write("\nSS = {}".format(SS))

        # To run in Sigma can't use matplotlib - don't forget to comment this line
        runLogs(AllG, SJ, SH, IS, SS, CC, APL, x_axis, initialValues['typeOfSimulation'], filename=filename)
        '''
        coopBar = list(np.zeros(4))
        if SJ:
            SternJudging = sum(SJ)/len(SJ)
            coopBar[0] = SternJudging
        if SS:
            SimpleStanding = sum(SS)/len(SS)
            coopBar[1] = SimpleStanding
        if SH:
            Shunning = sum(SH)/len(SH)
            coopBar[2] = Shunning
        if IS:
            ImageScoring = sum(IS)/len(IS)
            coopBar[3] = ImageScoring

        print('Coop Bar =  {}'.format(coopBar))

        f = open('pltOutput.txt', "a")
        f.write("\n######################################")
        #f.write("\nAllG: {}".format(AllG))
        f.write("\nSJ = {}".format(SJ))
        f.write("\nSH = {}".format(SH))
        f.write("\nIS = {}".format(IS))
        f.write("\nSS = {}".format(SS))
        f.write("\ncoopRatio = {}".format(coopBar))
        #f.write("\nx_axis = {}".format(x_axis))
        #f.write("\ntypeOfSimulation = \'{}\'".format(initialValues['typeOfSimulation']))
        #f.write("\nfilename: \'{}\' ".format(filename))
        f.close()

        #barPlot(coopBar)
        '''
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))
    # exit()
