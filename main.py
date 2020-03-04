import numpy as np
import random
import os
from os import mkdir
from os.path import join
from utils import *
import json


class IndirectReciprocityMultiplexNetworks:

	nodes = []
	def __init__(self, numNodes=100, prob=0.5, avgDegree=2, numGenerations=100, logFreq=1, cost=0.1, benefit=1, layer1=None, layer2=None, socialNorm = 'SternJudging'):
		self.numNodes = numNodes
		self.prob = prob # Rewire probability for Watts-Strogatz
		self.avgDegree = avgDegree
		self.numGenerations = numGenerations
		self.logFreq = logFreq # Generate graphs at every X simulations
		self.cost = cost # Donation Game
		self.benefit = benefit # Donation Game
		self.socialNorm = socialNorm # Global social norm (the entire population follows this)

		self.layer1 = layer1 # Layer1 topology
		self.layer2 = layer2 # Layer2 topology

		self.idIterator = 0
		self.idToIndex = {}  # id:index
		self.initiateNodes()

	def initiateNodes(self):

		initialStrategies = self.calculateInitialStrategies()

		if self.layer1 == 'Random':
			self.layer1 = MultiplexNetwork(self.numNodes, self.avgDegree)

		elif self.layer1 == 'WattsStrogatz':
			self.layer1 = wattsStrogatz(self.numNodes, self.avgDegree, self.prob)

		else:
			print('Wrong layer1 parameter!')
			exit()

		if self.layer2 == 'Random':
			self.layer2 = MultiplexNetwork(self.numNodes, self.avgDegree)

		elif self.layer2 == 'WattsStrogatz':
			self.layer2 = wattsStrogatz(self.numNodes, self.avgDegree, self.prob)

		elif self.layer2 == 'Layer1':
			self.layer2 = self.layer1

		else:
			print('Wrong layer2 parameter!')
			exit()

		self.nodePos = list(self.layer1.nodes())
		for i in range(self.numNodes):
			self.nodes.append({
				'pos': self.nodePos[i], # Redundante?
				'id': self.idIterator,
				'payoff': 0,
				#'reputation': self.calculateInitialReputation(), # Not used because each node has its perception of all others
				'strategy': initialStrategies[i]
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

	def calculateInitialStrategies(self): # Randomly attribute an initial strategy
		initialStrategies = []
		for _ in range(self.numNodes):
			rand = random.randrange(4)
			if rand == 0:
				initialStrategies.append('AllC')
			elif rand == 1:
				initialStrategies.append('AllD')
			elif rand == 2:
				initialStrategies.append('Disc')
			elif rand == 3:
				initialStrategies.append('pDisc')

		return initialStrategies

	def runSimulation(self):
		print('=====    Initiating simulation   ======')
		LogsPerGen=[]
		for i in range(self.numGenerations):
			lg = self.runGeneration()
			lg['generation'] = i
			l = None

			if i % self.logFreq == 0:
				print('== Logging {} =='.format(i))
				#l = self.LogsPerGen(i)
				drawGraph(self.layer1, self.nodes, dir, i)

			self.socialLearning()

			if l != None:
				lg.update(l)

			LogsPerGen.append(lg)

			# print(LogsPerGen)

	def runGeneration(self):

		interactionPairs = getNeighborPairs(self.layer1, self.nodes, self.nodePos)
		actions = []
		for j, pair in enumerate(interactionPairs):
			actions.append(self.runInteraction(pair))
			self.runGossip(pair, actions[-1]) # Update perceptions of the gossipers neighbors in L2

		actionFreq = countFreq(actions)
		cooperationRatio = actionFreq['Cooperate'] if 'Cooperate' in actionFreq.keys() else 0

		# todo - Add stationary fraction of good and bad reputations
		# # print(self.nodes)
		return {'cooperationRatio': cooperationRatio}

	def runInteraction(self, pair): # Donation Game
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
		gossiper = getGossiper(self.layer1, pair[0], self.nodes, self.nodePos)
		updatePerception(self.socialNorm, gossiper, pair[0], pair[1], action)

		neighbors = self.layer2.neighbors(gossiper['pos'])
		for neighbor in neighbors:
			self.nodes[neighbor]['perception'][pair[0]['pos']]['reputation'] = gossiper['perception'][pair[0]['pos']]['reputation']
			# todo - add chance for transmission error to occur
			# to repeat this for the neighbors' neighbors, simply make this function recursive

	def socialLearning(self):
		# social learning where nodes copy another node's strategy with a given probability if
		# that node's payoff is higher
		interactionPairs = getNeighborPairs(self.layer1, self.nodes, self.nodePos)
		beta = 10
		for pair in interactionPairs:
			mine = pair[0]
			partner = pair[1]
			if partner['payoff'] > mine['payoff']:
				prob = 1 / (1 + math.exp(-beta * (partner['payoff'] - mine['payoff'])))
				if probability(prob):
					mine['strategy'] = partner['strategy']

if __name__ == "__main__":
	# Variables used
	initialValues = {
		'numNodes': 100, # Number of nodes
		'prob': 0.25, # Probability of rewiring links (WattsStrogatz)
		'avgDegree': 4,
		'numGenerations': 10000,
		'logFreq': 1000,
		'cost': 0.1, # Cost of cooperation
		'benefit': 1, # Benefit of receiving cooperation
		'layer1': 'WattsStrogatz', # Graph topology: 'WattsStrogatz', 'Random', ...
		'layer2': 'WattsStrogatz', # Graph topology: 'WattsStrogatz', 'Random', 'Layer1' (Layers are equal)
		'socialNorm': 'ImageScoring', # SimpleStanding, ImageScoring, Shunning or SternJudging

	}

	config = initialValues.copy()
	dir = join('output', 'testrun{}'.format(1))

	if not os.path.exists(dir):
		mkdir(dir)

	sim = IndirectReciprocityMultiplexNetworks(**config)
	sim.runSimulation()

	with open(join(dir, 'config.json'), 'w') as fp:
		json.dump(config, fp)

	# todo - add mutation to socialLearning - adopt a random strategy
	# todo - add more graph topologies
	# todo - stationary fraction of good and bad reputations
	# todo - add more plots
	# todo - test the entire program