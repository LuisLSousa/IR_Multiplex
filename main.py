import numpy as np
import random
from os import mkdir
from utils import *

class IndirectReciprocityMultiplexNetworks:

	nodes = []
	def __init__(self, numNodes=100, graphType=None, prob=0.5, avgDegree=2):
		self.numNodes = numNodes
		self.graphType = graphType # Graph topology
		self.prob = prob # Rewire probability for Watts-Strogatz
		self.avgDegree = avgDegree
		self.layer1 = None
		self.layer2 = None

		self.idIterator = 0
		self.idToIndex = {}  # id:index
		self.initiateNodes()

	def initiateNodes(self):

		initialReputations = self.calculateInitialReputations()
		initialStrategies = self.calculateInitialStrategies()
		'''
		for i in range(self.numNodes):
			self.nodes.append({
				'id': self.idIterator,
				'payoff': 0,
				'reputation': initialReputations[i],
				'strategy': initialStrategies[i]
			})
			self.idToIndex[self.idIterator] = len(self.nodes) - 1
			self.idIterator += 1
		'''

		if self.graphType == 'Random':
			self.layer1 = MultiplexNetwork(self.numNodes, self.avgDegree)

		elif self.graphType == 'WattsStrogatz':
			self.layer1 = wattsStrogatz(self.numNodes, self.avgDegree, self.prob)

			self.nodePos = list(self.layer1.nodes())
			for i in range(self.numNodes):
				self.nodes.append({
					'pos': self.nodePos[i], # Redundante?
					'id': self.idIterator,
					'payoff': 0,
					'reputation': initialReputations[i], # Update this so each node has his opinion about all others
					'strategy': initialStrategies[i]
				})

				# Check reputations and strategies
				#print('Node:', self.nodes[i]['id'], 'Pos:', self.nodes[i]['pos'], '| Reputation:', self.nodes[i]['reputation'], '| Strategy:', self.nodes[i]['strategy'])

				self.idToIndex[self.idIterator] = len(self.nodes) - 1
				self.idIterator += 1

		print(self.layer1.nodes())

	def calculateInitialReputations(self):
		initialReputations = ['Good' if random.choice([0,1]) == 0 else 'Bad' for _ in
							 range(self.numNodes)]
		return initialReputations

	def calculateInitialStrategies(self): # Pode ficar mais bonito
		initialStrategies = []
		for _ in range(self.numNodes):
			rand = random.randrange(4)
			if rand == 0:
				initialStrategies.append('SternJudging')
			elif rand == 1:
				initialStrategies.append('SimpleStanding')
			elif rand == 2:
				initialStrategies.append('Shunning')
			elif rand == 3:
				initialStrategies.append('ImageScoring')

		return initialStrategies
	
'''
	def reproduce_Social(self):  # social learning where nodes copy another node's strategy with a given probability if that node's payoff is higher
		beta = 10
		for pair in interactionPairs:
			mine = pair[0]
			partner = pair[1]
			if partner['payoff'] > mine['payoff']:
				prob = 1 / (1 + math.exp(-beta * (partner['payoff'] - mine['payoff'])))
				if probability(prob):
                	mine['strategy'] = partner['strategy']
'''

if __name__ == "__main__":
	# Variables used
	initialValues = {
		'numNodes': 100, # Number of nodes
		'graphType': 'WattsStrogatz', # Graph topology: 'WattsStrogatz', 'Random', ...
		'prob': 1, # Probability of rewiring links (WattsStrogatz)
		'avgDegree': 2,
	}
	# Adicionar graphType1 e 2 para o layer 1 e 2 respectivamente

	config = initialValues.copy()
	sim = IndirectReciprocityMultiplexNetworks(**config)

