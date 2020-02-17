import numpy as np
import random
from os import mkdir
from utils import *

class IndirectReciprocityMultiplexNetworks:

	nodes = []

	def __init__(self, numNodes=100):
		self.numNodes = numNodes




		self.idIterator = 0
		#self.idToIndex = {}  # id:index

		self.initiateNodes()

	def initiateNodes(self):

		initialReputations = self.calculateInitialReputations()
		initialStrategies = self.calculateInitialStrategies()

		for i in range(self.numNodes):
			self.nodes.append({
				'id': self.idIterator,
				'payoff': 0,
				'reputation': initialReputations[i],
				'strategy': initialStrategies[i]
			})
			#self.idToIndex[self.idIterator] = len(self.nodes) - 1
			self.idIterator += 1

			# Check reputations and strategies
			#print('Node:', i, '| Reputation:', self.nodes[i]['reputation'], '| Strategy:', self.nodes[i]['strategy'])

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
	

	def reproduce_Social(self):  # social learning where nodes copy another node's strategy with a given probability if that node's payoff is higher
		beta = 10
		for pair in interactionPairs:
			mine = pair[0]
			partner = pair[1]
			if partner['payoff'] > mine['payoff']:
				prob = 1 / (1 + math.exp(-beta * (partner['payoff'] - mine['payoff'])))


if __name__ == "__main__":
	# Variables used
	initialValues = {
		'numNodes' : 100,
	}
	
	config = initialValues.copy()
	sim = IndirectReciprocityMultiplexNetworks(**config)
