import networkx as nx
import numpy as np
import argparse
import time
import math
import random
import progressbar
import nxmetis
import sys
import operator
import secrets
from itertools import islice
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib
from scipy import sparse
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power
import scipy.sparse
import scipy
font = {'size':20}
matplotlib.rc('font', **font)
matplotlib.use('Agg')
def parse_args():
	parser = argparse.ArgumentParser(description="motif-aware parition")

	parser.add_argument('--truth', nargs='?', default='BK.txt', help='original network with groundtruth edge weights')
	
	parser.add_argument('--triangle', nargs='?', default='BK.txt', help='original network with triangle edge weights')

	parser.add_argument('--output', nargs='?', default='',help='default output directory')
	
	parser.add_argument('--motif', nargs='?', default='q2',
			help='q3:clique4,q7:clique5, q11:clique6')
	parser.add_argument('--iteration', type=int, default='10',help='iterations of partitions loop')
	
	parser.add_argument('--parts', type=int, default='10',
			                          help='the number of partitions with Fennel')
	parser.add_argument('--sample', type=int, default='10000',
						help='the number of samples per batch')	
	
	parser.add_argument('--startingError', type=float, default='0.1',help='estimation error in each iteration')

	parser.add_argument('--ec', type=int, default='3',help='the threshold count for the estimation to be below the estimation error')

	parser.add_argument('--stopIteration', type=int, default='3',help='the number of iterations involving probability calculation')

	parser.add_argument('--nf', type=int, default='1',help='the strength of neighborhood factor in each iteration')

	parser.add_argument('--pf', type=int, default='1',help='whether involving partiton factor computation.')

	parser.add_argument('--index', type=int, default='50',help='the starting index for sweeping over in MAPPR.')

	parser.add_argument('--condthreshold', type=float, default='0.1',help='conductance below this threshold will be treated as 1.')

	parser.add_argument('--baseline', type=int, default='1',
		help='baseline: 1.Fennel 2.MAPPR')

	return parser.parse_args()

class graph_processor(object):
	def __init__(self,args):
		
		self.samples=args.sample
		self.G=nx.Graph()
		self.query=nx.Graph()
		self.parts=args.parts
		self.instance=nx.Graph()
		self.stopIteration=args.stopIteration
		self.nf=args.nf
		self.querySize=0
		self.tolerance=10
		self.iteration=args.iteration+1
		self.baseline=args.baseline
		self.motif=args.motif
		self.readGraphInfo(args.truth,args.triangle)
		self.current_iter=0
		self.minimum=-sys.maxsize
		self.ordering=self.getOrdering()
		self.preprocess=0
		self.partitioning=0
		self.error=args.startingError
		self.ec=args.ec
		self.pf=args.pf
		self.startingIndex=args.index
		self.condthreshold=args.condthreshold
		
		self.timeOutput=args.output+"b"+str(self.baseline)+"_time.txt"
		self.condOutput=args.output+"b"+str(self.baseline)+"_cond.txt"
		self.modOutput=args.output+"b"+str(self.baseline)+"_mod.txt"
		self.indexOutput=args.output+"b"+str(self.baseline)+"_mod.txt"

		self.graphsize=self.G.size(weight='trueweight')
		self.seed=123
		
		self.uniformSample=1/len(self.G.edges())
		self.probInit()
		random.seed(self.seed)

		self.baseLineIndex=-1
	
		self.baselineTime=0.0
		self.adaptTime=[0.0 for i in range(args.iteration+1)]
	
		self.baselineCond=[0.0 for i in range(args.iteration+1)]
		self.adaptCond=[0.0 for i in range(args.iteration+1)]
		self.adaptIndex=[0 for i in range(args.iteration+1)]

		self.baselineMod=[0.0 for i in range(args.iteration+1)]
		self.adaptMod=[0.0 for i in range(args.iteration+1)] 


		if (self.baseline==1):
			self.gamma=1.5
			self.upsilon=5
			print('compare fennel clustering')
			self.alpha=len(self.G.edges())*(pow(self.parts,self.gamma-1)/pow(len(self.G.nodes()),self.gamma))
			self.partThreshold=int(self.upsilon*len(self.G.nodes())/self.parts)
			self.penalty=self.penalty()
			self.compFennelClustering()
			label='Ratio against Fennel'

		if (self.baseline==2):
			print ('compare MAPPR clustering')
			self.sweep=len(self.G)-1
			self.target=self.sortedList[0][0]
			print ("target ",str(self.target))
						
			self.compareMAPPR()
			label='Ratio against MAPPR'



		for i in range(self.iteration-1):
			
			print ("adpative sampling")
			self.sampling(1)
			self.current_iter+=1

		self.writeFile()
		

	def writeFile(self):


		f=open(self.timeOutput,"a")
		temp="baseLine "+str(self.baselineTime)
		f.write(temp+"\n")
		temp="AS "
		for i in self.adaptTime:
			temp=temp+str(i)+" "
		f.write(temp+"\n")

		if (self.baseline==2):
			f=open(self.condOutput,"a")
			temp="startingIndex "+str(self.baseLineIndex)+" baseLine "+str(self.baselineCond)
			f.write(temp+"\n")
			temp="AS "
			for i in self.adaptCond:
				temp=temp+str(i)+" "
			f.write(temp+"\n")
			
			temp="Index "
			for i in self.adaptIndex:
				temp=temp+str(i)+" "
			f.write(temp+"\n")	


		if (self.baseline==1):
			f=open(self.modOutput,"a")
			temp="baseLine "+str(self.baselineMod)
			f.write(temp+"\n")
			temp="AS "
			for i in self.adaptMod:
				temp=temp+str(i)+" "
			f.write(temp+"\n")					


	def probInit(self):
		for u,v,data in self.G.edges(data=True):
				data['current_prob']=self.uniformSample
	
	def readGraphInfo(self,groundtruth,triangleTruth):
		print ('load graphs')
					
		with open(groundtruth) as f:
			
			for line in f:
				strlist = line.split()

				if (len(strlist)==1):
					continue
				n1 = int(strlist[0])+1
				n2 = int(strlist[1])+1
				t = int(strlist[2])
				if (self.baseline==1):
					t+=1

				self.G.add_edge(n1, n2,ratio=1,current_prob=0.0,current_weight=0.0,totalWeight=1,cn=0,trueweight=t)


		nx.set_node_attributes(self.G, 'truedegree', 0)
		nx.set_node_attributes(self.G, 'degree', 0)
		nx.set_node_attributes(self.G, 'partitionID', 0)
		nx.set_node_attributes(self.G, 'degreex', 0)
		nx.set_node_attributes(self.G, 'neighbors', [])
		
		for u,v, data in self.G.edges(data=True):
			self.G.node[u]['truedegree']+=data['trueweight']
			self.G.node[v]['truedegree']+=data['trueweight']

		for n in self.G.nodes():
			self.G.node[n]['neighbors']=self.G.neighbors(n)
			self.G.node[n]['degreex']=self.G.degree(n)


		self.edges=list(self.G.edges())		
		self.sortedList=[(n,degree) for n,degree in self.G.degree().items()]
		self.sortedList=sorted(self.sortedList,key=lambda x: x[1], reverse=True)
		#self.nodeOrdering=sorted(list(self.G.nodes()))
		
		self.attention=[]

		for i in range(10000):
			self.attention.append(self.sortedList[i][0])

		with open(triangleTruth) as f:
			
			for line in f:
				strlist = line.split()

				if (len(strlist)==1):
					continue
				n1 = int(strlist[0])+1
				n2 = int(strlist[1])+1
				t = int(strlist[2])+1
				self.G[n1][n2]['cn']=math.log(t,10)



		if (self.motif=='q0'):
			self.query.add_edge(1,2)
			self.query.add_edge(2,3)
			self.query.add_edge(3,1)

		if (self.motif=='q1'):
			self.query.add_edge(1,2)
			self.query.add_edge(2,3)
			self.query.add_edge(3,4)
			self.query.add_edge(4,1)

		if (self.motif=='q2'):
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)
			self.query.add_edge(2,3)
			self.query.add_edge(3,4)

		#clique4
		if (self.motif=='q3'):
			self.query=nx.complete_graph(4)
		
		if (self.motif=='q4'):
			self.query.add_edge(1,2)
			self.query.add_edge(1,5)
			self.query.add_edge(2,3)
			self.query.add_edge(3,1)
			self.query.add_edge(3,4)			
			self.query.add_edge(4,5)			

		if (self.motif=='q5'):
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)
			self.query.add_edge(1,5)
			self.query.add_edge(1,6)				
			self.query.add_edge(2,3)
			self.query.add_edge(3,4)		
			self.query.add_edge(4,5)
			self.query.add_edge(5,6)			
			
		#clique5
		if (self.motif=='q7'):
			self.query=nx.complete_graph(5)
		
		if (self.motif=='q6'):
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)
			self.query.add_edge(3,2)
			self.query.add_edge(3,4)
			self.query.add_edge(3,5)
			self.query.add_edge(4,5)
			self.query.add_edge(4,2)

		if (self.motif=='q8'):
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)
			self.query.add_edge(3,2)
			self.query.add_edge(3,5)
			self.query.add_edge(4,5)
			self.query.add_edge(4,2)
	
		if (self.motif=='q9'):
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)
			self.query.add_edge(2,3)
			self.query.add_edge(3,4)
			self.query.add_edge(3,5)
			self.query.add_edge(4,5)

		if (self.motif=='q10'):
			self.query.add_edge(1,2)
			self.query.add_edge(2,3)
			self.query.add_edge(3,4)
			self.query.add_edge(4,5)
			self.query.add_edge(5,1)
			self.query.add_edge(1,4)
			self.query.add_edge(5,3)
			self.query.add_edge(4,2)

		#clique6
		if (self.motif=='q11'):
			self.query=nx.complete_graph(6)

		if (self.motif=='q12'):
			self.query.add_edge(1,2)
			self.query.add_edge(2,3)
			self.query.add_edge(3,4)
			self.query.add_edge(4,5)
			self.query.add_edge(5,1)
			self.query.add_edge(1,4)
			self.query.add_edge(5,2)
			self.query.add_edge(5,3)
			self.query.add_edge(4,2)

		if (self.motif=='q13'):
			self.query.add_edge(1,3)
			self.query.add_edge(1,5)
			self.query.add_edge(2,5)
			self.query.add_edge(2,6)
			self.query.add_edge(3,6)
			self.query.add_edge(3,5)
			self.query.add_edge(3,4)
			self.query.add_edge(4,6)
			self.query.add_edge(5,6)

		if (self.motif=='q14'):
			self.query.add_edge(0,1)
			self.query.add_edge(0,2)
			self.query.add_edge(0,3)
			self.query.add_edge(0,4)
			self.query.add_edge(1,2)
			self.query.add_edge(3,4)
			

		if (self.motif=='q15'):
			self.query.add_edge(0,1)
			self.query.add_edge(0,2)
			self.query.add_edge(0,3)
			self.query.add_edge(0,4)
			self.query.add_edge(0,5)
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)
			self.query.add_edge(1,5)

		if (self.motif=='q16'):
			self.query.add_edge(0,1)
			self.query.add_edge(0,2)
			self.query.add_edge(0,3)
			self.query.add_edge(0,4)
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)

		if (self.motif=='q17'):
			self.query.add_edge(0,1)
			self.query.add_edge(0,2)
			self.query.add_edge(0,3)
			self.query.add_edge(1,2)
			self.query.add_edge(3,4)
			self.query.add_edge(3,5)	
			self.query.add_edge(4,5)	
		
		if (self.motif=='q18'):
			self.query.add_edge(0,1)
			self.query.add_edge(0,2)
			self.query.add_edge(1,2)
			self.query.add_edge(1,3)
			self.query.add_edge(1,4)
			self.query.add_edge(2,3)
			self.query.add_edge(2,5)
			self.query.add_edge(3,4)
			self.query.add_edge(3,5)		

		self.querySize=len(self.query.nodes())	
		self.queryEdge=self.query.size()
	
				
		
	
	def getOrdering(self):

		if (self.motif=='q0'):
			return 3
		if (self.motif=='q1'):
			return 8		
		if (self.motif=='q2'):
			return 10
		if (self.motif=='q3'):
			return 12
		if (self.motif=='q4'):
			return 30
		if (self.motif=='q5'):
			return 172
		if (self.motif=='q6'):
			return 46
		if (self.motif=='q7'):
			return 60
		if (self.motif=='q8'):
			return 40
		if (self.motif=='q9'):
			return 38
		if (self.motif=='q10'):
			return 48
		if (self.motif=='q11'):
			return 360
		if (self.motif=='q12'):
			return 54
		if (self.motif=='q13'):
			return 168
		if (self.motif=='q14'):
			return 28
		if (self.motif=='q15'):
			return 216			
		if (self.motif=='q16'):
			return 42	
		if (self.motif=='q17'):
			return 60			
		if (self.motif=='q18'):
			return 168			

	def pagerank_scipy(self, flag, alpha=0.85, personalization=None,
					   max_iter=100, tol=1.0e-8,
					   dangling=None):
		start=time.time()
		N = len(self.G)
		if N == 0:
			return {}

		nodelist = sorted(self.G.nodes())
		if (flag==0):
			M = nx.to_scipy_sparse_matrix(self.G, nodelist=nodelist, weight='trueweight',dtype=float)	
		else:	
			M = nx.to_scipy_sparse_matrix(self.G, nodelist=nodelist, weight='totalWeight',dtype=float)	

		S = scipy.array(M.sum(axis=1)).flatten()

		S[S != 0] = 1.0 / S[S != 0]
		Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
		M = Q * M

		# initial vector
		x = scipy.repeat(1.0 / N, N)

		# Personalization vector
		if personalization is None:
			p = scipy.repeat(1.0 / N, N)
		else:
			missing = set(nodelist) - set(personalization)
			if missing:
				raise NetworkXError('Personalization vector dictionary '
									'must have a value for every node. '
									'Missing nodes %s' % missing)
			p = scipy.array([personalization[n] for n in nodelist],
							dtype=float)
			p = p / p.sum()

		# Dangling nodes
		if dangling is None:
			dangling_weights = p
		else:
			missing = set(nodelist) - set(dangling)
			if missing:
				raise NetworkXError('Dangling node dictionary '
									'must have a value for every node. '
									'Missing nodes %s' % missing)
			# Convert the dangling dictionary into an array in nodelist order
			dangling_weights = scipy.array([dangling[n] for n in nodelist],
										   dtype=float)
			dangling_weights /= dangling_weights.sum()
		is_dangling = scipy.where(S == 0)[0]
		end=time.time()
		self.preprocess=end-start
		#print ("preprocess ",self.preprocess,"\n")
		# power iteration: make up to max_iter iterations
		for _ in range(max_iter):
			xlast = x
			x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
				(1 - alpha) * p
			# check convergence, l1 norm
			err = scipy.absolute(x - xlast).sum()
			if err < N * tol:
				return dict(zip(nodelist, map(float, x)))
		raise NetworkXError('pagerank_scipy: power iteration failed to converge '
							'in %d iterations.' % max_iter)


########################## MAPPR ##############################

	def ppr(self,flag,target, alpha=0.85, epsilon=10e-8, iters=100):
		
		
		
		pref = {}
		for node in self.G.nodes():
			if (node==target):
				pref.update({node:1.0})
			else:
				pref.update({node:0.0})
		
		return self.pagerank_scipy(flag,personalization=pref)

	def min_cond_cut(self,flag,rank):
		self.sigma=0.0
		self.vol1=0.0
		
		
		if (flag==0):
			self.vol2=self.G.size(weight='trueweight')*2
		else:
			self.vol2=self.G.size(weight='totalWeight')*2
		
		self.nbunch=[]
		self.nbunchset=set()
		
		def conductance(flag):
			add_node=self.nbunch[-1]
			for n in self.G.neighbors(add_node):
					if n not in self.nbunchset:
						if (flag==0):
							self.sigma += self.G[add_node][n]['trueweight']
						else:
							self.sigma += self.G[add_node][n]['totalWeight']
					else:
						if (flag==0):
							self.sigma -= (self.G[add_node][n]['trueweight'])
						else:
							self.sigma -= (self.G[add_node][n]['totalWeight'])
							
			
			if (flag==0):
				temp=self.G.degree(add_node,weight='trueweight')
			else:
				temp=self.G.degree(add_node,weight='totalWeight')

			self.vol1+=temp
			self.vol2-=temp
			cond= ((self.sigma+1) / min(self.vol1+1, self.vol2+1))
			
			if (cond<=self.condthreshold):
				return 1.0
			else:
				return cond
				
			
		conductance_list = []
		limit = self.sweep

		for i in range(0, limit):
			self.nbunch.append(rank[i][0])
			c = (i, conductance(flag=flag))
			self.nbunchset.add(rank[i][0])
			conductance_list.append(c)
		
		mincond=sys.maxsize
		topcond={}
		threshold=self.condthreshold
		for index, value in conductance_list:
			if (flag==0):
				if (value<mincond):
					finalIndex=index
					mincond=value
					topcond[index]=value
			else:
				if (index>=self.startingIndex):
					

					#value=value+int(math.log10(index))*0.1
					if (value<threshold):
						continue
					
					if (value<mincond):
						finalIndex=index
						mincond=value
						topcond[index]=value


		topcond = sorted(topcond.items(), key=operator.itemgetter(1))
		topcond=list(islice(topcond, 10))
		self.cond=mincond
		estimate=mincond
		finalList=self.nbunch[0:finalIndex+1]
		if (flag==0):
			self.finalIndex=finalIndex
		if (flag==0):
			for i in range(len(self.baselineCond)):
				self.baselineCond[i]=self.cond
				self.trueCond=self.cond
				self.baseLineIndex=finalIndex
			print ("conductance:",self.cond," finalindex,",finalIndex)
			print (topcond)
		else:
			self.sigma=0.0
			self.vol1=0.0
			self.vol2=self.G.size(weight='trueweight')*2
			self.nbunch=[]
			self.nbunchset=set()
			for i in range(finalIndex+1):
				self.nbunch.append(finalList[i])
				self.cond=conductance(flag=0)
				self.nbunchset.add(finalList[i])

				if (i==finalIndex):
					self.adaptCond[self.current_iter]+=self.cond
					self.adaptIndex[self.current_iter]=finalIndex
					print ("conductance:",self.cond," estimate:",estimate," finalIndex:",finalIndex," TrueCond:",str(self.trueCond))
					print (topcond)
		partitions=[set() for i in range(2)]
		finalList=set(finalList)
		s=time.time()
		for n in self.G.nodes():
			if (n in finalList):
				partitions[0].add(n)
				self.G.node[n]['partitionID']=0
			else:
				partitions[1].add(n)
				self.G.node[n]['partitionID']=1
		e=time.time()
		self.preprocess+=(e-s)
		return partitions

	def MAPPR(self,flag):
		s=time.time()
		rank=self.ppr(flag,self.target, alpha=0.85, epsilon=10e-8, iters=100)
		rank=sorted(rank.items(), key=itemgetter(1), reverse=True)
		partitions=self.min_cond_cut(flag, rank=rank)
		e=time.time()
		self.partitioning=(e-s)-self.preprocess
		return partitions
	
	def compareMAPPR(self):
		start=time.time()
		print ("Baseline performance:")
		partitions=self.MAPPR(0)
		end=time.time()
		self.baselineTime=self.partitioning
		self.quality(partitions,0)
		print('baseline running time is (s): ',self.partitioning)


	####################  Fennel ########################	
	def penalty(self):
		#print ('load penalty precomputation')
		
		penalty=[0 for i in range(int(self.partThreshold)+1)]
		intraCost=[0 for i in range(int(self.partThreshold)+1)]

		for index in range(len(intraCost)):
			intraCost[index]=self.alpha*pow(index,self.gamma)

		for index in range(len(penalty)):
		
			if (index==0):
				penalty[index]=0.0
			else:
				penalty[index]=intraCost[index]-intraCost[index-1]
			#print (penalty[index])
		return penalty

	def Fennel(self, flag):
		s=time.time()
		partitions=[set() for i in range(self.parts)]
		for node,degree in self.sortedList:
			maxgain=self.minimum
			count=-1
			for part in partitions:
				count+=1
				if (len(part)>=int(self.partThreshold)):
					continue
				intra_gain=0
				for neighbor in self.G.neighbors(node):
					if (neighbor in part):
						if (flag==0):
							intra_gain+=(self.G[node][neighbor]['trueweight'])
						else:
							intra_gain+=(self.G[node][neighbor]['totalWeight'])
											

				total_gain=intra_gain-self.penalty[len(part)+1]
				#total_gain=intra_gain
				if (total_gain>maxgain):
					maxgain=total_gain
					index=count
			partitions[index].add(node)
			self.G.node[node]['partitionID']=index
		e=time.time()
		self.partitioning=e-s
		return partitions	


	def compFennelClustering(self):
		print ('baseline fennel clustering')
		start=time.time()
		partitions=self.Fennel(0)
		end=time.time()
		self.baselineTime=end-start
		self.quality(partitions,0)
		print('baseline running time is (s): ',end-start)

	##########################    Metis   ############################# 
	def normalize(self, values, actual_bounds, desired_bounds):
		return [int(desired_bounds[0] + (x - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0])) for x in values]

	
	def Metis(self,flag):
		s=time.time()			

		(cut,parts)=nxmetis.partition(self.G,edge_weight='totalWeight',nparts=self.parts,recursive=True)
		e=time.time()
		self.partitioning=e-s
		print ("partition cost:",e-s)

		partitions=[set() for i in range(self.parts)]

		for i in range(len(parts)):
			for node in parts[i]:
				#record the index of the partition the node belong to.
				self.G.node[node]['partitionID']=i
				partitions[i].add(node)

		return partitions
			
	def compareMetis(self):
		
		print ("baseline metis")
		s=time.time()
		(cut,parts)=nxmetis.partition(self.G,edge_weight='trueweight',nparts=self.parts,recursive=True)
		e=time.time()
		self.baselineTime=e-s
		print ("partition cost:",e-s)
		partitions=[set() for i in range(self.parts)]

		print ("edge_cut by library:",cut)

		for i in range(len(parts)):
			for node in parts[i]:
				self.G.node[node]['partitionID']=i
				partitions[i].add(node)

		self.quality(partitions,0)
###################### instance probs########################
	def probCalculate(self,order):
		prob=1
		
		#outDegree=self.G.degree[order[0]]+self.G.degree[order[1]]
		outDegree=self.G.node[order[0]]['degreex']+self.G.node[order[1]]['degreex']
		inEdge=1
		for i in range(len(order)):
			#skip the first two nodes.
			if (i!=0 and i!=1):
				links=0
				for node in order[0:i]:
					if self.instance.has_edge(node,order[i]):
						links+=1
				#the prob of node i,i+1,i+2,..., being sampled in the neighborhood of the gradually growing subgraph.
				prob=prob*( links/ (outDegree - 2*inEdge )  )
				outDegree+=self.G.node[order[i]]['degreex']
				inEdge+=links
		return prob
		

	#compute the prob for sampling this subgraph instance
	def computeWeight(self, order, flag):
		
		sample_prob=self.G[order[0]][order[1]]['current_prob']
		sample_prob=sample_prob*self.probCalculate(order)
		weight=1/sample_prob
		return weight

	def quality(self,partitions,flag):

		Q=0

		if (self.baseline==1):
			denominator=self.graphsize*2
			for u,v,data in self.G.edges(data=True):
				if (self.G.node[u]['partitionID']==self.G.node[v]['partitionID']):
					Q+=(data['trueweight']- self.G.node[u]['truedegree']*self.G.node[v]['truedegree']/denominator)
			Q=Q/(self.graphsize*2)
			

		if (flag==0):
			if (self.baseline==1):
				for i in range(len(self.baselineMod)):
					self.baselineMod[i]=Q
					self.trueMod=Q
				print ("modularity: ", Q, "True modularity:", str(self.trueMod))					
		else:

			if (self.baseline==1):
				self.adaptMod[self.current_iter]+=Q
				print ("modularity: ", Q, "True modularity:", str(self.trueMod))		


	def importance(self,intraWeight,interWeight,degree):
		ratio= abs(intraWeight-interWeight)/(degree+0.0000001)
		return (math.exp(-ratio)+1)
	

	def sampling(self,flag):
		start=time.time()
		normalizeTime=0
		self.totalsamples=0
		batch=0
		error=1000000000000
		pweight=0
		cweight=0
		errorcount=0
		allProbs=[]
		samplingTime=0
		consecutive=False
		for n1,n2,data in self.G.edges(data=True):
			allProbs.append(data['current_prob'])

		while (errorcount<self.ec):
			batch+=1
			s=time.time()				
			if (self.current_iter!=0):
				random.seed(self.current_iter*10+self.seed+batch*100)
				sampled_edges=random.choices(self.edges,weights=allProbs,k=self.samples)
			
			else :
				random.seed(self.current_iter*10+self.seed+batch*100)
				sampled_edges=random.choices(self.edges,k=self.samples)
			e=time.time()
			normalizeTime+=(e-s)

			
			#sample a subgraph for each sample edge and check if the sampled subgraph is isomorphic to the query graph.
			for sample in sampled_edges:					
				q7=True
				self.instance=nx.Graph()
				self.instanceEdge=1
				self.instance.add_edge(sample[0],sample[1])
				order=[sample[0],sample[1]]
				existed=set()
				existed.add(sample[0])
				existed.add(sample[1])

				length=0
				endIndex=[0]
				
				for jj in range(self.querySize-2):

					add_node=-1
					if (jj==0):
						for node in order:
							#length+=self.G.degree[node]
							length+=self.G.node[node]['degreex']
							endIndex.append(length)
					else:
						#length+=self.G.degree[order[-1]]
						length+=self.G.node[order[-1]]['degreex']
						endIndex.append(length)
					
					founded=False
					
					for attempt in range(self.tolerance):
						number=random.randint(1,length)
						for i in range(len(endIndex)):
							if (i!=0):
								if (number<=endIndex[i]):
									pick=i-1
									index=number-endIndex[i-1]-1
									break
						
						add_node=self.G.node[order[pick]]['neighbors'][index]
						
						if (add_node in existed):
							continue
						founded=True
						break
				

					if not founded:
						candidate=[]
						for node in order:
							for neighbor in self.G.neighbors(node):
								if neighbor not in existed:
									candidate.append(neighbor)
						if (len(candidate)==0):
							break
						add_node=random.choice(candidate)



					for node in order:
						if (self.G.has_edge(node,add_node)):
							self.instance.add_edge(add_node,node)
							self.instanceEdge+=1
						
						else:
							if (self.motif=='q7'):
								q7=False
								break
						
					if (not q7):
						break		


					order.append(add_node)
					existed.add(add_node)

				#here it uses VF2 algorithm. Replace it if there is a better/faster algrithm in C++
				iso=False
				#isos=time.time()
				if (self.queryEdge==self.instanceEdge):
					if (self.motif=='q0' or self.motif=='q7'):
						iso=True
						
					else:	
						iso=nx.is_isomorphic(self.query,self.instance)

				if iso:
					#update weight for each edge. 
					sample_weight=self.computeWeight(order,flag)

					for u,v in self.instance.edges():
						self.G.get_edge_data(u,v)['current_weight']+=sample_weight
						if(batch==1):
							pweight+=sample_weight
						else:
							cweight+=sample_weight
			
			
			self.totalsamples+=self.samples
			if (batch!=1):
				tempweight=pweight+cweight
				curcount=tempweight/self.totalsamples
				pcount=pweight/(self.totalsamples-self.samples)
				error=abs(curcount-pcount)/curcount
				pweight=tempweight
				if (error<self.error):
					if (consecutive):
						errorcount+=1
					else:
						errorcount=1
						consecutive=True
				else:
					consecutive=False		

		samplingTime=time.time()-start
		print ("total batch:",str(batch))

		temp=1.0/(self.totalsamples*self.ordering)

		ss=time.time()
		for u,v,data in self.G.edges(data=True):
			if (data['current_weight']>0):
				data['current_weight']=data['current_weight']*temp
				self.G.node[u]['degree']+=data['current_weight']
				self.G.node[v]['degree']+=data['current_weight']				
				data['totalWeight']+=data['current_weight']

		tt=time.time()
		averaging=tt-ss

		print ("partition begin")
			
		if (self.baseline==1):
			partitions=self.Fennel(flag)			

		if (self.baseline==2):
			partitions=self.MAPPR(flag)

		factors=time.time()
		if (self.pf==1):
			progress=progressbar.ProgressBar(maxval=len(self.attention),widgets=['partition factor calculation: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()])
			progress.start()
			clock=0

			for target in self.attention:
				
				maxRatio=-sys.maxsize
				size=self.G.node[target]['degreex']
				intraID=self.G.node[target]['partitionID']
				interNeighbor=[[[],0] for i in range(self.parts)]
				intraID=self.G.node[target]['partitionID']
				
				for neighbor in self.G.neighbors(target):
					interNeighbor[self.G.node[neighbor]['partitionID']][0].append(neighbor)
					interNeighbor[self.G.node[neighbor]['partitionID']][1]+=self.G[target][neighbor]['totalWeight']

				intraWeight=interNeighbor[intraID][1]

				for index in range(self.parts):
					if (index!=intraID):
						if (interNeighbor[index][1]==0):
							continue
						
						ratio=self.importance(intraWeight,interNeighbor[index][1],self.G.node[target]['degree'])
						
						if (ratio>maxRatio):
							maxRatio=ratio

						for neighbor in interNeighbor[index][0]:
							self.G[target][neighbor]['ratio']=max(self.G[target][neighbor]['ratio'],ratio)	

				for neighbor in interNeighbor[intraID][0]:
					self.G[target][neighbor]['ratio']=max(self.G[target][neighbor]['ratio'],maxRatio)			
				clock+=1
				progress.update(clock)
			progress.finish()
		partitionFactor=time.time()-factors
			
		end=time.time()

		if (self.current_iter<self.stopIteration):
			self.adaptTime[self.current_iter]=(end-start)-self.preprocess-normalizeTime
		else:
			self.adaptTime[self.current_iter]=(end-start)-self.preprocess-normalizeTime-self.partitioning-partitionFactor

		if (self.current_iter!=0):
			self.adaptTime[self.current_iter]+=self.adaptTime[self.current_iter-1]
		print ('iteration: ',self.current_iter+1,' cumulate running time: ',self.adaptTime[self.current_iter],
			"normlization: ",str(normalizeTime),"partitioning: ",str(self.partitioning),
			"averaging:",str(averaging),"sampling:",str(samplingTime))

		print("quality evaluation")
		self.quality(partitions,flag)

		print ("update probability")
		for u,v,data in self.G.edges(data=True):
			if (flag==1):
				if (self.current_iter<self.stopIteration):
					data['current_prob']=data['current_prob']*(self.G[u][v]['cn']**self.nf)*data['ratio']
					
			data['ratio']=1
			data['current_weight']=0

		nx.set_node_attributes(self.G, 'degree', 0)
		print ("\n")


def main(args):
	processor=graph_processor(args)

if __name__ == "__main__":
	args = parse_args()
	main(args)


