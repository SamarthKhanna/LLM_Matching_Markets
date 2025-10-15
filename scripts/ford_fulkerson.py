from copy import deepcopy

class VectorizedGraph():
	
	def __init__(self, graph, size) -> None:
		self.graph = graph
		self.original = deepcopy(self.graph)
		self.ROW = len(self.graph) 
		self.org_graph = [i[:] for i in graph] 
		self.COL = len(graph[0])
		self.size = size
		self.zerov = [0]*self.size
		self.min_cut = set()
		
	def compare_vectors(self, e1, e2):
		i = 0
		# print(e1, e2)
		while i < len(e1):
			if e1[i] > e2[i]:
				return 1
			elif e1[i] < e2[i]:
				return -1
			i += 1
		return 0
	
	def get_vector_min(self, e1, e2):
		# if self.compare_vectors(e1, e2) == 0:
			# print(e1)
		return e1 if self.compare_vectors(e1, e2) < 0 else e2
	
	def add_vectors(self, e1, e2):
		i = 0
		added = []
		# print(e1, e2)
		while i < len(e1):
			added.append(e1[i]+e2[i])
			i += 1
		return added
	
	def subtract_vectors(self, e1, e2):
		i = 0
		subbed = []
		while i < len(e1):
			subbed.append(e1[i]-e2[i])
			i += 1
		return subbed
	
	def get_min_cut(self):
		print("IAM HERE")
		for i, u in enumerate(self.graph):
			for j, v in enumerate(u):
				if v == self.original[j][i] and any(v) and i != j:
					self.min_cut |= {i, j}
					print(f"{i}-{j}: {v} {self.original[j][i]}")
		self.min_cut -= {0, len(self.graph)-1}
		print(self.min_cut)

	def print_graph(self):
		for i, u in enumerate(self.graph):
			print(u)
			# for j, v in enumerate(u):
				# print(f'{i} --> {j}: {v}')
			
	# Function for Depth first search 
	# Traversal of the graph
	def dfs(self, graph,s,visited):
		visited[s]=True
		for i in range(len(graph)):
			if graph[s][i] > self.zerov and not visited[i]:
				self.dfs(graph,i,visited)

	def BFS(self, s, t, parent):

		# Mark all the vertices as not visited
		visited = [False]*(self.ROW)

		# Create a queue for BFS
		queue = []

		# Mark the source node as visited and enqueue it
		queue.append(s)
		visited[s] = True

		# Standard BFS Loop
		while queue:

			# Dequeue a vertex from queue and print it
			u = queue.pop(0)

			# Get all adjacent vertices of the dequeued vertex u
			# If a adjacent has not been visited, then mark it
			# visited and enqueue it
			for ind, val in enumerate(self.graph[u]):
				# print(ind, val)
				if visited[ind] == False and self.compare_vectors(val, self.zerov) == 1:
					# print("ENTER HERE")
					# If we find a connection to the sink node, 
					# then there is no point in BFS anymore
					# We just have to set its parent and can return true
					queue.append(ind)
					visited[ind] = True
					parent[ind] = u
					if ind == t:
						return True

		# We didn't reach sink in BFS starting 
		# from source, so return false
		return False
			
	
	# Returns the maximum flow from s to t in the given graph
	def FordFulkerson(self, source, sink):

		# This array is filled by BFS and to store path
		parent = [-1]*(self.ROW)

		max_flow = [0]*self.size # There is no flow initially

		# Augment the flow while there is path from source to sink
		while self.BFS(source, sink, parent) :

			# Find minimum residual capacity of the edges along the
			# path filled by BFS. Or we can say find the maximum flow
			# through the path found.
			path_flow = [float("Inf")]*(self.ROW-1)
			s = sink
			while(s != source):
				path_flow = self.get_vector_min(path_flow, self.graph[parent[s]][s])
				s = parent[s]

			# Add path flow to overall flow
			max_flow = self.add_vectors(max_flow, path_flow)

			# update residual capacities of the edges and reverse edges
			# along the path
			v = sink
			while(v != source):
				u = parent[v]
				self.graph[u][v] = self.subtract_vectors(self.graph[u][v], path_flow)
				self.graph[v][u] = self.add_vectors(self.graph[v][u], path_flow)
				v = parent[v]

		return max_flow
	
	def minCut(self, source, sink): 
 
		# This array is filled by BFS and to store path 
		parent = [-1]*(self.ROW) 
 
		max_flow = [0]*self.size # There is no flow initially 
		s = sink

		# self.print_graph()
		# print()
		
		# Augment the flow while there is path from source to sink 
		while self.BFS(source, sink, parent) : 
			# Find minimum residual capacity of the edges along the 
			# path filled by BFS. Or we can say find the maximum flow 
			# through the path found. 
			path_flow = [float("Inf")]*(self.ROW-1)
			s = sink
			while(s != source):
				path_flow = self.get_vector_min(path_flow, self.graph[parent[s]][s])
				s = parent[s] 
 
			# Add path flow to overall flow 
			max_flow += path_flow 
 
			# update residual capacities of the edges and reverse edges 
			# along the path 
			v = sink
			while(v != source):
				u = parent[v]
				self.graph[u][v] = self.subtract_vectors(self.graph[u][v], path_flow)
				self.graph[v][u] = self.add_vectors(self.graph[v][u], path_flow)
				v = parent[v]
 
		visited=len(self.graph)*[False]
		self.dfs(self.graph,s,visited)

		# print the edges which initially had weights 
		# but now have 0 weight 
		for i in range(self.ROW): 
			for j in range(self.COL): 
				if self.graph[i][j] == self.zerov and self.org_graph[i][j] > self.zerov and visited[i]: 
					# print(str(i) + " - " + str(j))
					self.min_cut.add(('s' if i == source else i, 't' if j == sink else j))
		# print(f"Min-cut is {self.min_cut}")
		return self.min_cut


# Create a graph given in the above diagram
size = 8	

zerov = [0]*size
infv = [float('inf')]*size

# graph = [
# 	[zerov, [2,-1,-1,-1,0,1,0,0], zerov, zerov, [1,0,-1,-1,1,0,0,0], zerov, zerov],
# 	[zerov, zerov, infv, infv, zerov, zerov, zerov],
# 	[zerov, zerov, zerov, zerov, zerov, infv, [0,0,1,-1,0,0,0,0]],
# 	[zerov, zerov, zerov, zerov, zerov, zerov, [2,0,-1,-1,-1,-2,1,2]],
# 	[zerov, zerov, zerov, zerov, zerov, infv, zerov],
# 	[zerov, zerov, zerov, zerov, zerov, zerov, [1,-2,0,0,0,1,0,0]],
# 	[zerov, zerov, zerov, zerov, zerov, zerov, zerov]
# ]

graph = [
	[zerov, zerov, zerov, [2, 1, -2, -1, -1, -1, 0, 2], zerov, [0, 0, 1, 0, 0, 0, -2, 1], zerov],
	[zerov, zerov, infv, infv, zerov, zerov, [0, 0, 1, 0, -1, -1, -1, 2]],
	[zerov, zerov, zerov, zerov, zerov, zerov, [0, 0, 0, 0, 1, -1, 0, 0]],
	[zerov, zerov, zerov, zerov, zerov, zerov, zerov],
	[zerov, zerov, zerov, zerov, zerov, infv, [0, 0, 0, 1, -1, -1, 0, 1]],
	[zerov, zerov, zerov, zerov, zerov, zerov, zerov],
	[zerov, zerov, zerov, zerov, zerov, zerov, zerov]
]

# g = VectorizedGraph(graph, size)

# source = 0; sink = 6

# print ("The maximum possible flow is ", g.FordFulkerson(source, sink))

# g.get_min_cut()

# g.minCut(source, sink)

# This code is contributed by Neelam Yadav