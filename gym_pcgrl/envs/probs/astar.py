import numpy as np

class Node:
	def __init__(self, parent=None, position=None):
		self.parent = parent
		self.position = position

		self.g = 0
		self.h = 0
		self.f = 0

	def __eq__(self, other):
		return self.position == other.position

def manhattan_distance(a, b):
	dx = abs(a[0] - b[0])
	dy = abs(a[1] - b[1])
	return dx + dy

def heuristic(node, goal) :
	dx = abs(node.position[0] - goal.position[0])
	dy = abs(node.position[1] - goal.position[1])
	return dx + dy

def aStar(maze, start, end):
	# startNode와 endNode 초기화
	startNode = Node(None, start)
	endNode = Node(None, end)

	# openList, closeList 초기화
	openList = []
	closedList = []

	# openList에 시작 노드 추가
	openList.append(startNode)

	# endNode를 찾을 때까지 실행
	while openList:
		# 현재 노드 지정
		currentNode = openList[0]
		currentIdx= 0

		# 이미 같은 노드가 openList에 있고 f 값이 더 크면
		# currentNode를 OpenList 안에 있는 값으로 교체
		for index, item in enumerate(openList):
			if item.f < currentNode.f:
				currentNode = item
				currentIdx = index

		# openList에서 제거하고 closedList에 추가
		openList.pop(currentIdx)
		closedList.append(currentNode)

		# 현재 노드가 목적지면 current.position 추가하고
		# current의 부모로 이동
		if currentNode == endNode:
			path = []
			current = currentNode
			while current is not None:
				# maze 에 길 표시
				x, y = current.position
				maze[x][y] = 7
				path.append(current.position)
				current = current.parent
			return True, len(path), 0  # reverse

		children = []

		# 4방향으로 인접한 좌표 추가
		for newPosition in [(1,0), (-1, 0), (0, 1), (0, -1)]:

			# Node 위치 업데이트
			nodePosition = (
				currentNode.position[0] + newPosition[0],
				currentNode.position[1] + newPosition[1])

			# 미로 maze index 범위 안에 있어야 함
			within_range_criteria = [
				nodePosition[0] > (len(maze) - 1),
				nodePosition[0] < 0,
				nodePosition[1] > (len(maze[len(maze) - 1]) - 1),
				nodePosition[1] < 0,
			]

			if any(within_range_criteria): # 하나라도 true이면 범위 밖
				continue

			# 장애물이 있으면 다른 위치 불러오기
			if int(maze[nodePosition[0]][nodePosition[1]]):
				continue

			new_node = Node(currentNode, nodePosition)
			children.append(new_node)

		# 자식들 모두 loop
		for child in children:

			# 자식이 closedList에 있으면 continue
			if child in closedList:
				continue

			# f, g, h 값 업데이트
			child.g = currentNode.g + 1
			child.h = heuristic(child, endNode)
			child.f = child.g + child.h

			# 자식이 openList에 있고, g값이 더 크면  continue
			if len([openNode for openNode in openList
					if child == openNode and child.g > openNode.g]) > 0:
				continue

			openList.append(child)

	# 길이 없는 경우 도착 지점까지 가장 가까운 거리를 계산
	# todo : 일반화를 위해 바꿀 필요 있음(ex. start, end 거리 가깝지만 막혀있는 경우)
	min_length = 10e8

	for node in closedList:
		length = heuristic(node, endNode)
		if length < min_length:
			min_length = length
	return False, 0, min_length

if __name__ == '__main__':

	# 1은 장애물
	maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

	start = (0, 0)
	end = (9, 9)

	arrived, path_length, not_arrived_length = aStar(maze, start, end)
	print(np.asarray(maze))
	print(f'arrived : {arrived}, path_length : {path_length}, not_arrived_length : {not_arrived_length}')