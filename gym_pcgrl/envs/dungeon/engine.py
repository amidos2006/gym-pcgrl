from queue import PriorityQueue

directions = [{"x":-1, "y":0}, {"x":1, "y":0}, {"x":0, "y":-1}, {"x":0, "y":1}]
class Node:
    balance = 0.5
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = 0
        if self.parent != None:
            self.depth = parent.depth + 1

    def getChildren(self):
        children = []
        for d in directions:
            childState = self.state.clone()
            childState.update(d["x"], d["y"])
            children.append(Node(childState, self, d))
        return children

    def getKey(self):
        return self.state.getKey()

    def getCost(self):
        return self.depth

    def getHeuristic(self):
        return self.state.getHeuristic()

    def checkWin(self):
        return self.state.checkWin()

    def getActions(self):
        actions = []
        current = self
        while(current.parent != None):
            actions.insert(0,current.action)
            current = current.parent
        return actions

    def __str__(self):
        return str(self.depth) + "," + str(self.state.getHeuristic()) + "\n" + str(self.state)

    def __lt__(self, other):
        return self.getHeuristic()+Node.balance*self.getCost() < other.getHeuristic()+Node.balance*other.getCost()

class Agent:
    def getSolution(self, state, maxIterations):
        return []

class BFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        iterations = 0
        bestNode = None
        queue = [Node(state.clone(), None, None)]
        visisted = set()
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1
            current = queue.pop(0)
            if current.checkWin():
                return current.getActions(), current, iterations
            if current.getKey() not in visisted:
                if bestNode == None or current.getHeuristic() < bestNode.getHeuristic():
                    bestNode = current
                elif current.getHeuristic() == bestNode.getHeuristic() and current.getCost() < bestNode.getCost():
                    bestNode = current
                visisted.add(current.getKey())
                queue.extend(current.getChildren())
        return bestNode.getActions(), bestNode, iterations

class DFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        iterations = 0
        bestNode = None
        queue = [Node(state.clone(), None, None)]
        visisted = set()
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1
            current = queue.pop()
            if current.checkWin():
                return current.getActions(), current, iterations
            if current.getKey() not in visisted:
                if bestNode == None or current.getHeuristic() < bestNode.getHeuristic():
                    bestNode = current
                elif current.getHeuristic() == bestNode.getHeuristic() and current.getCost() < bestNode.getCost():
                    bestNode = current
                visisted.add(current.getKey())
                queue.extend(current.getChildren())
        return bestNode.getActions(), bestNode, iterations

class AStarAgent(Agent):
    def getSolution(self, state, balance=1, maxIterations=-1):
        iterations = 0
        bestNode = None
        Node.balance = balance
        queue = PriorityQueue()
        queue.put(Node(state.clone(), None, None))
        visisted = set()
        while (iterations < maxIterations or maxIterations <= 0) and queue.qsize() > 0:
            iterations += 1
            # queue = sorted(queue, key=lambda node: balance*node.getCost() + node.getHeuristic())
            current = queue.get()
            if current.checkWin():
                return current.getActions(), current, iterations
            if current.getKey() not in visisted:
                if bestNode == None or current.getHeuristic() < bestNode.getHeuristic():
                    bestNode = current
                elif current.getHeuristic() == bestNode.getHeuristic() and current.getCost() < bestNode.getCost():
                    bestNode = current
                visisted.add(current.getKey())
                children = current.getChildren()
                for c in children:
                    queue.put(c)
        return bestNode.getActions(), bestNode, iterations

class State:
    def __init__(self):
        self.solid = []
        self.enemies = []
        self.treasures = []
        self.potions = []
        self.player = None
        self.door = None

    def stringInitialize(self, lines):
        # clean the input
        for i in range(len(lines)):
            lines[i]=lines[i].replace("\n","")

        for i in range(len(lines)):
            if len(lines[i].strip()) != 0:
                break
            else:
                del lines[i]
                i-=1
        for i in range(len(lines)-1,0,-1):
            if len(lines[i].strip()) != 0:
                break
            else:
                del lines[i]
                i+=1

        #get size of the map
        self.width=0
        self.height=len(lines)
        for l in lines:
            if len(l) > self.width:
                self.width = len(l)

        #set the level
        for y in range(self.height):
            l = lines[y]
            self.solid.append([])
            for x in range(self.width):
                if x > len(l)-1:
                    self.solid[y].append(False)
                    continue
                c=l[x]
                if c == "#":
                    self.solid[y].append(True)
                else:
                    self.solid[y].append(False)
                    if c == "@":
                        self.player={"x":x, "y":y, "health":5, "potions":0, "treasures":0, "enemies":0}
                    if c=="H":
                        self.door={"x":x, "y":y}
                    if c=="*":
                        self.potions.append({"x":x, "y":y})
                    if c=="$":
                        self.treasures.append({"x":x, "y":y})
                    if c=="g":
                        self.enemies.append({"x":x, "y":y, "damage":1})
                    if c=="o":
                        self.enemies.append({"x":x, "y":y, "damage":2})

    def clone(self):
        clone = State()
        clone.solid = self.solid
        clone.player = {"x":self.player["x"], "y":self.player["y"],
            "health":self.player["health"], "potions": self.player["potions"],
            "treasures":self.player["treasures"],"enemies":self.player["enemies"]}
        clone.door = self.door
        for t in self.treasures:
            clone.treasures.append(t)
        for e in self.enemies:
            clone.treasures.append(e)
        return clone

    def update(self, dirX, dirY):
        pass

    def getKey(self):
        return ""

    def getHeuristic(self):
        return 0

    def checkWin(self):
        return False

    def __str__(self):
        return ""
