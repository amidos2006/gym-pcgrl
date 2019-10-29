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
            crateMove = childState.update(d["x"], d["y"])
            if childState.player["x"] == self.state.player["x"] and childState.player["y"] == self.state.player["y"]:
                continue
            if crateMove and childState.checkDeadlock():
                continue
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
        self.solid=[]
        self.deadlocks = []
        self.targets=[]
        self.crates=[]
        self.player=None

    def randomInitialize(self, width, height):
        self.width-width
        self.height=height

        return

    def stringInitialize(self, lines):
        self.solid=[]
        self.targets=[]
        self.crates=[]
        self.player=None

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
                    if c == "@" or c=="+":
                        self.player={"x":x, "y":y}
                    if c=="$" or c=="*":
                        self.crates.append({"x":x, "y":y})
                    if c=="." or c=="+" or c=="*":
                        self.targets.append({"x":x, "y":y})
        self.intializeDeadlocks()

    def clone(self):
        clone=State()
        clone.width = self.width
        clone.height = self.height
        # since the solid is not changing then copy by value
        clone.solid = self.solid
        clone.deadlocks = self.deadlocks
        clone.player={"x":self.player["x"], "y":self.player["y"]}

        for t in self.targets:
            clone.targets.append({"x":t["x"], "y":t["y"]})

        for c in self.crates:
            clone.crates.append({"x":c["x"], "y":c["y"]})

        return clone

    def intializeDeadlocks(self):
        sign = lambda x: int(x/max(1,abs(x)))

        self.deadlocks = []
        for y in range(self.height):
            self.deadlocks.append([])
            for x in range(self.width):
                self.deadlocks[y].append(False)

        corners = []
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1 or self.solid[y][x]:
                    continue
                if (self.solid[y-1][x] and self.solid[y][x-1]) or (self.solid[y-1][x] and self.solid[y][x+1]) or (self.solid[y+1][x] and self.solid[y][x-1]) or (self.solid[y+1][x] and self.solid[y][x+1]):
                    if not self.checkTargetLocation(x, y):
                        corners.append({"x":x, "y":y})
                        self.deadlocks[y][x] = True

        for c1 in corners:
            for c2 in corners:
                dx,dy = sign(c1["x"] - c2["x"]), sign(c1["y"] - c2["y"])
                if (dx == 0 and dy == 0) or (dx != 0 and dy != 0):
                    continue
                walls = []
                x,y=c2["x"],c2["y"]
                if dx != 0:
                    x += dx
                    while x != c1["x"]:
                        if self.checkTargetLocation(x,y) or self.solid[y][x] or (not self.solid[y-1][x] and not self.solid[y+1][x]):
                            walls = []
                            break
                        walls.append({"x":x, "y":y})
                        x += dx
                if dy != 0:
                    y += dy
                    while y != c1["y"]:
                        if self.checkTargetLocation(x,y) or self.solid[y][x] or (not self.solid[y][x-1] and not self.solid[y][x+1]):
                            walls = []
                            break
                        walls.append({"x":x, "y":y})
                        y += dy
                for w in walls:
                    self.deadlocks[w["y"]][w["x"]] = True

    def checkDeadlock(self):
        for c in self.crates:
            if self.deadlocks[c["y"]][c["x"]]:
                return True
        return False

    def checkOutside(self, x, y):
        return x < 0 or y < 0 or x > len(self.solid[0]) - 1 or y > len(self.solid) - 1

    def checkTargetLocation(self, x, y):
        for t in self.targets:
            if t["x"] == x and t["y"] == y:
                return t
        return None

    def checkCrateLocation(self, x, y):
        for c in self.crates:
            if c["x"] == x and c["y"] == y:
                return c
        return None

    def checkMovableLocation(self, x, y):
        return not self.checkOutside(x, y) and not self.solid[y][x] and self.checkCrateLocation(x,y) is None

    def checkWin(self):
        if len(self.targets) != len(self.crates) or len(self.targets) == 0 or len(self.crates) == 0:
            return False

        for t in self.targets:
            if self.checkCrateLocation(t["x"], t["y"]) is None:
                return False

        return True

    def getHeuristic(self):
        targets=[]
        for t in self.targets:
            targets.append(t)
        distance=0
        for c in self.crates:
            bestDist = self.width + self.height
            bestMatch = 0
            for i,t in enumerate(targets):
                if bestDist > abs(c["x"] - t["x"]) + abs(c["y"] - t["y"]):
                    bestMatch = i
                    bestDist = abs(c["x"] - t["x"]) + abs(c["y"] - t["y"])
            distance += abs(targets[bestMatch]["x"] - c["x"]) + abs(targets[bestMatch]["y"] - c["y"])
            del targets[bestMatch]
        return distance

    def update(self, dirX, dirY):
        if abs(dirX) > 0 and abs(dirY) > 0:
            return
        if self.checkWin():
            return
        if dirX > 0:
            dirX=1
        if dirX < 0:
            dirX=-1
        if dirY > 0:
            dirY=1
        if dirY < 0:
            dirY=-1
        newX=self.player["x"]+dirX
        newY=self.player["y"]+dirY
        if self.checkMovableLocation(newX, newY):
            self.player["x"]=newX
            self.player["y"]=newY
        else:
            crate=self.checkCrateLocation(newX,newY)
            if crate is not None:
                crateX=crate["x"]+dirX
                crateY=crate["y"]+dirY
                if self.checkMovableLocation(crateX,crateY):
                    self.player["x"]=newX
                    self.player["y"]=newY
                    crate["x"]=crateX
                    crate["y"]=crateY
                    return True
        return False

    def getKey(self):
        key=str(self.player["x"]) + "," + str(self.player["y"]) + "," + str(len(self.crates)) + "," + str(len(self.targets))
        for c in self.crates:
            key += "," + str(c["x"]) + "," + str(c["y"]);
        for t in self.targets:
            key += "," + str(t["x"]) + "," + str(t["y"]);
        return key

    def __str__(self):
        result = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.solid[y][x]:
                    result += "#"
                else:
                    crate=self.checkCrateLocation(x,y) is not None
                    target=self.checkTargetLocation(x,y) is not None
                    player=self.player["x"]==x and self.player["y"]==y
                    if crate:
                        if target:
                            result += "*"
                        else:
                            result += "$"
                    elif player:
                        if target:
                            result += "+"
                        else:
                            result += "@"
                    else:
                        if target:
                            result += "."
                        else:
                            result += " "
            result += "\n"
        return result[:-1]
