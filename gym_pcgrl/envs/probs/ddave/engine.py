from queue import PriorityQueue

directions = [{"x":0, "y":0}, {"x":-1, "y":0}, {"x":1, "y":0}, {"x":0, "y":-1}]
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

    def checkLose(self):
        return self.state.checkLose()

    def checkOver(self):
        return self.state.checkOver()

    def getGameStatus(self):
        return self.state.getGameStatus()

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
            if current.checkLose():
                continue
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
            if current.checkLose():
                continue
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
            current = queue.get()
            if current.checkLose():
                continue
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
        self.spikes = []
        self.diamonds = []
        self.player = None
        self.key = None
        self.door = None
        self._airTime = 3
        self._hangTime = 1

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
                    if c == "$":
                        self.diamonds.append({"x": x, "y": y})
                    elif c == "*":
                        self.spikes.append({"x": x, "y": y})
                    elif c == "@":
                        self.player = {"x": x, "y": y, "health": 1, "airTime": 0, "diamonds": 0, "key": 0, "jumps": 0}
                    elif c == "H":
                        self.door = {"x": x, "y": y}
                    elif c == "V":
                        self.key = {"x": x, "y": y}

    def clone(self):
        clone = State()
        clone.width = self.width
        clone.height = self.height
        clone.solid = self.solid
        clone.door = self.door
        clone.spikes = self.spikes
        clone.key = self.key
        clone.player = {"x":self.player["x"], "y":self.player["y"],
            "health":self.player["health"], "airTime": self.player["airTime"],
            "diamonds":self.player["diamonds"], "key": self.player["key"], "jumps":self.player["jumps"]}
        for d in self.diamonds:
            clone.diamonds.append(d)
        return clone

    def checkMovableLocation(self, x, y):
        return not (x < 0 or y < 0 or x >= self.width or y >= self.height or self.solid[y][x])

    def checkSpikeLocation(self, x, y):
        for s in self.spikes:
            if s["x"] == x and s["y"] == y:
                return s
        return None

    def checkDiamondLocation(self, x, y):
        for d in self.diamonds:
            if d["x"] == x and d["y"] == y:
                return d
        return None

    def checkKeyLocation(self, x, y):
        if self.key is not None and self.key["x"] == x and self.key["y"] == y:
            return self.key
        return None

    def updatePlayer(self, x, y):
        self.player["x"] = x
        self.player["y"] = y
        toBeRemoved = self.checkDiamondLocation(x, y)
        if toBeRemoved is not None:
            self.player["diamonds"] += 1
            self.diamonds.remove(toBeRemoved)
            return
        toBeRemoved = self.checkSpikeLocation(x, y)
        if toBeRemoved is not None:
            self.player["health"] = 0
            return
        toBeRemoved = self.checkKeyLocation(x, y)
        if toBeRemoved is not None:
            self.player["key"] += 1
            self.key = None
            return

    def update(self, dirX, dirY):
        if self.checkOver():
            return
        if dirX > 0:
            dirX=1
        if dirX < 0:
            dirX=-1
        if dirY < 0:
            dirY=-1
        else:
            dirY=0

        ground = self.solid[self.player["y"] + 1][self.player["x"]]
        cieling = self.solid[self.player["y"] - 1][self.player["x"]]
        newX = self.player["x"]
        newY = self.player["y"]
        if abs(dirX) > 0:
            if self.checkMovableLocation(newX + dirX, newY):
                newX = newX + dirX
        elif dirY == -1:
            if ground and not cieling:
                self.player["airTime"] = self._airTime
                self.player["jumps"] += 1

        if self.player["airTime"] > self._hangTime:
            self.player["airTime"] -= 1
            if self.checkMovableLocation(newX, newY - 1):
                newY = newY - 1
            else:
                self.player["airTime"] = self._hangTime
        elif self.player["airTime"] > 0 and self.player["airTime"] <= self._hangTime:
            self.player["airTime"] -= 1
        else:
            if self.checkMovableLocation(newX, newY + 1):
                newY = newY + 1

        self.updatePlayer(newX, newY)

    def getKey(self):
        key = str(self.player["x"]) + "," + str(self.player["y"]) + "," + str(self.player["health"]) + "|"
        key += str(self.door["x"]) + "," + str(self.door["y"]) + "|"
        if self.key is not None:
            key += str(self.key["x"]) + "," + str(self.key["y"]) + "|"
        for d in self.diamonds:
            key += str(d["x"]) + "," + str(d["y"]) + ","
        key = key[:-1] + "|"
        for s in self.spikes:
            key += str(s["x"]) + "," + str(s["y"]) + ","
        return key[:-1]

    def getHeuristic(self):
        playerDist = abs(self.player["x"] - self.door["x"]) + abs(self.player["y"] - self.door["y"])
        if self.key is not None:
            playerDist = abs(self.player["x"] - self.key["x"]) + abs(self.player["y"] - self.key["y"]) + (self.width + self.height)
        diamondCosts = -self.player["diamonds"]
        return playerDist + 5*diamondCosts

    def getGameStatus(self):
        gameStatus = "running"
        if self.checkWin():
            gameStatus = "win"
        if self.checkLose():
            gameStatus = "lose"
        return {
            "status": gameStatus,
            "health": self.player["health"],
            "airTime": self.player["airTime"],
            "num_jumps": self.player["jumps"],
            "col_diamonds": self.player["diamonds"],
            "col_key": self.player["key"]
        }

    def checkOver(self):
        return self.checkWin() or self.checkLose()

    def checkWin(self):
        return self.player["key"] > 0 and self.player["x"] == self.door["x"] and self.player["y"] == self.door["y"]

    def checkLose(self):
        return self.player["health"] <= 0

    def __str__(self):
        result = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.solid[y][x]:
                    result += "#"
                else:
                    spike=self.checkSpikeLocation(x,y) is not None
                    diamond=self.checkDiamondLocation(x,y) is not None
                    key=self.checkKeyLocation(x,y) is not None
                    player=self.player["x"]==x and self.player["y"]==y
                    door=self.door["x"]==x and self.door["y"]==y
                    if player:
                        if spike:
                            result += "-"
                        elif door:
                            result += "+"
                        else:
                            result += "@"
                    elif spike:
                        result +="*"
                    elif diamond:
                        result +="$"
                    elif key:
                        result += "V"
                    elif door:
                        result += "H"
                    else:
                        result += " "
            result += "\n"
        return result[:-1]
