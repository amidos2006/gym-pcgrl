from queue import PriorityQueue

directions = [{"x":0, "y":0}, {"x":1, "y":0}, {"x":0, "y":-1}, {"x":1, "y":-1}]
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
        self.player = None
        self.exit = -1

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
                        self.player = {"x": x, "y": y, "airTime": 0, "jumps": 0, "jump_locs": []}
                    if c == "|":
                        self.exit = x

    def clone(self):
        clone = State()
        clone.width = self.width
        clone.height = self.height
        clone.solid = self.solid
        clone.exit = self.exit
        clone.player = {"x":self.player["x"], "y":self.player["y"], "airTime":self.player["airTime"],
            "jumps":self.player["jumps"], "jump_locs": []}
        for l in self.player["jump_locs"]:
            clone.player["jump_locs"].append(l)
        return clone

    def checkMovableLocation(self, x, y):
        if y < 0:
            return True
        return not (x < 0 or x >= self.width or y >= self.height or self.solid[y][x])

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
        ground = False
        if self.player["y"] < len(self.solid) - 1 and self.player["y"] >= -1:
            ground = self.solid[self.player["y"] + 1][self.player["x"]]
        newX = self.player["x"]
        newY = self.player["y"]
        if abs(dirX) > 0:
            if self.checkMovableLocation(newX + dirX, newY):
                newX = newX + dirX
        if dirY == -1:
            if ground and self.checkMovableLocation(newX, newY-1):
                self.player["airTime"] = 5
                self.player["jumps"] += 1
                self.player["jump_locs"].append((self.player["x"], self.player["y"]))
        else:
            if self.player["airTime"] > 0:
                self.player["airTime"] = 1

        if self.player["airTime"] > 1:
            self.player["airTime"] -= 1
            if self.checkMovableLocation(newX, newY - 1):
                newY = newY - 1
            else:
                self.player["airTime"] = 1
        elif self.player["airTime"] == 1:
            self.player["airTime"] = 0
        else:
            if self.checkMovableLocation(newX, newY + 1):
                newY = newY + 1
        self.player["x"] = newX
        self.player["y"] = newY

    def getKey(self):
        return str(self.player["x"]) + "," + str(self.player["y"]) + "," + str(self.player["airTime"])

    def getHeuristic(self):
        return self.exit - self.player["x"]

    def getGameStatus(self):
        gameStatus = "running"
        if self.checkWin():
            gameStatus = "win"
        if self.checkLose():
            gameStatus = "lose"
        return {
            "status": gameStatus,
            "airTime": self.player["airTime"],
            "jumps": self.player["jumps"],
            "jump_locs": self.player["jump_locs"]
        }

    def checkOver(self):
        return self.checkWin() or self.checkLose()

    def checkWin(self):
        return self.player["x"] >= self.exit

    def checkLose(self):
        return self.player["y"] >= self.height

    def __str__(self):
        result = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.solid[y][x]:
                    result += "#"
                else:
                    player=self.player["x"]==x and self.player["y"]==y
                    exit=self.exit==x
                    if player:
                        if exit:
                            result += "w"
                        else:
                            result += "@"
                    elif exit:
                        result += "|"
                    else:
                        result += " "
            result += "\n"
        return result[:-1]
