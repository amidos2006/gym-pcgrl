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
        clone.width = self.width
        clone.height = self.height
        clone.solid = self.solid
        clone.player = {"x":self.player["x"], "y":self.player["y"],
            "health":self.player["health"], "potions": self.player["potions"],
            "treasures":self.player["treasures"],"enemies":self.player["enemies"]}
        clone.door = self.door
        for p in self.potions:
            clone.potions.append(p)
        for t in self.treasures:
            clone.treasures.append(t)
        for e in self.enemies:
            clone.enemies.append(e)
        return clone

    def checkMovableLocation(self, x, y):
        return not (x < 0 or y < 0 or x >= self.width or y >= self.height or self.solid[y][x])

    def checkPotionLocation(self, x, y):
        for p in self.potions:
            if p["x"] == x and p["y"] == y:
                return p
        return None

    def checkTreasureLocation(self, x, y):
        for t in self.treasures:
            if t["x"] == x and t["y"] == y:
                return t
        return None

    def checkEnemyLocation(self, x, y):
        for e in self.enemies:
            if e["x"] == x and e["y"] == y:
                return e
        return None

    def updatePlayer(self, x, y):
        self.player["x"] = x
        self.player["y"] = y
        toBeRemoved = self.checkPotionLocation(x, y)
        if toBeRemoved is not None:
            self.player["health"] += 2
            self.player["potions"] += 1
            if self.player["health"] > 5:
                self.player["health"] = 5
            self.potions.remove(toBeRemoved)
            return
        toBeRemoved = self.checkTreasureLocation(x, y)
        if toBeRemoved is not None:
            self.player["treasures"] += 1
            self.treasures.remove(toBeRemoved)
            return
        toBeRemoved = self.checkEnemyLocation(x, y)
        if toBeRemoved is not None:
            self.player["enemies"] += 1
            self.player["health"] -= toBeRemoved["damage"]
            if self.player["health"] < 0:
                self.player["health"] = 0
            self.enemies.remove(toBeRemoved)
            return

    def update(self, dirX, dirY):
        if self.checkOver():
            return
        if abs(dirX) > 0 and abs(dirY) > 0:
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
            self.updatePlayer(newX, newY)

    def getKey(self):
        key = str(self.player["x"]) + "," + str(self.player["y"]) + "," + str(self.player["health"]) + "|"
        key += str(self.door["x"]) + "," + str(self.door["y"]) + "|"
        for p in self.potions:
            key += str(p["x"]) + "," + str(p["y"]) + ","
        key = key[:-1] + "|"
        for t in self.treasures:
            key += str(t["x"]) + "," + str(t["y"]) + ","
        key = key[:-1] + "|"
        for e in self.enemies:
            key += str(e["x"]) + "," + str(e["y"]) + ","
        return key[:-1]

    def getHeuristic(self):
        playerDist = abs(self.player["x"] - self.door["x"]) + abs(self.player["y"] - self.door["y"])
        healthCost = 5 - self.player["health"]
        treasureCost = -self.player["treasures"]
        return playerDist + 4*healthCost + 4*treasureCost

    def getGameStatus(self):
        gameStatus = "running"
        if self.checkWin():
            gameStatus = "win"
        if self.checkLose():
            gameStatus = "lose"
        return {
            "status": gameStatus,
            "health": self.player["health"],
            "col_treasures": self.player["treasures"],
            "col_potions": self.player["potions"],
            "col_enemies": self.player["enemies"]
        }

    def checkOver(self):
        return self.checkWin() or self.checkLose()

    def checkWin(self):
        return self.player["x"] == self.door["x"] and self.player["y"] == self.door["y"]

    def checkLose(self):
        return self.player["health"] <= 0

    def __str__(self):
        result = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.solid[y][x]:
                    result += "#"
                else:
                    potion=self.checkPotionLocation(x,y) is not None
                    treasure=self.checkTreasureLocation(x,y) is not None
                    enemy=self.checkEnemyLocation(x,y)
                    if enemy is not None:
                        enemy=enemy["damage"]
                    player=self.player["x"]==x and self.player["y"]==y
                    door=self.door["x"]==x and self.door["y"]==y
                    if player:
                        if door:
                            result += "+"
                        else:
                            result += "@"
                    elif potion:
                        result +="*"
                    elif treasure:
                        result +="$"
                    elif enemy == 1:
                        result += "g"
                    elif enemy == 2:
                        result += "o"
                    elif door:
                        result += "H"
                    else:
                        result += " "
            result += "\n"
        return result[:-1]
