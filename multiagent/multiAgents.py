# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


bigNumber =100000000
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foodGrid = currentGameState.getFood()
       

        nearestGhost = bigNumber
        #counts the minimal manhattan distance to the nearest ghost
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                ghostPos = ghost.getPosition()
                #print(ghostPos)
                cur = manhattanDistance(ghostPos, newPos)
                if cur <= 0 :
                    return -bigNumber
                if nearestGhost > cur:
                    nearestGhost = cur
        
        nearestFood = bigNumber
        #counts the minimal manhattan distance to the nearest food
        for x in range(foodGrid.width):
            for y in range(foodGrid.height):
                if foodGrid[x][y] == True:
                    cur = manhattanDistance(newPos,(x,y))
                    if nearestFood > cur:
                        nearestFood = cur
        if nearestFood==0:
            return bigNumber
        #To avoid zero division or other error I decrease them 0.5 
        ghostCost = 1 / (nearestGhost - 0.5)
        foodCost = 1 / (nearestFood)
        return foodCost - ghostCost
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def pacmanMove(state, depth, totalNum):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            result = -bigNumber

            legalMoves = state.getLegalActions(0)

            #for each move checks if it is the best between the before moves
            for move in legalMoves:
                currentResult = ghostMove(state.generateSuccessor(0,move),depth, 1, totalNum)
                if result < currentResult:
                    result = currentResult
            return result

        def ghostMove(state,depth, ghostNum, totalNum):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            result = bigNumber

            legalMoves = state.getLegalActions(ghostNum)
            #Checks if the current ghostIndex is the last one and if it is goes to the pacman move else next index
            if ghostNum != totalNum:
                for move in legalMoves:
                    result = min(result,ghostMove(state.generateSuccessor(ghostNum,move),depth,ghostNum+1,totalNum))
            else:
                for move in legalMoves:
                    result = min(result,pacmanMove(state.generateSuccessor(ghostNum,move),depth-1,totalNum))
            return result


        result = -bigNumber
        legalMoves = gameState.getLegalActions(0)
        totalNum = gameState.getNumAgents() - 1
        resultAction = Directions.STOP
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0,move);

            #foreach move checks if this is the best one and changes the direction
            curResult = ghostMove(nextState,self.depth,1,totalNum)
            if curResult > result:
                result = curResult
                resultAction = move

        return resultAction

    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def finished(state,depth):
            return state.isWin() or state.isLose() or depth == 0 

        def pacmanMove(state,depth, alpha, beta,totalNum):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            result = -bigNumber
            legalMoves = state.getLegalActions(0)

            #foreach legal move checks if it is the best one
            for move in legalMoves:
                current = ghostMove(state.generateSuccessor(0,move),depth,1,totalNum,alpha,beta)
                if result < current:
                    result = current

                if result > beta:
                    return result
                alpha = max(alpha,result)

            return result


        def ghostMove(state, depth, ghostNum,totalNum, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            result = bigNumber
            legalMoves = state.getLegalActions(ghostNum)
            #Checks if the current ghostIndex is the last one and if it is goes to the pacman move else next index
            if ghostNum == totalNum:     
                for move in legalMoves:
                    result = min(result, pacmanMove(state.generateSuccessor(ghostNum, move), depth-1, alpha, beta,totalNum))
                    if result < alpha:
                        return result
                    beta = min(beta, result)
            else:
                for move in legalMoves:
                    result = min(result, ghostMove(state.generateSuccessor(ghostNum, move), depth, ghostNum + 1,totalNum, alpha, beta))
                    if result < alpha:
                        return result
                    beta = min(beta, result)
            return result

        totalNum = gameState.getNumAgents() - 1
        alpha = -bigNumber
        beta = bigNumber
        answer = -bigNumber
        initialMoves = gameState.getLegalActions(0)
        answerDir = Directions.STOP
        #Foreach move checks if it is the best move
        for move in initialMoves:
            nextState = gameState.generateSuccessor(0,move)

            current  = ghostMove(nextState,self.depth,1,totalNum,alpha,beta)

            if current > answer:
                answer = current
                answerDir = move
            if answer > beta:
                return answerDir
            alpha = max(alpha,answer)
        return answerDir

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def pacmanMove(state, depth, numghosts):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            result = -bigNumber
            legalMoves = state.getLegalActions(0)
            for move in legalMoves:
                result = max(result, ghostMove(state.generateSuccessor(0, move), depth, 1, numghosts))
            return result
        
        def ghostMove(state, depth, ghostNum, totalNum):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            result = 0
            legalMoves = state.getLegalActions(ghostNum)
            denominator = len(legalMoves)

            #Checks if the current ghostIndex is the last one and if it is goes to the pacman move else next index
            if ghostNum == totalNum:
                for move in legalMoves:
                    result += (1.0/denominator) * pacmanMove(state.generateSuccessor(ghostNum, move), depth-1, totalNum)
            else:
                for move in legalMoves:
                    result+= (1.0/denominator) * ghostMove(state.generateSuccessor(ghostNum, move), depth, ghostNum + 1, totalNum)
            return result

        legalMoves = gameState.getLegalActions()
        numghosts = gameState.getNumAgents() - 1
        direction = Directions.STOP
        result = -bigNumber
        #foreach move chekcs if it is the best one
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
           
            current =  ghostMove(nextState, self.depth, 1, numghosts)
            if current > result:
                direction = move
                result = current
        return direction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    At firt find the coordinates of the nearest ghost and food.
    Than cound ghostcost and foodCost using manhattan distances and some constant.
    Also count the total number of food in evaluation function.
    """
    x,y = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostsStates = currentGameState.getGhostStates()
    foodNum= len(foodGrid.asList())
        
    ghostPos = (0,0)
    nearestGhost = bigNumber
    #Cycle to find the nearest ghost
    for ghost in ghostsStates:
        if ghost.scaredTimer <= 0:
            current = manhattanDistance((x,y),ghost.getPosition())
            if current < nearestGhost:
                nearestGhost = current
                ghostPos = ghost.getPosition()

    ghostCost = manhattanDistance(ghostPos,(x,y))
    if manhattanDistance(ghostPos,(x,y))==0:
        ghostCost -= 500
    
    nearestFood = (0,0)
    nearestDist = bigNumber

    #Two cycles to find the nearest food
    for i in range(foodGrid.width):
        for j in range(foodGrid.height):
            current = manhattanDistance((x,y),(i,j))
            if current < nearestDist:
                nearestDist = current
                nearestFood = (i,j)

    foodCost = -manhattanDistance((x,y),nearestFood)
    foodNumCost = -15*(foodNum)
    return ghostCost + foodCost + foodNumCost

# Abbreviation
better = betterEvaluationFunction
