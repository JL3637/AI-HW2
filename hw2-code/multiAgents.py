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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"

        food_list = newFood.asList()
        food_num = len(food_list)
        min_dis = 1000000
        for i in food_list:
            dis = manhattanDistance(i,newPos) + food_num * 100
            if dis < min_dis:
                min_dis = dis
        if food_num == 0:
            min_dis = 0
        score = -min_dis

        ghost_pos_list = successorGameState.getGhostPositions()
        for i in ghost_pos_list:
            if manhattanDistance(newPos,i)<=1 :
                score -= 1000000

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        agent_num = gameState.getNumAgents()
        action_score = []

        # def rm_stop(actions):
        #     if 'Stop' in actions:
        #         actions.remove('Stop')
        #     return actions

        def max_value(state, depth, agentIndex):
            v = -1000000
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minimax(successor, depth+1))
                if depth == 0:
                    action_score.append(v)
            return v
        
        def min_value(state, depth, agentIndex):
            v = 1000000
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, minimax(successor, depth+1))
            return v

        def minimax(state, depth):
            if state.isWin() or state.isLose() or depth >= self.depth * agent_num:
                return self.evaluationFunction(state)
            if depth % agent_num == 0:
                return max_value(state, depth, 0)
            else:
                return min_value(state, depth, depth % agent_num)

        minimax(gameState, 0)
        return gameState.getLegalActions(0)[action_score.index(max(action_score))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agent_num = gameState.getNumAgents()
        action_score = []

        def max_value(state, depth, agentIndex, alpha, beta):
            v = -1000000
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, alpha_beta(successor, depth+1, alpha, beta))
                if depth == 0:
                    action_score.append(v)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(state, depth, agentIndex, alpha, beta):
            v = 1000000
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, alpha_beta(successor, depth+1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def alpha_beta(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth >= self.depth * agent_num:
                return self.evaluationFunction(state)
            if depth % agent_num == 0:
                return max_value(state, depth, 0, alpha, beta)
            else:
                return min_value(state, depth, depth % agent_num, alpha, beta)

        alpha_beta(gameState, 0, -1000000, 1000000)
        return gameState.getLegalActions(0)[action_score.index(max(action_score))]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agent_num = gameState.getNumAgents()
        action_score = []

        def max_value(state, depth, agentIndex):
            v = -1000000
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(successor, depth+1))
                if depth == 0:
                    action_score.append(v)
            return v
        
        def exp_value(state, depth, agentIndex):
            v = 0
            legal_actions = state.getLegalActions(agentIndex)
            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                v += expectimax(successor, depth+1) / len(legal_actions)
            return v

        def expectimax(state, depth):
            if state.isWin() or state.isLose() or depth >= self.depth * agent_num:
                return self.evaluationFunction(state)
            if depth % agent_num == 0:
                return max_value(state, depth, 0)
            else:
                return exp_value(state, depth, depth % agent_num)

        expectimax(gameState, 0)
        return gameState.getLegalActions(0)[action_score.index(max(action_score))]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def _food_score(state):
        food_list = state.getFood().asList()
        score = 0
        for food in food_list:
            tmp = 1 / manhattanDistance(food, state.getPacmanPosition())
            if tmp > score:
                score = tmp
        return score
    
    def _capsules_score(state):
        capsules_list = state.getCapsules()
        score = 0
        for capsule in capsules_list:
            tmp = 50 / manhattanDistance(capsule, state.getPacmanPosition())
            if tmp > score:
                score = tmp
        return score
    
    def _ghost_score(state):
        ghost_list = state.getGhostStates()
        score = 0
        for ghost in ghost_list:
            dis = manhattanDistance(ghost.getPosition(), state.getPacmanPosition())
            if ghost.scaredTimer > 0:
                score += pow(max(7 - dis, 0), 2)
            else:
                score -= pow(max(8 - dis, 0), 2)
        return score



    current_score = currentGameState.getScore()
    food_score = _food_score(currentGameState)
    capsules_score = _capsules_score(currentGameState)
    ghost_score = _ghost_score(currentGameState)
    return current_score + food_score + capsules_score + ghost_score

# Abbreviation
better = betterEvaluationFunction
