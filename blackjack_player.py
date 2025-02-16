import util, math, random
from collections import defaultdict
from util import ValueIteration

class CounterexampleMDP(util.MDP):
    def startState(self):
        return 'FLAT'

    def actions(self, state):
        return ['kukay']

    def succAndProbReward(self, state, action):
        result = []
        if state == "FLAT":
            result.append(("UP", 0.1, 9))
            result.append(("DOWN", 0.9, 1))
        return result

    def discount(self):
        return 1


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        result = []
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state
        if deckCardCounts is None:
            return []
        elif action == 'Quit' or sum(deckCardCounts) == 0:
            nextState = (0, None, None)
            nextReward = totalCardValueInHand
            if totalCardValueInHand > self.threshold:
                nextReward = 0
            result.append((nextState, 1.0, nextReward))
        elif action == 'Take':
            if nextCardIndexIfPeeked is not None:
                deckCardCountsList = list(deckCardCounts)
                deckCardCountsList[nextCardIndexIfPeeked] -= 1
                newValueInHand = self.cardValues[nextCardIndexIfPeeked] + totalCardValueInHand
                if newValueInHand > self.threshold:
                    nextState = (newValueInHand, None, None)
                elif sum(deckCardCountsList) == 0:
                    nextState = (newValueInHand, None, None)
                    nextReward = newValueInHand
                else:
                    nextState = (newValueInHand, None, tuple(deckCardCountsList))
                result.append((nextState, 1.0, 0))
            else:
                for index, item in enumerate(deckCardCounts):
                    if item > 0:
                        deckCardCountsList = list(deckCardCounts)
                        nextProb = float(item) / sum(deckCardCounts)
                        deckCardCountsList[index] -= 1
                        newValueInHand = self.cardValues[index] + totalCardValueInHand
                        nextReward = 0
                        if newValueInHand > self.threshold:
                            nextState = (newValueInHand, None, None)
                        elif sum(deckCardCountsList) == 0:
                            nextState = (newValueInHand, None, None)
                            nextReward = newValueInHand
                        else:
                            nextState = (newValueInHand, None, tuple(deckCardCountsList))
                        result.append((nextState, nextProb, nextReward))
        elif action == 'Peek':
            if nextCardIndexIfPeeked is not None:
                return []
            for index, item in enumerate(deckCardCounts):
                if item > 0:
                    nextProb = float(item) / sum(deckCardCounts)
                    nextState = (totalCardValueInHand, index, deckCardCounts)
                    nextReward = -self.peekCost
                    result.append((nextState, nextProb, nextReward))
        return result
        # END_YOUR_CODE

    def discount(self):
        return 1


def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    mdp1 = BlackjackMDP(cardValues=[1, 5, 15], multiplicity=10,
                        threshold=20, peekCost=1)
    return mdp1


# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state, action, reward, newState):
        V_opt = 0.0
        if newState is not None:
            V_opt = max([self.getQ(newState, newAction) for newAction in self.actions(newState)])
        Q_opt = self.getQ(state, action)
        adjustment = -self.getStepSize() * (Q_opt - (reward + self.discount * V_opt))
        for item in self.featureExtractor(state, action):
            key, value = item
            self.weights[key] = self.weights[key] + adjustment * value


def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)


def simulate_QL_over_MDP(mdp, featureExtractor):
    mdp.computeStates()



def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    features = []
    featureKey = (action, total)
    featureValue = 1
    features.append((featureKey, featureValue))
    if counts is not None:
        countsList = list(counts)
        for index, item in enumerate(counts):
            featureKey = (action, index, item)
            featureValue = 1
            features.append((featureKey, featureValue))
            if item > 0:
                countsList[index] = 1
        featureKey = (action, tuple(countsList))
        featureValue = 1
        features.append((featureKey, featureValue))
    return features



# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)


def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    pass
