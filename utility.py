import numpy as np

# are not utility functions
def utilityIdeal(delta, sigma):
    decision_maker = np.array([delta, sigma])
    best_decision = np.array([0, 0])
    return np.linalg.norm(decision_maker - best_decision)

def utilitySimple(delta, sigma):
    return -(delta + sigma)

def fairnessMinObjective(delta, sigma):
    decision_maker = np.array([delta, sigma])
    best_decision = np.array([0, 0])
    worst_decision = np.array([1, 0])
    return np.linalg.norm(decision_maker - best_decision) - np.linalg.norm(decision_maker - worst_decision)

# fulfill utility function axioms
def utilityFraction(delta, sigma, eps=1e-6):
    decision_maker = np.array([delta, sigma])
    best_decision = np.array([0, 0])
    worst_decision = np.array([1, 0])
    return np.linalg.norm(decision_maker - worst_decision) / (np.linalg.norm(decision_maker - best_decision)+eps)
    
def utilityMinus(delta, sigma):
    decision_maker = np.array([delta, sigma])
    best_decision = np.array([0, 0])
    worst_decision = np.array([1, 0])
    return np.linalg.norm(decision_maker - worst_decision) - np.linalg.norm(decision_maker - best_decision)

def utilityTopsis(delta, sigma):
    return utilityMinus(delta, sigma)
    
def utilityTopsisNorm(delta, sigma):
    return (utilityTopsis(delta, sigma) + 1) / 2
    

def testUtilityFunction(utility):
    """
    Test utility function
    
    Parameters
    ----------
    utility : function
        Utility function to test
    
    Returns
    -------
    bool
        True if test passed, False otherwise
    """
    test1 = utility(0, 0) > utility(0, 1)
    test2 = utility(0, 1) > utility(1, 1)
    test3 = utility(1, 1) > utility(1, 0)
    # Transitivity
    test4 = utility(0, 0) > utility(1, 1)
    test5 = utility(0, 1) > utility(1, 0)
    test6 = utility(0, 0) > utility(1, 0)
    print("Test 1:", test1)
    print("Test 2:", test2)
    print("Test 3:", test3)
    print("Test 4:", test4)
    print("Test 5:", test5)
    print("Test 6:", test6)
    return test1 and test2 and test3 and test4 and test5 and test6

def main():
    print("Utility function test passed:", testUtilityFunction(utilityIdeal))
    print("Utility function test passed:", testUtilityFunction(utilitySimple))
    print("Utility function test passed:", testUtilityFunction(utilityFraction))
    print("Utility function test passed:", testUtilityFunction(utilityTopsisNorm))
    print("Utility function test passed:", testUtilityFunction(utilityTopsis))

if __name__ == "__main__":
    main()
