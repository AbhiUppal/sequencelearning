import math
import time
import os
import numpy as np
import pandas as pd
import random as rd
# import concurrent.futures
import matplotlib.pyplot as plt
# import seaborn as sns
import scipy as sp
import scipy.optimize as spo
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
# from bayes_opt import BayesianOptimization
import multiprocessing as mp

"""
AU comment: I'll leave some comments on this code.
Really sorry for the mess, not my proudest work lol

Probably not gonna be able to be comprehensive in this, but feel free to contact me with questions:

auppal22@cmc.edu
(510) 408-8376

Happy to help figure out what parts of this code do (honestly I'll need to stare at it for a bit too)
"""


def orderRMarkov_generation(R=0, alpha=1, length=200):
    # number of symbols generated is length
    # alpha is the concentration parameter of the Dirichlet distribution we draw probabilities from
    # R is the order of the order R Markov model
    symbols = ''
    # start with a random seed of R symbols
    for i in range(R):
        symbols = symbols + str(np.random.binomial(p=0.5,n=1))
    # formulate order R Markov model
    dict_of_strings = {}
    for i in range(np.power(2,R)):
        string = np.binary_repr(i,width=R)
        dict_of_strings[string] = np.random.dirichlet([alpha,alpha])[0]
    # generate signal
    for i in range(R,length):
        p = dict_of_strings[symbols[i-R:i]]
        newsymbol = str(np.random.binomial(p=p,n=1))
        symbols = symbols + newsymbol
    return [int(symbols[i]) for i in range(length)]


# Define the moments of the Dirichlet distribution

def expectedValueDir(sigInterest, sigTotal, alpha = 1, beta = 1):
    numerator = alpha + beta * sigInterest
    denominator = 2 * alpha + beta * sigTotal
    if type(denominator) == type(0.3) or type(denominator) == type(0):
        if denominator == 0:
            res = np.nan
            return 0.5
        else:
            res = numerator / denominator
            return res
    else:
        res = np.array([numerator / denominator]) # Allows us to vectorize ngram_observer, making it (hopefully) faster
        res[np.isnan(res)] = 0.5
        res[np.isinf(res)] = 0.5
        return res[0]

def modeDir(sigInterest, sigTotal, alpha = 1, beta = 1):
    numerator = alpha - 1 + beta * sigInterest
    denominator = 2*alpha - 2 + beta * sigTotal
    if type(denominator) == type(0.3) or type(denominator) == type(0):
        if denominator == 0:
            res = np.nan
            return 0.5
        else:
            res = numerator / denominator
            return res
    else:
        res = np.array([numerator / denominator]) # Allows us to vectorize ngram_observer, making it (hopefully) faster
        res[np.isnan(res)] = 0.5
        res[np.isinf(res)] = 0.5
        return res[0]

# Define the function that actually does the simulation

def simulate(repetitions, probTrue, sigTrue, sigFalse, alpha = 1, beta = 1):
    """
    AU Note: I think this function is useless in the final product. DO NOT USE THIS

    Generate a string of data from a one-state, IID process. Calculates the moments of the Dirichlet distribution at each point in time.
    repetitions: Number of data points to simulate
    probTrue: The probability of the "True" signal being generated
    sigTrue: The signal to be generated probTrue percent of the time
    sigFalse: The signal to be generated 1-probTrue percent of the time
    alpha: Hyperparameter for the Dirichlet distribution
    beta: Probability that the agent will actually update (i.e., they don't update 1-beta percent of the time)

    Returns a Pandas dataframe of the results with column names "Time", "Mode", and "Signal"
    note that, within the dataframe, moments are calculated BEFORE signals are generated.
    So, the mode at time 0 is the prior, and the mode at time 1 represents the mode after seeing the signal generated at t = 0
    """
    time = np.array([])
    signals = np.array([]) # SM: probably want to generate these first to start just as one when there's memory
    modeBelief = np.array([])
    for t in range(0, repetitions):
        time = np.append(time, t)
        signals = np.append(signals, IID(p = probTrue, sigTrue = sigTrue, sigFalse = sigFalse))
        # SM comments:
        # modeBelief: need to look only at n(x|sigma) and n(sigma) for the "right" order R Markov model
        # to get "right" order R Markov model, must calculate argmax of p(R|data) first
        # that probably needs new function
        # would be interesting to store the argmax_R's as another array to see how the Bayesian observer changes his or her response
        modeBelief = np.append(modeBelief, modeDir(sigInterest = np.sum(signals[:-1] == sigTrue), sigTotal = len(signals[:-1]), alpha = alpha, beta = beta))
    results = pd.DataFrame(
        {"Time": time,
        "Mode": modeBelief,
        "Signal": signals}
    )
    return results

#----------------------------------------------
# Calculating the ML Order R model from data
#----------------------------------------------

def NumSearch(big, sub):
    """
    Helper function for ModelML. Returns the number of times the string sub appears in the string big.
    sub: string of shorter or equal length to big
    big: string to search in
    """
    results = 0
    subLen = len(sub)
    for i in range(len(big)): 
        if big[i:i + subLen] == sub:
            results += 1
    return results 

def modelML(data, R = 0, alpha = 1, beta = 1, gamma = 1):
    """
    AU comment: This is the big function, also the computational beast. Given a strinbg (or list) of signals (0 or 1), it calculates the likelihoods for the data being generated from each order.
    Note that this likelihood does NOT depend on beta. This is intended, but I'm looking into if this was a mistake or not.
    
    Using a string of signals, calculates the unnormalized likelihood of the data being generated from a model of order 0 up to order R.

    Returns a dictionary with keys representing the order and values representing the likelihoods
    """
    probabilities = {}
    if type(data) == type([]):
        newdata = ""
        for item in data:
            newdata += str(item)
    else:
        newdata = data
    
    # Calculate the (non-normalized) logged probability of each order

    for order in range(0, R + 1): # Loop over the possible Markov model orders
        numWords = 2**(order)       # Define the number of possible words
        words = []
        for num in range(0, numWords): # Prepare the list of possible words
            if len(np.binary_repr(num)) < order:
                words.append("0" * (order - len(np.binary_repr(num))) + np.binary_repr(num))
            else:
                words.append(np.binary_repr(num))
        
        matches = {} # Create a dictionary containing the number of times a word is found in the data
        for word in words:
            if len(word) > len(data):
                matches[word] = 0
            else:
                matches[word] = NumSearch(newdata, word) # The dict value for each word (the key) is the number of times we see the word in the data
                # This provides easy lookup for the next step

        # Now for the fun part: calculating the probabilities
        term1 = -(2**order) * gamma
        term2 = (2**order) * (sp.special.gammaln(2*alpha) - 2*sp.special.gammaln(alpha))
        # Calculating terms 3 and 4 involve sums, so they're a bit more involved -- I'll define them iteratively
        term3 = 0
        for word in words:
            for i in [0, 1]:
                term3 += sp.special.gammaln(beta * NumSearch(newdata, word + str(i)) + alpha)
        term4 = 0
        if order != 0:
            for word in words:
                term4 += sp.special.gammaln(beta * matches[word] + 2*alpha)
        else:
            term4 = sp.special.gammaln(beta * (len(data)) + 2*alpha)
        
        #print("Model order:", order)
        #print("Term 1", term1)
        #print("Term 2:", term2)
        #print("Term 3", term3)
        #print("Term 4", term4)
        
        probabilities[order] = term1 + term2 + term3 - term4
    return probabilities

#----------------------------------------------
# Analyzing Order R data + Helper Functions
#----------------------------------------------

def ngram_observer(data, R = 0, alpha = 1, beta = 1, gamma = 1, recordLiks = False):
    """
    Takes in a string of data and returns a dataframe with all of the information for an observer using an ngram model.
    That is, it records the data, the most likely model, the number of signals that are 1 (sigInterest), and the total number of signals.
    If recordLiks is set to true, returns all of the model likelihoods as well.


    data: String of data to be taken in as input
    R: The maximum order Markov model the observer considers
    alpha: Concentration hyperparameter
    beta: Probability an observer uses an observation to update their belief
    gamma: term adjusting the prior on larger size models
    recordLiks: If true, returns likelihoods, modes, etc. of all models, rather than just returning the ML model. This provides a dataframe capable of calculating L4 from.
    """
    time = np.array([])
    modeBelief = np.array([])
    argmaxR = np.array([])
    datastr = ""
    for obs in data:
        datastr += str(obs)
    if recordLiks:              # Adds arrays that will allow us to store information for calculating L4
        orderLiks = [["LikR0"]] # Allows us to store likelihoods at each point in time (could probably rewrite with a list comprehension to make it faster)
        orderNInterest = [["InterestR0"]]
        orderNTotal = [["TotalR0"]]
        for i in range(1, R + 1):
            orderLiks.append(["LikR" + str(i)])
            orderNInterest.append(["InterestR" + str(i)])
            orderNTotal.append(["TotalR" + str(i)])
    nInterest = np.array([]) # For calculating mode belief with varying values of alpha, beta
    nTotal = np.array([])    # Same, also for optimization
    
    for t in range(0, len(data)):
        time = np.append(time, t)
        signals = datastr[0:t]
        modelLikelihoods = modelML(data = signals, R = R, alpha = alpha, beta = beta, gamma = gamma)

        if recordLiks:
            for i in range(0, R + 1):
                orderLiks[i].append(modelLikelihoods[i])
                if i != 0:
                    lastRsignals = signals[-1 - i: len(signals)]
                    nSigma = NumSearch(signals, lastRsignals)
                    nSigma1 = NumSearch(signals, lastRsignals + "1")
                    orderNTotal[i].append(nSigma)
                    orderNInterest[i].append(nSigma1)
                else:
                    nSigma = len(signals)
                    nSigma1 = NumSearch(signals, "1")
                    orderNTotal[i].append(nSigma)
                    orderNInterest[i].append(nSigma1)

        orderML = max(modelLikelihoods, key = modelLikelihoods.get)
        argmaxR = np.append(argmaxR, orderML)

        # Now that we have the order, we want to find out what state we're in
        if orderML != 0: # Deals with higher-order Markov chains
            lastRsignals = signals[-1 - orderML: len(signals)]
            nSigma = NumSearch(signals, lastRsignals)
            nSigma1 = NumSearch(signals, lastRsignals + "1")
            nInterest = np.append(nInterest, nSigma1)
            nTotal = np.append(nTotal, nSigma)
        else: # Deals with the IID case
            nSigma = len(signals)
            nSigma1 = NumSearch(signals, "1")
            nInterest = np.append(nInterest, nSigma1)
            nTotal = np.append(nTotal, nSigma)
        
        # Calculate the mode of the dirichlet distribution and append it to the mode belief array
        mode = modeDir(sigInterest = nSigma1, sigTotal = nSigma, alpha = alpha, beta = beta)
        modeBelief = np.append(modeBelief, mode)

    results = pd.DataFrame(
        {"Time": time,
        "Mode": modeBelief,
        "Signal": data,
        "maxR": argmaxR,
        "nInterest": nInterest,
        "nTotal": nTotal}
    )

    if recordLiks:
        for i in range(0, R + 1):
            results[orderLiks[i][0]] = orderLiks[i][1:]
            results[orderNInterest[i][0]] = orderNInterest[i][1:]
            results[orderNTotal[i][0]] = orderNTotal[i][1:]

        for i in range(0, R + 1):
            results["ModeR" + str(i)] = modeDir(sigInterest = np.array(orderNInterest[i][1:]), sigTotal = np.array(orderNTotal[i][1:]), alpha = alpha, beta = beta)
            results["EVR" + str(i)] = expectedValueDir(sigInterest = np.array(orderNInterest[i][1:]), sigTotal = np.array(orderNTotal[i][1:]), alpha = alpha, beta = beta)

    return results

def observer_L4(data, R = 0, alpha = 1, beta = 1):
    """
    Takes in a string or list of data and returns a dataframe with the model likelihoods and other information for an L4 observer (ngram-average)


    data: String or list of data to be taken in as input
    R: The maximum order Markov model the observer considers
    alpha: Concentration hyperparameter
    beta: Probability an observer uses an observation to update their belief
    gamma: term adjusting the prior on larger size models
    """
    time = np.array([])
    datastr = ""
    for obs in data:
        datastr += str(obs)

    orderNInterest = [["InterestR0"]]
    orderNTotal = [["TotalR0"]]

    for i in range(1, R + 1):
        orderNInterest.append(["InterestR" + str(i)])
        orderNTotal.append(["TotalR" + str(i)])
    
    for t in range(0, len(data)):
        time = np.append(time, t)
        for i in range(0, R + 1):
            signals = datastr[0:t]
    
            if i != 0:
                lastRsignals = signals[-1 - i: len(signals)]
                nSigma = NumSearch(signals, lastRsignals)
                nSigma1 = NumSearch(signals, lastRsignals + "1")
                orderNTotal[i].append(nSigma)
                orderNInterest[i].append(nSigma1)
            else:
                nSigma = len(signals)
                nSigma1 = NumSearch(signals, "1")
                orderNTotal[i].append(nSigma)
                orderNInterest[i].append(nSigma1)
        
    results = pd.DataFrame(
        {"Time": time,
        "Signal": data}
    )

    for i in range(0, R + 1):
        results[orderNInterest[i][0]] = orderNInterest[i][1:]
        results[orderNTotal[i][0]] = orderNTotal[i][1:]
        results["ModeR" + str(i)] = modeDir(sigInterest = np.array(orderNInterest[i][1:]), sigTotal = np.array(orderNTotal[i][1:]), alpha = alpha, beta = beta)
        results["EVR" + str(i)] = expectedValueDir(sigInterest = np.array(orderNInterest[i][1:]), sigTotal = np.array(orderNTotal[i][1:]), alpha = alpha, beta = beta)
    
    return results
    


def calc_L2(data, R = 0, alpha = 1, beta = 1, gamma = 1, alphaTrue = 1, betaTrue = 1, gammaTrue = 1):
    ngramTrue = ngram_observer(data, R = R, alpha = alphaTrue, beta =  betaTrue, gamma = gammaTrue)
    ngramGuess = ngram_observer(data, R = R, alpha = alpha, beta = beta, gamma = gamma)
    L2 = np.array([])
    term1 = ngramTrue["Mode"] * np.log(ngramGuess["Mode"])
    term2 = (1 - ngramTrue["Mode"]) * np.log(1 - ngramGuess["Mode"])
    L2 = np.append(L2, term1 + term2)
    return (alpha, beta, gamma, np.nansum(L2))

def calc_L2_phi(data, R = 4, beta = 1, phi = 0.7, phiTrue = 0.3, gamma = 1, gammaTrue = 1):
    """
    AU comment: This is the one to use. Any calculations not using phi are misleading at best.
    Used heavily in the optimization code.

    Calculates L2 using the ratio of alpha and beta (phi)
    Returns a tuple: (phi, gamma, L2)
    
    progress = False
    if progress:
        print(f"Phi: {phi} | Gamma: {gamma}")
    """
    alphaTrue = 1 + (beta/phiTrue)
    alphaGuess = 1 + (beta/phi)
    ngramTrue = ngram_observer(data, R = R, alpha = alphaTrue, beta =  beta, gamma = gammaTrue)
    ngramGuess = ngram_observer(data, R = R, alpha = alphaGuess, beta = beta, gamma = gamma)
    L2 = np.array([])
    term1 = ngramTrue["Mode"] * np.log(ngramGuess["Mode"])
    term2 = (1 - ngramTrue["Mode"]) * np.log(1 - ngramGuess["Mode"])
    L2 = np.append(L2, term1 + term2)
    return (phi, gamma, np.nansum(L2))

def predStr_L2(df):
    """
    AU comment: Used only for simulation purposes

    Produces a string of predictions for an ngram observer
    Helper function for calculating a stochastic version of L2 
    df: output of ngram_observer
    """
    preds = np.array([])
    modes = df["Mode"]
    for item in modes:
        prob = item
        pred = np.random.binomial(1, p = prob)
        preds = np.append(preds, pred)
    
    return preds


def calcStochastic_L2(data, R = 0, alpha = 1, beta = 1, gamma = 1):
    """
    Calculates the L2 value if the observer actually makes a random guess
    """
    L2 = np.array([])
    ngramGuess = ngram_observer(data, R = R, alpha = alpha, beta = beta, gamma = gamma)
    ngramGuess["pred"] = predStr_L2(ngramGuess)
    for i in range(len(ngramGuess["pred"])):
        if ngramGuess.at[i, "pred"] == 1:
            L2 = np.append(L2, np.log(ngramGuess.at[i, "Mode"]))
        else:
            L2 = np.append(L2, np.log(1 - ngramGuess.at[i, "Mode"]))
    
    return np.nansum(L2)

def likelihood_L2(data, obs_preds, alpha, beta, gamma, R = 5):
    """
    AU comment: This is useful for analyzing experimental data

    Calculates L2 from a string of predictions
    """
    observer = ngram_observer(data, R = R, alpha = alpha, beta = beta, gamma = gamma, recordLiks = False)
    L2 = obs_preds * np.log(observer["Mode"]) + (1 - obs_preds) * np.log(1 - observer["Mode"])
    
    return [(alpha, beta, gamma), np.nansum(L2)]

def sumTerm(data, priors):
    """
    HELPER FUNCTION
    Calculates the sum term used in calculating L4 in the method where we average over observers
    data: output of ngram_observer or observer_L4 with recordLiks set to True (important!)
    """
    add = np.zeros(len(data["Time"]))
    for i in range(0, (len(priors))):
        add += priors[i] * data["EVR" + str(i)]

    return add

def calc_L4(data, R = 0, alpha = 1, beta = 1, gamma = 1, alphaTrue = 1, betaTrue = 1, gammaTrue = 1):
    """
    AU Comment: do not use this. Use calc_L4_phi

    Calculates L4 if we average over infinite identical observers
    data: string of data from OrderRMarkov_generation
    """
    # Get the priors over model orders
    priorsGuess = np.array([])
    priorsTrue = np.array([])
    for order in range(0, R + 1):
        priorsGuess = np.append(priorsGuess, -gamma * (2**order))
        priorsTrue = np.append(priorsTrue, -gammaTrue * (2**order))
    
    # Normalize
    priorsGuess = np.exp(priorsGuess) / np.exp(priorsGuess).sum()
    priorsTrue = np.exp(priorsTrue) / np.exp(priorsTrue).sum()

    ngramTrue = observer_L4(data, R = R, alpha = alphaTrue, beta = betaTrue)
    ngramGuess = observer_L4(data, R = R, alpha = alpha, beta = beta)

    sumTrue = sumTerm(ngramTrue, priorsTrue)
    sumGuess = sumTerm(ngramGuess, priorsGuess)

    L4 = sumTrue * np.log(sumGuess) + (1 - sumTrue) * np.log(1 - sumGuess)

    return (alpha, beta, gamma, np.nansum(L4))

def calc_L4_phi(data, R = 0, beta = 1, phi = 0.5, phiTrue = 0.3, gamma = 1, gammaTrue = 1):
    """
    AU Comment: Use this one.

    Calculates L4 given the data and model parameters.
    """
    
    # Get the priors over model orders
    priorsGuess = np.array([])
    priorsTrue = np.array([])
    for order in range(0, R + 1):
        priorsGuess = np.append(priorsGuess, -gamma * (2**order))
        priorsTrue = np.append(priorsTrue, -gammaTrue * (2**order))
    
    # Normalize
    priorsGuess = np.exp(priorsGuess) / np.exp(priorsGuess).sum()
    priorsTrue = np.exp(priorsTrue) / np.exp(priorsTrue).sum()

    # Recover the alpha value from the ratios
    alphaGuess = (beta/phi)
    alphaTrue = (beta/phiTrue)

    ngramTrue = observer_L4(data, R = R, alpha = alphaTrue, beta = beta)
    ngramGuess = observer_L4(data, R = R, alpha = alphaGuess, beta = beta)

    sumTrue = sumTerm(ngramTrue, priorsTrue)
    sumGuess = sumTerm(ngramGuess, priorsGuess)

    L4 = sumTrue * np.log(sumGuess) + (1 - sumTrue) * np.log(1 - sumGuess)

    return (phi, gamma, np.nansum(L4))


def calc_L4_average(data, R = 0, alpha = 1, beta = 1, gamma = 1, n_obs = 1):
    """
    AU comment: Not actually sure why I left this one empty...

    Calculates L4 if we average over multiple observer predictions
    n_obs: number of observers
    """

def likelihood_L4(data, obs_preds, alpha, beta, gamma, R = 5):
    """
    Calculates L4 when given a string of predictions or a list of predictions.
    This is useful for analyzing actual experimental data.
    """
    priors = np.array([])
    for order in range(0, R + 1):
        priors = np.append(priors, - gamma * (2**order))
    priors = np.exp(priors)/np.exp(priors).sum()

    observer = observer_L4(data, R = R, alpha = alpha, beta = beta)

    sums = sumTerm(data = observer, priors = priors)
    # print("Sums:", len(sums))
    # print(len(obs_preds))

    if len(obs_preds) > 1:
        p = sum(obs_preds)/len(obs_preds)
        probs = p * np.log(sums) + (1 - p) * np.log(1 - sums)
        

    else:
        p = sum(obs_preds)/len(obs_preds)
        probs = p * np.log(sums) + (1 - p) * np.log(1 - sums)


    return [(alpha, beta, gamma), np.nansum(probs)]

def predStr_L4(data,  R = 5, alpha = 3, beta = 1, gamma = 1):
    """
    Returns a numpy array of observer predictions given a string of data and observer parameters

    AU comment: I think this is for simulation?
    """
    observer = observer_L4(data = data, R = R, alpha = alpha, beta = beta)
    
    priors = np.array([])
    for order in range(0, R + 1):
        priors = np.append(priors, - gamma * (2**order))
    priors = np.exp(priors)/np.exp(priors).sum()

    sums = sumTerm(data = observer, priors = priors)

    preds = []

    for term in sums:
        prob = term
        pred = np.random.binomial(1, p = prob)
        preds.append(pred)
    
    preds = np.array(preds)

    return preds
 

def predict_GLM(data, lags = 5):
    """
    Parameters
    ---------------------
    data: A list of 0s and 1s
    lags: the number of symbols the observer looks back. Must be a positive integer.

    Outputs
    ------------------------
    A set of probabilities for the next symbol being a 1
    """
    # Create a dataframe
    df = pd.DataFrame({"x": data})
    
    # Add lagged terms
    for lag in range(1, lags + 1):
        df["x_lag" + str(lag)] = df["x"].shift(periods = lag)
    
    probPreds = np.array([]) # Array of predicted probabilities

    # Create feature and response matrix / vector
    yLarge = df.loc[:, "x"]
    firstCol = "x_lag1"
    finalCol = "x_lag" + str(lags)
    XLarge = df.loc[:, firstCol:finalCol]
    for t in range(len(data) - lags):
        X = XLarge.iloc[lags : lags + t + 1]
        y = yLarge.iloc[lags : lags + t + 1]

        if not ((1 in y.unique()) and (0 in y.unique())):
            # print("Using this case")
            probPreds = np.append(probPreds, list(y)[len(y) - 1])
        else:
            logit = LogisticRegression(solver = "lbfgs")
            logit.fit(X = X, y = y)
            pred = logit.predict_proba(X)[:, 0]
            probPreds = np.append(probPreds, pred[-1])

    for i in range(lags):
        probPreds = np.insert(probPreds, 0, 0.5)

    df["prob"] = probPreds

    return df

def predstr_GLM(df):
    """
    df: Output of predict_GLM
    Helper function for predict_GLM that allows us to produce predictions from the data
    Returns an array of predictions that could be produced from a GLM observer
    """
    prediction = np.array([])
    for i in range(len(df["prob"])):
        p = df["prob"][i]
        prediction = np.append(prediction, np.random.binomial(1, p))
    
    return prediction


def likelihood_GLM(data, obs_preds, lags = 5):
    """
    Calculates the likelihood of an observer using a GLM model given data and observer predictions.

    AU Comment: This function handles everything for analyzing the GLM case.
    """
    predFrame = predict_GLM(data, lags)
    likelihood = np.array([])
    probs = predFrame["prob"]

    p = sum(obs_preds) / len(obs_preds)
    likelihood = p * np.log(probs) + (1 - p) * np.log(1 - probs)

    return np.nansum(likelihood)

def predFunc_guess(data, preds, lags = 5, simulate = "", alphaTrue = 3, betaTrue = 1, gammaTrue = 1, numObs = 1):
    """
    Takes a string of model output and predictions, returns the likelihood of each prediction strategy
    Parameters
    -----------------
    data: Time series of symbols to be analyzed
    preds: A string of observer predictions. Pass an empty list if you want to simulate observations
    alpha: The true alpha value of the observer
    beta: The true beta value of the observer
    gamma: The true gamma value of the observer
    lags: # of lags to consider in the GLM
    simulate: Empty if prediction list is supplied, if not, pass "L2", "L4", or "GLM" for simulating predictions from that particular method
    
    Outputs
    ------------------
    A dictionary of prediction strategies and their corresponding likelihoods
    """

    if simulate == "GLM":
        GLM = predict_GLM(data, lags = lags)
        preds = [predstr_GLM(GLM) for i in range(numObs)]
    
    if simulate == "L2":
        L2 = ngram_observer(data = data, R = lags, alpha = alphaTrue, beta = betaTrue, gamma = gammaTrue)
        preds = predStr_L2(L2)

    if simulate == "L4" or simulate == "Bayesian":
        preds = [predStr_L4(data = data, R = lags, alpha = alphaTrue, beta = betaTrue, gamma = gammaTrue) for i in range(numObs)]
    
    lik_GLM = likelihood_GLM(data = data, obs_preds = preds, lags = lags)
    print("Finished Calculating GLM. Now calculating L4....")

    # lik_L2 = varyParams_predL2(data = data, obs_preds = preds, R = lags, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.1)
    # print("Finished Calculating L2. Now calculating L4....")

    lik_L4 = varyParams_predL4(data = data, obs_preds = preds, R = lags, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.1)
    print("Finished Calculating L4.")

    res = {"GLM" : lik_GLM, "L4" : lik_L4}

    return res

def evalStrategy(n_each = 75, lags = 5, alphaTrue = 3, betaTrue = 1, gammaTrue = 1, sampleSize = 200, numObs = 1):
    strategies = ["GLM", "Bayesian"]
    true = []
    preds = []
    totalIter = str(n_each * len(strategies))
    counter = 1
    for strategy in strategies:
        for i in range(n_each):
            print("Step", str(counter) + "/" + totalIter)
            order = rd.randint(1, 6)
            dataAlpha = rd.uniform(1.01, 5)
            data = orderRMarkov_generation(R = order, alpha = dataAlpha, length = sampleSize)
            true.append(strategy)
            likelihoods = predFunc_guess(data = data, preds = [], lags = lags, simulate = strategy, alphaTrue = alphaTrue, betaTrue = betaTrue, gammaTrue = gammaTrue, numObs = numObs)
            pred = max(likelihoods, key = likelihoods.get)
            preds.append(pred)
    
    true = np.array(true)
    preds = np.array(preds)
    
    df = pd.DataFrame({"true": true, "predicted": preds})

    return df

def str_to_lst(string):
    """
    Converts a string to a list containing each of its elements

    This is a helper function to assist in parsing experimental data presented as long strings
    """
    res = []
    for char in string:
        res.append(int(char))
    return res

#----------------------------------------------
# Optimization Code
#----------------------------------------------
"""
AU Comment: This stuff is all essentially the same optimization code repeated for different use cases.
I think the ones with "phi" in their names are the ones we ended up using.
My hope is that you will never have to open any of these functions. It's quite ugly and I'm honestly scared of it.
In the unfortunate case that you do, reach out to me and we'll walk through it together
"""


def varyParams(data, R = 4, alphaMax = 10, betaMax = 1, gamma = 1, step = 0.01, alphaTrue = 1, betaTrue = 1, gammaTrue = 1):
    """
    Creates a dataframe of L2 with varied parametersm given a dataset
    Varies alpha from [1, alphaMax], beta from [0, betaMax]
    """
    # Initiate arrays used in the output
    alphaVals = np.array([])
    betaVals = np.array([])
    L2 = np.array([])
    
    # Create the domain of values (alpha, beta)
    alphaRange = np.array([i for i in range(int(1/step), int(alphaMax/step) + 1, 1)]) * step
    betaRange = np.array([i for i in range(0, int(betaMax/step) + 1, 1)]) * step

    for i in range(0, len(alphaRange)):
        alphaRange[i] = round(alphaRange[i], 3)
    for i in range(0, len(betaRange)):
        betaRange[i] = round(betaRange[i], 3)

    # Iterate over (alpha, beta) pairs in the domain
    for alpha in alphaRange:
        for beta in betaRange:
            print("alpha:", alpha, "| beta:", beta)
            betaVals = np.append(betaVals, beta)
            alphaVals = np.append(alphaVals, alpha)
            Lval = calc_L2(data, R = R, alpha = alpha, beta = beta, gamma = gamma, alphaTrue = alphaTrue, betaTrue = betaTrue, gammaTrue = gammaTrue)
            L2 = np.append(L2, Lval)

    results = pd.DataFrame({
        "Alpha": alphaVals,
        "Beta": betaVals,
        "L2": L2
    })
    return results

def varyParams_stochastic(data, R = 4, alphaMax = 10, betaMax = 1, gamma = 1, step = 0.1):
    """
    Creates a dataframe of L2 with varied parametersm given a dataset
    Varies alpha from [1, alphaMax], beta from [0, betaMax]
    """
    # Initiate arrays used in the output
    alphaVals = np.array([])
    betaVals = np.array([])
    L2 = np.array([])
    
    # Create the domain of values (alpha, beta)
    alphaRange = np.array([i for i in range(int(1/step), int(alphaMax/step) + 1, 1)]) * step
    betaRange = np.array([i for i in range(0, int(betaMax/step) + 1, 1)]) * step

    for i in range(0, len(alphaRange)):
        alphaRange[i] = round(alphaRange[i], 3)
    for i in range(0, len(betaRange)):
        betaRange[i] = round(betaRange[i], 3)

    # Iterate over (alpha, beta) pairs in the domain
    for alpha in alphaRange:
        for beta in betaRange:
            print("alpha:", alpha, "| beta:", beta)
            betaVals = np.append(betaVals, beta)
            alphaVals = np.append(alphaVals, alpha)
            Lval = calcStochastic_L2(data, R = R, alpha = alpha, beta = beta, gamma = gamma)
            L2 = np.append(L2, Lval)

    results = pd.DataFrame({
        "Alpha": alphaVals,
        "Beta": betaVals,
        "L2": L2
    })
    return results

def varyParams_gamma(data, R = 4, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.01, alphaTrue = 1, betaTrue = 1, gammaTrue = 2):
    """
    Creates a dataframe of L2 with varied parametersm given a dataset
    Varies alpha from [1, alphaMax], beta from [0, betaMax]
    """
    # Initiate arrays used in the output
    alphaVals = np.array([])
    betaVals = np.array([])
    gammaVals = np.array([])
    L2 = np.array([])
    
    # Create the domain of values (alpha, beta)
    alphaRange = np.array([i for i in range(int(1/step), int(alphaMax/step) + 1, 1)]) * step
    betaRange = np.array([i for i in range(0, int(betaMax/step) + 1, 1)]) * step
    gammaRange = np.array([i for i in range(0, int(gammaMax/step) + 1, 1)]) * step

    for i in range(0, len(alphaRange)):
        alphaRange[i] = round(alphaRange[i], 3)
    for i in range(0, len(betaRange)):
        betaRange[i] = round(betaRange[i], 3)
    for i in range(0, len(gammaRange)):
        gammaRange[i] = round(gammaRange[i], 3)

    pool = mp.Pool(processes = mp.cpu_count() - 2)
    results = pool.starmap(calc_L2, ([(data, R, alpha, beta, gamma, alphaTrue, betaTrue, gammaTrue) for alpha in alphaRange for beta in betaRange for gamma in gammaRange]))

    for res in results:
        alphaVals = np.append(alphaVals, res[0])
        betaVals = np.append(betaVals, res[1])
        gammaVals = np.append(gammaVals, res[2])
        L2 = np.append(L2, res[3])
    
    out = pd.DataFrame({
        "Alpha": alphaVals,
        "Beta": betaVals,
        "Gamma": gammaVals,
        "L2": L2
    })
    return out

def varyParams_gammaL4(data, R = 4, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.1, alphaTrue = 1, betaTrue = 1, gammaTrue = 2):
    """
    Creates a dataframe of L2 with varied parametersm given a dataset
    Varies alpha from [1, alphaMax], beta from [0, betaMax]
    """
    # Initiate arrays used in the output
    alphaVals = np.array([])
    betaVals = np.array([])
    gammaVals = np.array([])
    L4 = np.array([])

    # Create the domain of values (alpha, beta)
    alphaRange = np.array([i for i in range(int(1/step), int(alphaMax/step) + 1, 1)]) * step
    betaRange = np.array([i for i in range(0, int(betaMax/step) + 1, 1)]) * step
    gammaRange = np.array([i for i in range(0, int(gammaMax/step) + 1, 1)]) * step

    for i in range(0, len(alphaRange)):
        alphaRange[i] = round(alphaRange[i], 3)
    for i in range(0, len(betaRange)):
        betaRange[i] = round(betaRange[i], 3)
    for i in range(0, len(gammaRange)):
        gammaRange[i] = round(gammaRange[i], 3)

    pool = mp.Pool(processes = mp.cpu_count() - 1)
    results = pool.starmap(calc_L4, ([(data, R, alpha, beta, gamma, alphaTrue, betaTrue, gammaTrue) for alpha in alphaRange for beta in betaRange for gamma in gammaRange]))

    for res in results:
        alphaVals = np.append(alphaVals, res[0])
        betaVals = np.append(betaVals, res[1])
        gammaVals = np.append(gammaVals, res[2])
        L4 = np.append(L4, res[3])
    
    out = pd.DataFrame({
        "Alpha": alphaVals,
        "Beta": betaVals,
        "Gamma": gammaVals,
        "L4": L4
    })
    return out

def varyParams_predL2(data, obs_preds, R = 4, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.1):
    """
    Finds the optimal alpha, beta, gamma combination when we calculate L2 given a string of predictions
    Optimal means that it produces the MLE for L2
    Gamma will produce a range -- this won't really matter
    """
    
    alphaVals = np.array([])
    betaVals = np.array([])
    gammaVals = np.array([])
    L2 = np.array([])
    
    # Create the domain of values (alpha, beta)
    alphaRange = np.array([i for i in range(int(1/step), int(alphaMax/step) + 1, 1)]) * step
    betaRange = np.array([i for i in range(0, int(betaMax/step) + 1, 1)]) * step
    gammaRange = np.array([i for i in range(0, int(gammaMax/step) + 1, 1)]) * step

    for i in range(0, len(alphaRange)):
        alphaRange[i] = round(alphaRange[i], 3)
    for i in range(0, len(betaRange)):
        betaRange[i] = round(betaRange[i], 3)
    for i in range(0, len(gammaRange)):
        gammaRange[i] = round(gammaRange[i], 3)

    pool = mp.Pool()
    results = pool.starmap(likelihood_L2, ([(data, obs_preds, alpha, beta, gamma, R) for alpha in alphaRange for beta in betaRange for gamma in gammaRange]))

    for res in results:
        alphaVals = np.append(alphaVals, res[0][0])
        betaVals = np.append(betaVals, res[0][1])
        gammaVals = np.append(gammaVals, res[0][2])
        L2 = np.append(L2, res[1])
    
    return max(L2)
    
    """
    out = pd.DataFrame({
        "Alpha": alphaVals,
        "Beta": betaVals,
        "Gamma": gammaVals,
        "L2": L2
    })
    
    maxim = max([out["L2"]])
    outSub = out[out["L2"] == maxim]

    return out
    """

def varyParams_predL4(data, obs_preds, R = 4, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.1):
    """
    Finds the optimal alpha, beta, gamma combination when we calculate L2 given a string of predictions
    Optimal means that it produces the MLE for L4
    """
    alphaVals = np.array([])
    betaVals = np.array([])
    gammaVals = np.array([])
    L4 = np.array([])
    
    # Create the domain of values (alpha, beta)
    alphaRange = np.array([i for i in range(int(1/step), int(alphaMax/step) + 1, 1)]) * step
    betaRange = np.array([i for i in range(0, int(betaMax/step) + 1, 1)]) * step
    gammaRange = np.array([i for i in range(0, int(gammaMax/step) + 1, 1)]) * step

    for i in range(0, len(alphaRange)):
        alphaRange[i] = round(alphaRange[i], 3)
    for i in range(0, len(betaRange)):
        betaRange[i] = round(betaRange[i], 3)
    for i in range(0, len(gammaRange)):
        gammaRange[i] = round(gammaRange[i], 3)

    pool = mp.Pool()
    results = pool.starmap(likelihood_L4, ([(data, obs_preds, alpha, beta, gamma, R) for alpha in alphaRange for beta in betaRange for gamma in gammaRange]))
    
    for res in results:
        alphaVals = np.append(alphaVals, res[0][0])
        betaVals = np.append(betaVals, res[0][1])
        gammaVals = np.append(gammaVals, res[0][2])
        L4 = np.append(L4, res[1])
    
    return max(L4)

def varyParams_predL4_copy(data, alphaTrue = 2, betaTrue = 0.8, gammaTrue = 1.5, R = 4, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.1):
    """
    Identical to varyParams_predL4, but returns the error in the estimates instead of the estimate of maximum likelihood
    """
    alphaVals = np.array([])
    betaVals = np.array([])
    gammaVals = np.array([])
    L4 = np.array([])

    obs_preds = predStr_L4(data = data, R = R, alpha = alphaTrue, beta = betaTrue, gamma = gammaTrue)
    
    # Create the domain of values (alpha, beta)
    alphaRange = np.array([i for i in range(int(1/step), int(alphaMax/step) + 1, 1)]) * step
    betaRange = np.array([i for i in range(0, int(betaMax/step) + 1, 1)]) * step
    gammaRange = np.array([i for i in range(0, int(gammaMax/step) + 1, 1)]) * step

    for i in range(0, len(alphaRange)):
        alphaRange[i] = round(alphaRange[i], 3)
    for i in range(0, len(betaRange)):
        betaRange[i] = round(betaRange[i], 3)
    for i in range(0, len(gammaRange)):
        gammaRange[i] = round(gammaRange[i], 3)

    pool = mp.Pool()
    results = pool.starmap(likelihood_L4, ([(data, obs_preds, alpha, beta, gamma, R) for alpha in alphaRange for beta in betaRange for gamma in gammaRange]))
    
    for res in results:
        alphaVals = np.append(alphaVals, res[0][0])
        betaVals = np.append(betaVals, res[0][1])
        gammaVals = np.append(gammaVals, res[0][2])
        L4 = np.append(L4, res[1])
    
    MLE = max(L4)
    
    df = pd.DataFrame({"Alpha": alphaVals, "Beta" : betaVals, "Gamma" : gammaVals, "L4" : L4})

    sub = df[df["L4"] == MLE]

    sub["ErrorAlpha"] = sub["Alpha"] - alphaTrue
    sub["ErrorBeta"] = sub["Beta"] - betaTrue
    sub["ErrorGamma"] = sub["Gamma"] - gammaTrue

    errors = {"Alpha": np.nanmean(sub["ErrorAlpha"]), "Beta" : np.nanmean(sub["ErrorBeta"]), "Gamma": np.nanmean(sub["ErrorGamma"])}

    return [df, errors]

def varyParams_phi_L2(data, R = 4, beta = 1, phiMax = 1.5, step = 0.01, phiTrue = 5, gamma = 1, gammaTrue = 1):
    # 335
    phiVals = np.array([])
    gammaVals = np.array([])
    L2 = np.array([])

    phiRange = np.array([i for i in range(int(1/step), int(phiMax/step) + 1, 1)]) * step

    pool = mp.Pool(processes = mp.cpu_count() - 2)
    results = pool.starmap(calc_L2_phi, ([(data, R, beta, phi, phiTrue, gamma, gammaTrue) for phi in phiRange]))

    for res in results:
        phiVals = np.append(phiVals, res[0])
        gammaVals = np.append(gammaVals, res[1])
        L2 = np.append(L2, res[2])
    
    out = pd.DataFrame({"Phi" : phiVals, "Gamma" : gammaVals, "L2" : L2})

    return out

def varyParams_phi_gamma_L2(data, R = 4, beta = 1, phiMax = 1.5, step_phi = 0.01, phiTrue = 5, gammaMax = 2, step_gamma = 0.1, gammaTrue = 1):
    """
    Insert docstring here... in a rush
    """
    phiVals = np.array([])
    L2 = np.array([])
    gammaVals = np.array([])

    phiRange = np.array([i for i in range(0, int(phiMax / step_phi) + 1, 1)]) * step_phi
    gammaRange = np.array([i for i in range(0, int(gammaMax / step_gamma) + 1, 1)]) * step_gamma

    pool = mp.Pool(processes = mp.cpu_count() - 2)
    results = pool.starmap(calc_L2_phi, ([(data, R, beta, phi, phiTrue, gamma, gammaTrue) for phi in phiRange for gamma in gammaRange]))

    for res in results:
        phiVals = np.append(phiVals, res[0])
        gammaVals = np.append(gammaVals, res[1])
        L2 = np.append(L2, res[2])
    
    out = pd.DataFrame({"Phi" : phiVals, "Gamma" : gammaVals, "L2" : L2})

    return out

def varyParams_phi_gamma_L4(data, R = 4, beta = 1, phiMax = 2, step_phi = 0.01, phiTrue = 5, gammaMax = 2, step_gamma = 0.1, gammaTrue = 1) :
    phiVals = np.array([])
    L2 = np.array([])
    gammaVals = np.array([])

    phiRange = np.array([i for i in range(0, int(phiMax / step_phi) + 1, 1)]) * step_phi
    gammaRange = np.array([i for i in range(0, int(gammaMax / step_gamma) + 1, 1)]) * step_gamma

    pool = mp.Pool(processes = mp.cpu_count() - 2)
    results = pool.starmap(calc_L4_phi, ([(data, R, beta, phi, phiTrue, gamma, gammaTrue) for phi in phiRange for gamma in gammaRange]))

    for res in results:
        phiVals = np.append(phiVals, res[0])
        gammaVals = np.append(gammaVals, res[1])
        L2 = np.append(L2, res[2])
    
    out = pd.DataFrame({"Phi" : phiVals, "Gamma" : gammaVals, "L4" : L2})

    return out

"""
AU Comment: These functions were all used in calculating the errors for error bars in our parameter inference.
Full docstring is in ErrorCalcsL2, the rest are adaptations of it.
"""

def ErrorCalcsL2(R_vals = [3, 4, 5], n_vals = [200, 400, 600], alphaTrue = 1, betaTrue = 1, gammaTrue = 1, n_iter = 100, dataAlpha = 2, alphaMax = 5, gammaMax = 4, step = 0.1):
    """
    Function for simulation of inferring alpha, beta, and gamma. Iterates over each (R, n) pair an equal amount of times
    INPUTS:
    R_vals: model orders to iterate over
    n_vals: Sample sizes to iterate over
    alphaTrue: The observer's true alpha value
    betaTrue: The observer's true beta value
    gammaTrue: The observer's true gamma value
    n_iter: Number of total iterations to perform
    dataAlpha: Alpha value used to 

    ------------------------------------------------------------------------
    RETURNS a dataframe containing the following columns:
    
    inferAlphaLow: The lowest inferred value of alpha on a particular run
    inferAlphaHigh: The highest inferred value of alpha on a particular run
    inferBetaLow: The lowest inferred value of beta on a particular run
    inferBetaHigh: The highest inferred value of beta on a particular run
    inferGammaLow: The lowest inferred value of gamma on a particular run
    inferGammaHigh: The highest inferred value of higher on a particular run
    N: Sample size of the particular run
    R: The true order of the model in a particular run
    trueAlpha: The true value of alpha
    trueBeta: The true value of beta
    trueGamma: The true value of gamma

    ------------------------------------------------------------------------
    Doesn't actually calculate the error, but the output file can be easily used to do so.
    """

    # Calculate the number of partitions to split iterations across
    numPartitions = len(R_vals) * len(n_vals)
    partitionLength = round(n_iter / numPartitions)
    print(numPartitions)
    print(partitionLength)

    # Intialize arrays to store final outputs
    alphaLow = np.array([])
    alphaHigh = np.array([])
    betaLow = np.array([])
    betaHigh = np.array([])
    gammaLow = np.array([])
    gammaHigh = np.array([])
    trueAlpha = np.array([])
    trueBeta = np.array([])
    trueGamma = np.array([])
    sampleSize = np.array([])
    modelOrder = np.array([])

    # Iterate over (R, n) pairs partitionLength times per pair
    for R in R_vals:
        for n in n_vals:
            print("________________________________________")
            print("R:", R, "N:", n)
            for partition in range(partitionLength):
                data = orderRMarkov_generation(R = R, alpha = dataAlpha, length = n)
                simResults = varyParams_gamma(data = data, R = R + 1, alphaMax = alphaMax, betaMax = 1, gammaMax = gammaMax, step = step, alphaTrue = alphaTrue, betaTrue = betaTrue, gammaTrue = gammaTrue)
                simResultsSub = simResults[simResults["L2"] == max(simResults["L2"])]
                alphaLow = np.append(alphaLow, min(simResultsSub["Alpha"]))
                alphaHigh = np.append(alphaHigh, max(simResultsSub["Alpha"]))
                betaLow = np.append(betaLow, min(simResultsSub["Beta"]))
                betaHigh = np.append(betaHigh, max(simResultsSub["Beta"]))
                gammaLow = np.append(gammaLow, min(simResultsSub["Gamma"]))
                gammaHigh = np.append(gammaHigh, max(simResultsSub["Gamma"]))
                sampleSize = np.append(sampleSize, n)
                modelOrder = np.append(modelOrder, R)
                trueAlpha = np.append(trueAlpha, alphaTrue)
                trueBeta = np.append(trueBeta, betaTrue)
                trueGamma = np.append(trueGamma, gammaTrue)
                print(str(partition / partitionLength) + " percent complete with this step")
            df = pd.DataFrame(
            {"inferAlphaLow" : alphaLow,
            "inferAlphaHigh": alphaHigh,
            "inferBetaLow": betaLow,
            "inferBetaHigh": betaHigh,
            "inferGammaLow": gammaLow,
            "inferGammaHigh": gammaHigh,
            "N": sampleSize,
            "R": modelOrder,
            "trueAlpha": trueAlpha,
            "trueBeta": trueBeta,
            "trueGamma": trueGamma})
            # filename = "R" + str(R) + "_N" + str(n) + "_errors_L2.csv"
            # df.to_csv(filename, index = False)

    results = pd.DataFrame(
        {"inferAlphaLow" : alphaLow,
        "inferAlphaHigh": alphaHigh,
        "inferBetaLow": betaLow,
        "inferBetaHigh": betaHigh,
        "inferGammaLow": gammaLow,
        "inferGammaHigh": gammaHigh,
        "N": sampleSize,
        "R": modelOrder,
        "trueAlpha": trueAlpha,
        "trueBeta": trueBeta,
        "trueGamma": trueGamma})

    results.to_csv("ErrorCalcsL2Output.csv", index = False)
    
    return results

def ErrorCalcsL4(R_vals = [3, 4, 5], n_vals = [200, 400, 600], alphaTrue = 1, betaTrue = 1, gammaTrue = 1, n_iter = 120, dataAlpha = 2, alphaMax = 5, gammaMax = 4, step = 0.1):

    # Calculate the number of partitions to split iterations across
    test = False
    if test:
        n_iter = len(R_vals) + len(n_vals)

    numPartitions = len(R_vals) * len(n_vals)
    partitionLength = round(n_iter / numPartitions)
    print(numPartitions)
    print(partitionLength)
    # Intialize arrays to store final outputs
    alphaLow = np.array([])
    alphaHigh = np.array([])
    betaLow = np.array([])
    betaHigh = np.array([])
    gammaLow = np.array([])
    gammaHigh = np.array([])
    trueAlpha = np.array([])
    trueBeta = np.array([])
    trueGamma = np.array([])
    sampleSize = np.array([])
    modelOrder = np.array([])

    # Iterate over (R, n) pairs partitionLength times per pair
    for R in R_vals:
        for n in n_vals:
            print("________________________________________")
            print("R:", R, "N:", n)
            for partition in range(partitionLength):
                data = orderRMarkov_generation(R = R, alpha = dataAlpha, length = n)
                simResults = varyParams_gammaL4(data = data, R = R + 1, alphaMax = alphaMax, betaMax = 1, gammaMax = gammaMax, step = step, alphaTrue = alphaTrue, betaTrue = betaTrue, gammaTrue = gammaTrue)
                simResultsSub = simResults[simResults["L4"] == max(simResults["L4"])]
                alphaLow = np.append(alphaLow, min(simResultsSub["Alpha"]))
                alphaHigh = np.append(alphaHigh, max(simResultsSub["Alpha"]))
                betaLow = np.append(betaLow, min(simResultsSub["Beta"]))
                betaHigh = np.append(betaHigh, max(simResultsSub["Beta"]))
                gammaLow = np.append(gammaLow, min(simResultsSub["Gamma"]))
                gammaHigh = np.append(gammaHigh, max(simResultsSub["Gamma"]))
                sampleSize = np.append(sampleSize, n)
                modelOrder = np.append(modelOrder, R)
                trueAlpha = np.append(trueAlpha, alphaTrue)
                trueBeta = np.append(trueBeta, betaTrue)
                trueGamma = np.append(trueGamma, gammaTrue)
            df = pd.DataFrame(
            {"inferAlphaLow" : alphaLow,
            "inferAlphaHigh": alphaHigh,
            "inferBetaLow": betaLow,
            "inferBetaHigh": betaHigh,
            "inferGammaLow": gammaLow,
            "inferGammaHigh": gammaHigh,
            "N": sampleSize,
            "R": modelOrder,
            "trueAlpha": trueAlpha,
            "trueBeta": trueBeta,
            "trueGamma": trueGamma})
            # filename = "R" + str(R) + "_N" + str(n) + "_errors.csv"
            # outfile = open(filename, 'w')
            # df.to_csv(outfile, index = False)
            # outfile.close()


    results = pd.DataFrame(
        {"inferAlphaLow" : alphaLow,
        "inferAlphaHigh": alphaHigh,
        "inferBetaLow": betaLow,
        "inferBetaHigh": betaHigh,
        "inferGammaLow": gammaLow,
        "inferGammaHigh": gammaHigh,
        "N": sampleSize,
        "R": modelOrder,
        "trueAlpha": trueAlpha,
        "trueBeta": trueBeta,
        "trueGamma": trueGamma})

    results.to_csv("ErrorCalcsOutput.csv", index = False)
    
    return results

def ErrorCalcsL2_phi(R_vals = [2, 4, 6], n_vals = [200, 400, 600], phiTrue = 0.25, gammaTrue = 1, n_iter = 100, dataAlpha = 2, phiMax = 2, gammaMax = 3, phiStep = 0.01, gammaStep = 0.1):
    """
    Performs the ErrorCalcs function with phi instead
    """
    numPartitions = len(R_vals) * len(n_vals)
    partitionLength = round(n_iter / numPartitions)
    print(numPartitions)
    print(partitionLength)

    # Intialize arrays to store final outputs
    phiLow = np.array([])
    phiHigh = np.array([])
    gammaLow = np.array([])
    gammaHigh = np.array([])
    truePhi = np.array([])
    trueGamma = np.array([])
    sampleSize = np.array([])
    modelOrder = np.array([])

    # Iterate over (R, n) pairs partitionLength times per pair
    for R in R_vals:
        for n in n_vals:
            print("________________________________________")
            print("R:", R, "N:", n)
            for partition in range(partitionLength):
                data = orderRMarkov_generation(R = R, alpha = dataAlpha, length = n)
                # data, R = 4, beta = 1, phiMax = 1.5, step_phi = 0.01, phiTrue = 5, gammaMax = 2, step_gamma = 0.1, gammaTrue = 1
                simResults = varyParams_phi_gamma_L2(data = data, R = R + 1, beta = 1, phiMax = phiMax, step_phi = phiStep, phiTrue = phiTrue, gammaMax = gammaMax, step_gamma = gammaStep, gammaTrue = gammaTrue)
                simResultsSub = simResults[simResults["L2"] == max(simResults["L2"])]
                phiLow = np.append(phiLow, min(simResultsSub["Phi"]))
                phiHigh = np.append(phiHigh, max(simResultsSub["Phi"]))
                gammaLow = np.append(gammaLow, min(simResultsSub["Gamma"]))
                gammaHigh = np.append(gammaHigh, max(simResultsSub["Gamma"]))
                sampleSize = np.append(sampleSize, n)
                modelOrder = np.append(modelOrder, R)
                truePhi = np.append(truePhi, phiTrue)
                trueGamma = np.append(trueGamma, gammaTrue)
            df = pd.DataFrame(
            {"inferPhiLow" : phiLow,
            "inferPhiHigh": phiHigh,
            "inferGammaLow": gammaLow,
            "inferGammaHigh": gammaHigh,
            "N": sampleSize,
            "R": modelOrder,
            "truePhi": truePhi,
            "trueGamma": trueGamma})
            filename = "PHI_R" + str(R) + "_N" + str(n) + "_errors_L2.csv"
            df.to_csv(filename, index = False)
    
    results = pd.DataFrame(
        {"inferPhiLow" : phiLow,
        "inferPhiHigh": phiHigh,
        "inferGammaLow": gammaLow,
        "inferGammaHigh": gammaHigh,
        "N": sampleSize,
        "R": modelOrder,
        "truePhi": truePhi,
        "trueGamma": trueGamma})

    results.to_csv("ErrorCalcsL2Output_PHI.csv", index = False)
    
    return results


#----------------------------------------------
# Analyzing Error Data
#----------------------------------------------

"""
AU Comment: These are functions that were used in creating the error bar plots for the paper
Honestly a bit scared to open these, too. But if they become necessary I'm willing to walk through them.
"""

def analyzeErrors(errorData, params = ["Alpha", "Beta", "Gamma"], N_vals = [200, 400], R_vals = [2, 4, 6]):
    for param in params:
        errorData["errorLow" + param] = errorData["infer" + param + "Low"] - errorData["true" + param]
        errorData["errorHigh" + param] = errorData["infer" + param + "High"] - errorData["true" + param]

    bars = {}

    for param in params:
        X_space = []
        for R in R_vals:
            for N in N_vals:           
                subset = errorData[(errorData["R"] == R) & (errorData["N"] == N)]
                df = len(subset) - 1
                p = 0.975 # For a 95% CI
                tcrit = sp.stats.t.ppf(p, df)
                label = "R = " + str(R) + ", \n N = " + str(N)

                print(subset["errorLow" + param])

                errorLowMean = np.mean(subset["errorLow" + param])
                errorLowSD = np.std(subset["errorLow" + param])
                errorLowSE = errorLowSD / np.sqrt(len(subset))
                # print("Low SE:", errorLowSE)
                errorLowCI = [errorLowMean - tcrit * errorLowSE, errorLowMean + tcrit * errorLowSE]

                errorHighMean = np.mean(subset["errorHigh" + param])
                errorHighSD = np.std(subset["errorHigh" + param])
                errorHighSE = errorHighSD / np.sqrt(len(subset))
                # print("High SE:", errorHighSE)
                errorHighCI = [errorHighMean - tcrit * errorHighSE, errorHighMean + tcrit * errorHighSE]
                
                bars[label] = {"Low" : [errorLowMean, tcrit * errorHighSE], "High" : [errorHighMean, tcrit * errorHighSE]}
                X_space.append(label)
        
        # Plot stuff here
        print(bars)
        combos = []
        lowPoints = []
        lowBars = []
        highPoints = []
        highBars = []
        for combo in X_space:
            combos.append(combo)
            lowPoints.append(np.mean(bars[combo]["Low"][0])) 
            lowBars.append(bars[combo]["Low"][1])

            highPoints.append(np.mean(bars[combo]["High"][0])) 
            highBars.append(bars[combo]["High"][1])

        combos = np.array(combos)
        lowPoints = np.array(lowPoints)
        lowBars = np.array(lowBars)
        highPoints = np.array(highPoints)
        highBars = np.array(highBars)

        df = pd.DataFrame({
            "combo" : combos,
            "lowPoint" : lowPoints,
            "lowBar" : lowBars,
            "highPoint" : highPoints,
            "highBar" : highBars
        })

        print(df["combo"])

        plt.rc("text", usetex = True)
        plt.rc("font", family = "serif")

        ax = plt.axes()
        plt.errorbar(x = df["combo"], y = df["lowPoint"], yerr = df["lowBar"], label = "Error: Lower Bound", fmt = "o", alpha = 0.5, capsize = 3, color = "blue")
        plt.errorbar(x = df["combo"], y = df["highPoint"], yerr = df["highBar"], label = "Error: Upper Bound", fmt = "o", alpha = 0.5, capsize = 3, color = "red")

        
        if param == "Alpha":
            ax.set_title(r"Calculated Errors in Inferring $\alpha$ Parameter")
        elif param == "Beta":
            ax.set_title(r"Calculated Errors in Inferring $\beta$ Parameter")
        elif param == "Gamma":
            ax.set_title(r"Calculated Errors in Inferring $\gamma$ Parameter")
        
        ax.set_xlabel("(R, N) Combination")
        ax.set_ylabel("Inference Error")
        ax.axhline(y = 0, linestyle = "--", color = "black", alpha = 0.15)
        plt.gcf().subplots_adjust(bottom = 0.15)
        
        # ax.set_xticklabels(rotation = (15), labels = df["combo"])
        ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1), ncol = 2, fancybox = True, shadow = True)
        plt.savefig(param + "Errors.png", format = "png")
        plt.show()


    return errorData

def analyzeErrors_ratio(errorData, N_vals = [200, 400, 600], R_vals = [2, 4, 6], value = "L2"):
    if value == "L4":
        errorData["inferRatioLow"] = errorData["inferBetaLow"] / (errorData["inferAlphaLow"])
        errorData["trueRatio"] = errorData["trueBeta"] / (errorData["trueAlpha"])
        errorData["inferRatioHigh"] = errorData["inferBetaHigh"] / (errorData["inferAlphaHigh"])
    
    else:
        errorData["inferRatioLow"] = errorData["inferBetaLow"] / (errorData["inferAlphaLow"] - 1)
        errorData["trueRatio"] = errorData["trueBeta"] / (errorData["trueAlpha"] - 1)
        errorData["inferRatioHigh"] = errorData["inferBetaHigh"] / (errorData["inferAlphaHigh"] - 1)

    errorData["errorRatioLow"] = errorData["inferRatioLow"] - errorData["trueRatio"]
    errorData["errorRatioHigh"] = errorData["inferRatioHigh"] - errorData["trueRatio"]

    X_space = []
    bars = {}

    for R in R_vals:
        for N in N_vals:           
            subset = errorData[(errorData["R"] == R) & (errorData["N"] == N)]
            df = len(subset) - 1
            p = 0.975 # For a 95% CI
            tcrit = sp.stats.t.ppf(p, df)
            label = "R = " + str(R) + ", \n N = " + str(N)

            # print(subset["errorRatio" + param])

            errorLowMean = np.mean(subset["errorRatioLow"])
            errorLowSD = np.std(subset["errorRatioLow"])
            errorLowSE = errorLowSD / np.sqrt(len(subset))
            # print("Low SE:", errorLowSE)
            errorLowCI = [errorLowMean - tcrit * errorLowSE, errorLowMean + tcrit * errorLowSE]

            errorHighMean = np.mean(subset["errorRatioHigh"])
            errorHighSD = np.std(subset["errorRatioHigh"])
            errorHighSE = errorHighSD / np.sqrt(len(subset))
            # print("High SE:", errorHighSE)
            errorHighCI = [errorHighMean - tcrit * errorHighSE, errorHighMean + tcrit * errorHighSE]
            
            bars[label] = {"Low" : [errorLowMean, tcrit * errorHighSE], "High" : [errorHighMean, tcrit * errorHighSE]}
            X_space.append(label)
    
    combos = []
    lowPoints = []
    lowBars = []
    highPoints = []
    highBars = []
    for combo in X_space:
        combos.append(combo)
        lowPoints.append(np.mean(bars[combo]["Low"][0])) 
        lowBars.append(bars[combo]["Low"][1])

        highPoints.append(np.mean(bars[combo]["High"][0])) 
        highBars.append(bars[combo]["High"][1])

    combos = np.array(combos)
    lowPoints = np.array(lowPoints)
    lowBars = np.array(lowBars)
    highPoints = np.array(highPoints)
    highBars = np.array(highBars)

    df = pd.DataFrame({
        "combo" : combos,
        "lowPoint" : lowPoints,
        "lowBar" : lowBars,
        "highPoint" : highPoints,
        "highBar" : highBars
    })
    
    plt.rc("text", usetex = True)
    plt.rc("font", family = "serif")

    ax = plt.axes()
    plt.errorbar(x = df["combo"], y = df["lowPoint"], yerr = df["lowBar"], label = "Error: Lower Bound", fmt = "o", alpha = 0.5, capsize = 3, color = "blue")
    plt.errorbar(x = df["combo"], y = df["highPoint"], yerr = df["highBar"], label = "Error: Upper Bound", fmt = "o", alpha = 0.5, capsize = 3, color = "red")

    if value == "L4":
        ax.set_title(r"Calculated Errors in Inferring $\phi_{ngram-average}$ Parameter")
    
    else:
        ax.set_title(r"Calculated Errors in Inferring $\phi_{ngram-argmax}$ Parameter")
    
    ax.set_xlabel("(R, N) Combination")
    ax.set_ylabel("Inference Error")
    ax.axhline(y = 0, linestyle = "--", color = "black", alpha = 0.15)
    plt.gcf().subplots_adjust(bottom = 0.15)
    
    # ax.set_xticklabels(rotation = (15), labels = df["combo"])
    ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1), ncol = 2, fancybox = True, shadow = True)
    
    if value == "L4":
        plt.savefig("ratioErrorsL4.jpg", dpi = 1200, format = "jpg")
    else:
        plt.savefig("ratioErrorsL2.jpg", dpi = 1200, format = "jpg")
    
    plt.show()

def analyzeErrors_Phi(errorData, params = ["Phi", "Gamma"], N_vals = [200, 400], R_vals = [2, 4, 6], value = "L2"):
    for param in params:
        errorData["errorLow" + param] = errorData["infer" + param + "Low"] - errorData["true" + param]
        errorData["errorHigh" + param] = errorData["infer" + param + "High"] - errorData["true" + param]

    bars = {}

    for param in params:
        X_space = []
        for R in R_vals:
            for N in N_vals:           
                subset = errorData[(errorData["R"] == R) & (errorData["N"] == N)]
                df = len(subset) - 1
                p = 0.975 # For a 95% CI
                tcrit = sp.stats.t.ppf(p, df)
                label = "R = " + str(R) + ", \n N = " + str(N)

                errorLowMean = np.mean(subset["errorLow" + param])
                errorLowSD = np.std(subset["errorLow" + param])
                errorLowSE = errorLowSD / np.sqrt(len(subset))
                # print("Low SE:", errorLowSE)
                errorLowCI = [errorLowMean - tcrit * errorLowSE, errorLowMean + tcrit * errorLowSE]

                errorHighMean = np.mean(subset["errorHigh" + param])
                errorHighSD = np.std(subset["errorHigh" + param])
                errorHighSE = errorHighSD / np.sqrt(len(subset))
                # print("High SE:", errorHighSE)
                errorHighCI = [errorHighMean - tcrit * errorHighSE, errorHighMean + tcrit * errorHighSE]
                
                bars[label] = {"Low" : [errorLowMean, tcrit * errorHighSE], "High" : [errorHighMean, tcrit * errorHighSE]}
                X_space.append(label)
        
        # Plot stuff here
        combos = []
        lowPoints = []
        lowBars = []
        highPoints = []
        highBars = []
        for combo in X_space:
            combos.append(combo)
            lowPoints.append(np.mean(bars[combo]["Low"][0])) 
            lowBars.append(bars[combo]["Low"][1])

            highPoints.append(np.mean(bars[combo]["High"][0])) 
            highBars.append(bars[combo]["High"][1])

        combos = np.array(combos)
        lowPoints = np.array(lowPoints)
        lowBars = np.array(lowBars)
        highPoints = np.array(highPoints)
        highBars = np.array(highBars)

        df = pd.DataFrame({
            "combo" : combos,
            "lowPoint" : lowPoints,
            "lowBar" : lowBars,
            "highPoint" : highPoints,
            "highBar" : highBars
        })

        print(df["combo"])

        plt.rc("text", usetex = True)
        plt.rc("font", family = "serif")

        ax = plt.axes()
        plt.errorbar(x = df["combo"], y = df["lowPoint"], yerr = df["lowBar"], label = "Error: Lower Bound", fmt = "o", alpha = 0.5, capsize = 3, color = "blue")
        plt.errorbar(x = df["combo"], y = df["highPoint"], yerr = df["highBar"], label = "Error: Upper Bound", fmt = "o", alpha = 0.5, capsize = 3, color = "red")

        
        if param == "Phi" and value == "L2":
            ax.set_title(r"Calculated Errors in Inferring $\phi_{ngram-argmax}$ Parameter")
        elif param == "Phi" and value == "L4":
            ax.set_title(r"Calculated Errors in Inferring $\phi_{ngram-average}$ Parameter")
        elif param == "Gamma":
            ax.set_title(r"Calculated Errors in Inferring $\gamma$ Parameter")
        
        ax.set_xlabel("(R, N) Combination")
        ax.set_ylabel("Inference Error")
        ax.axhline(y = 0, linestyle = "--", color = "black", alpha = 0.15)
        plt.gcf().subplots_adjust(bottom = 0.15)
        
        # ax.set_xticklabels(rotation = (15), labels = df["combo"])
        ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1), ncol = 2, fancybox = True, shadow = True)
        plt.savefig(param + "Errors_highres.jpg", format = "jpg", dpi = 1200)
        plt.show()


    return errorData

def analyzePhi(phiGrid, value = "L2", gammaInterest = 1):
    sub = phiGrid[phiGrid["Gamma"] == gammaInterest]

    plt.rc("text", usetex = True)
    plt.rc("font", family = "serif")

    ax = plt.axes()
    plt.plot(sub["Phi"], sub[value])
    
    ax.set_xlabel(r"$\phi$ Parameter")
    
    if value == "L4":
        ax.set_title(r"Log Likelihood of $n$-gram Average as a Function of $\phi$ ($\gamma = 1, N = 200$)")
        ax.set_ylabel(r"Log Likelihood of $n$-gram Average Strategy")
    else:
        ax.set_ylabel(r"Log Likelihood of $n$-gram Argmax Strategy")
        ax.set_title(r"Log Likelihood of $n$-gram Argmax as a Function of $\phi$ ($\gamma = 1, N = 200$)")

    plt.show()

def analyzePhi_contour(phiGrid, value = "L2"):
    # mesh = np.meshgrid(phiGrid["Phi"], phiGrid["Gamma"], phiGrid["L"])

    phiGrid = phiGrid[phiGrid["Phi"] > 0.0]
    
    phi_unique = pd.unique(phiGrid["Phi"])
    phiCounter = 0
    gamma_unique = pd.unique(phiGrid["Gamma"])
    L = np.zeros(shape = (len(gamma_unique), len(phi_unique)))

    for phi in phi_unique:
        gammaCounter = 0
        for gamma in gamma_unique:
            sub = phiGrid[(phiGrid["Phi"] == phi) & (phiGrid["Gamma"] == gamma)]
            # print(sub["L"], "     ", type(sub["L"]))
            L[gammaCounter][phiCounter] = sub[value]
            gammaCounter += 1
        phiCounter += 1
    
    plt.rc("text", usetex = True)
    plt.rc("font", family = "serif")
    ax = plt.axes()

    CS = ax.contour(phi_unique, gamma_unique, L, levels = 10)
    ax.clabel(CS, inline = 1, fontsize = 10)

    
    ax.set_ylabel(r"$\gamma$ Parameter", size = "large")
    
    if value == "L4":
        ax.set_title(r"Contour Plot of Log Likelihoods for $n$-gram Average Strategy")
        ax.set_xlabel(r"$\phi_{ngram-average}$ Parameter", size = "large")
    
    else:
        ax.set_title(r"Contour Plot of Log Likelihoods for $n$-gram Argmax Strategy")
        ax.set_xlabel(r"$\phi_{ngram-argmax}$ Parameter", size = "large")
    
    plt.savefig("contour_" + value + "_phi_highres.jpg", format = "jpg", dpi = 1200)

    plt.show()

def analyzePhi_surface(phiGrid, value = "L2"):
    phiGrid = phiGrid[phiGrid["Phi"] > 0.0]

    plt.rc("text", usetex = True)
    plt.rc("font", family = "serif")

    fig = plt.figure()
    ax = fig.gca(projection = "3d")

    if value == "L4":
        surf = ax.plot_trisurf(phiGrid.Phi, phiGrid.Gamma, phiGrid.L4, linewidth = 0)
        ax.set_zlabel(r"Log Likelihood of $n$-gram Average Strategy")
        ax.set_xlabel(r"$\phi_{ngram-average}$ Parameter", size = "large")

    else:
        surf = ax.plot_trisurf(phiGrid.Phi, phiGrid.Gamma, phiGrid.L2, linewidth = 0)
        ax.set_zlabel(r"Log Likelihood of $n$-gram Argmax Strategy")
        ax.set_xlabel(r"$\phi_{ngram-argmax}$ Parameter", size = "large")
    
    
    ax.set_ylabel(r"$\gamma$ Parameter", size = "large")

    plt.savefig("surface_" + value + "_phi_highres.jpg", format = "jpg", dpi = 1200)

    plt.show()



#----------------------------------------------
# Debugging Code
#----------------------------------------------

#paramTest = varyParams(data = testData, alphaMax = 5, betaMax = 1, step = 0.1, gamma = 1, R = 4)

#paramTestSub = paramTest[(paramTest.Alpha > 1.0) & (paramTest.Beta > 0.0)]

"""
# Creating the surface plot:

fig = plt.figure()
ax = fig.gca(projection = "3d")
surf = ax.plot_trisurf(paramTestSub.Alpha, paramTestSub.Beta, paramTestSub.L2, linewidth = 0)
ax.set_xlabel("Concentration Hyperparameter")
ax.set_ylabel("Updating Probability")
ax.set_zlabel('Log Likelihood (L2)')

plt.show()

"""

"""
pbounds = {
    "alpha" : (1.1, 5),
    "beta" : (0.1, 1),
}

optimizer = BayesianOptimization(
    f = lambda alpha, beta: calc_L2(data = testData, R = 5 , gamma = 1, alphaTrue = 3, betaTrue = 0.75, gammaTrue = 1, alpha = alpha, beta = beta),
    pbounds = pbounds,
    random_state = 1
)

optimizer.maximize(init_points = 15, n_iter = 180)
"""

"""
True hyperparameter values
Run 1: (alpha, beta, gamma) = (1, 1, 2)
Run 2: (alpha, beta, gamma) = (2.5, 0.8, 1.5)
Run 3: (alpha, beta, gamma) = (4.03, 0.73, 2.34)
"""

# test = varyParams(testData, R = 5, alphaMax = 10, betaMax = 1, step = 0.1, alphaTrue = 3, betaTrue = 0.75, gammaTrue = 1)
# test = varyParams_gamma(testData, R = 4, alphaMax = 5, betaMax = 1, gammaMax = 4, step = 0.1, alphaTrue = 4.03, betaTrue = 0.73, gammaTrue = 2.34)
# test.to_csv("N200_AlphaBetaGammaRun3.csv")
# paramTestSub = test[(test.Alpha > 1.0) & (test.Beta > 0.0)]

# test = ngram_observer(testData, R = 4, alpha = 3, beta = 1, gamma = 1, recordLiks = True)


# data = ErrorCalcsL2(R_vals = [1, 2], n_vals = [200, 400], alphaTrue = 2, betaTrue = 1, gammaTrue = 1.3, n_iter = 4, dataAlpha = 2, alphaMax = 2.5, gammaMax = 2, step = 0.1)



if __name__ == "__main__":
    """
    Where the main code is run
    """
    __spec__ = None # Necessary to perform multiprocessing

    #testData = orderRMarkov_generation(R = 3, alpha = 2, length = 200)
    # t0 = time.time()
    # test = ErrorCalcsL2_phi(R_vals = [2], n_vals = [50], phiTrue = 0.25, gammaTrue = 1, n_iter = 4, dataAlpha = 2, phiMax = 2, gammaMax = 2, phiStep = 0.1, gammaStep = 0.1)
    # test = evalStrategy(n_each = 100, lags = 5, alphaTrue = 3, betaTrue = 1, gammaTrue = 0.8, sampleSize = 500, numObs = 2)
    # res = ErrorCalcsL4(R_vals = [6], n_vals = [400, 600], alphaTrue = 2.3, betaTrue = 0.8, gammaTrue = 1.7, n_iter = 16, dataAlpha = 2, alphaMax = 5, gammaMax = 4, step = 0.1)
    # res = ErrorCalcsL2(R_vals = [2], n_vals = [200, 400], alphaTrue = 4.5, betaTrue = 0.7, gammaTrue = 1.8, n_iter = 6, dataAlpha = 2, alphaMax = 5, gammaMax = 4, step = 0.1)
    # res = varyParams_predL4_copy(testData, alphaTrue = 2.7, betaTrue = 0.8, gammaTrue = 1.5, R = 4)
    # res = varyParams_phi_gamma_L2(data = testData, R = 4, beta = 1, phiMax = 2, step_phi = 0.01, phiTrue = 0.27, gammaMax = 2, step_gamma = 0.1, gammaTrue = 1)
    # res = varyParams_phi_gamma_L2(data = testData, R = 4, beta = 1, phiMax = 2, step_phi = 0.01, phiTrue = 0.27, gammaMax = 2, step_gamma = 0.1, gammaTrue = 1)
    # tf = time.time()
    # print(f"Time Elapsed: {tf - t0} seconds.")
    # time.sleep(1) # Makes sure all child processes are killed

    # test.to_csv("strategyInference_redo.csv", index = False)
    # res.to_csv("L2ErrorCalcsR2_RETRY3.csv", index = False)
    # test = pd.read_csv("L2ErrorsFInal.csv")
    # analyzeErrors(test)

    # errors = pd.read_csv("L4ErrorsFinal.csv")
    # analyzeErrors_Phi(errors)
    # analyzeErrors_ratio(errors, value = "L4")


    # paramTestSub = pd.read_csv("alphaBetaL2Part2_N600.csv")
    # paramTestSub = pd.read_csv("AlphaBetaL2Data.csv")
    # grid = pd.read_csv("phiGammaL2.csv")
    # analyzePhi_contour(grid, value = "L2")
    # analyzePhi_surface(grid, value = "L2")
    
    # paramTestSub = paramTestSub[(paramTestSub["Alpha"] > 1.0) & paramTestSub["Beta"] > 0]
    # paramTestSub["Phi"] = paramTestSub["Beta"] / (paramTestSub["Alpha"] - 1)

    # plt.scatter(paramTestSub["Phi"], paramTestSub["L2"])

    actual = str_to_lst("0010011001111001100100100110010011100100100110011111100110011001001001111110011110011100111111110011001111111110010010010011001111110010011111100110011001001001111001111001111111001100111111111001110010011110011001001111111001100100100110010011111001110011111111100100100100111001111100111110011001111001100111100100111111100111001100100110011111111100100110011111100100111111111111111111001001110011001111001111001001100111001001110011111111110011100111111111110010011100111100111111001110011111100100111111111100100111001001111001100110011100100111110011001001111100111111111100110011001001111111100111111100100111001100110010011111111")
    guess = str_to_lst("1010100000000000110101000010100100010101010101111111110011100101101010011111001111011011111111111001111111111111001111000000000011111100001111110111011101100111111101111101111111101110111111111101111011011111011101101111111101110110110111011011111101111011111111110110110110111101111110111111011101111101110111110110111111110111101110110101011111111110110111011111110110111111111111111111101101111011101111101111101101110111101101111011111111111011110111111111111011011110111110111111101111011111110110111111111110110111101101111101110111011110110111111011101101111110111111111110111011101101111111110111111110110111101110111011011111111")

    strat = predFunc_guess(actual, guess)
    """

    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    surf = ax.plot_trisurf(paramTestSub.Alpha, paramTestSub.Beta, paramTestSub.L2, linewidth = 0)
    ax.set_xlabel("Concentration Hyperparameter")
    ax.set_ylabel("Updating Probability")
    ax.set_zlabel('Log Likelihood (L2)')

    plt.show()

    contourGrid = np.meshgrid(paramTestSub.Alpha, paramTestSub.Beta, paramTestSub.L2)
    countourGrid = contourGrid.T()
    plt.contour(contourGrid)
    """
