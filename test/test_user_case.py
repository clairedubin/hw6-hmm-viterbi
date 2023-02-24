"""
UCSF BMI203: Biocomputing Algorithms
Author:
Date: 
Program: 
Description:
"""
import pytest
import numpy as np
from src.models.hmm import HiddenMarkovModel
from src.models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm

    #check that hidden states are unique
    assert len(set(use_case_one_viterbi.hmm_object.hidden_states)) == len(use_case_one_viterbi.hmm_object.hidden_states)
    #check that observation states are unique
    assert len(set(use_case_one_viterbi.hmm_object.observation_states)) == len(use_case_one_viterbi.hmm_object.observation_states)
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    #changed to exclude last element of each path because the example included a hidden state
    assert np.alltrue(use_case_decoded_hidden_states[:-1] == use_case_one_data['hidden_states'][:-1])


def test_user_case_two():
    """_summary_
    """
    prior_probabilities=np.array([0.5, 0.5])
    transition_probabilities=np.array([[0.75, 0.25], 
                                    [0.3, 0.7]])
    emission_probabilities=np.array([[0.8, 0.2],
                                [0.4, 0.6]])
    hidden_states = ['windy', 'calm']
    observation_states = ['birds in park', 'no birds in park']
    decode_observation_states = ['birds in park', 'no birds in park','birds in park']
    correct_hidden_states = ['windy', 'windy', 'windy',]

    hmm_test = HiddenMarkovModel(hidden_states=hidden_states,
                      prior_probabilities=prior_probabilities,
                      transition_probabilities=transition_probabilities,
                      emission_probabilities=emission_probabilities,
                      observation_states=observation_states)
    
    v = ViterbiAlgorithm(hmm_object=hmm_test)
    output_seq = v.best_hidden_state_sequence(decode_observation_states)

    assert correct_hidden_states == output_seq


def test_user_case_three():
    """_summary_
    """
    prior_probabilities=np.array([0.2, 0.8])
    transition_probabilities=np.array([[0.6, 0.4], 
                                    [0.4, 0.6]])
    emission_probabilities=np.array([[0.5, 0.5],
                                [0.9, 0.1]])
    hidden_states = ['ate breakfast', 'no breakfast']
    observation_states = ['tired', 'energetic']
    decode_observation_states = ['energetic', 'energetic','tired', 'tired']
    correct_hidden_states = ['ate breakfast','ate breakfast', 'no breakfast', 'no breakfast']

    hmm_test = HiddenMarkovModel(hidden_states=hidden_states,
                      prior_probabilities=prior_probabilities,
                      transition_probabilities=transition_probabilities,
                      emission_probabilities=emission_probabilities,
                      observation_states=observation_states)
    
    v = ViterbiAlgorithm(hmm_object=hmm_test)
    output_seq = v.best_hidden_state_sequence(decode_observation_states)

    assert correct_hidden_states == output_seq
