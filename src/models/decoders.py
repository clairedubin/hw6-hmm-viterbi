import copy
import numpy as np
class ViterbiAlgorithm:
    """_summary_
    """    

    def __init__(self, hmm_object):
        """_summary_

        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """        
        
        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]

        best_path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))        
        
        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        # 2. Scale      

        first_obs_idx = self.hmm_object.observation_states_dict[decode_observation_states[0]]
        delta = self.hmm_object.prior_probabilities * self.hmm_object.emission_probabilities[:,first_obs_idx]

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for node in range(1, len(decode_observation_states)):

            # TODO: comment the initialization, recursion, and termination steps

            obs = decode_observation_states[node]
            obs_idx = self.hmm_object.observation_states_dict[obs]
            probs=[]            

            for hidden_state_idx,hidden_state in enumerate(self.hmm_object.hidden_states):
                    
                prob = delta[hidden_state_idx]*self.hmm_object.transition_probabilities[hidden_state_idx, :]*self.hmm_object.emission_probabilities[:, obs_idx]
                probs += [prob]

            prob_matrix = np.vstack([np.array(prob) for prob in probs])
            delta = np.max(prob_matrix, axis=0)    
            path[node] = np.argmax(prob_matrix, axis=0)
                
           

        # Select the last hidden state, given the best path (i.e., maximum probability)
        prev_idx = np.argmax(delta)
        prev_val = self.hmm_object.hidden_states_dict[prev_idx]
        best_hidden_state_path = [prev_val]
                
        for i in range(1, path.shape[0])[::-1]:
            prev_idx = path[i, int(prev_idx)]
            best_hidden_state_path = [self.hmm_object.hidden_states_dict[prev_idx]] + best_hidden_state_path

        return best_hidden_state_path