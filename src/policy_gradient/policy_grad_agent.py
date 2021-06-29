import tensorflow as tf
from tensorflow import keras
from human_aware_rl_master.overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy

DEFAULT_MLP_PARAMS = {
    # Number of fully connected layers to use in our network
    "num_layers" : 2,
    # Each int represents a layer of that hidden size
    "net_arch" : [64, 64],
    "cell_size" : 256
}


class LSTMPolicy(NNPolicy):
    def __init__(self, observation_shape, action_shape, mlp_params, cell_size, max_seq_len=20, **kwargs):
        ## Inputs
        obs_in = keras.Input(shape=(None, *observation_shape), name="Overcooked_observation")
        seq_in = keras.Input(shape=(), name="seq_in", dtype=tf.int32)
        h_in = keras.Input(shape=(cell_size,), name="hidden_in")
        c_in = keras.Input(shape=(cell_size,), name="memory_in")
        x = obs_in

        ## Build fully connected layers
        assert len(mlp_params["net_arch"]) == mlp_params["num_layers"], "Invalid Fully Connected params"

        for i in range(mlp_params["num_layers"]):
            units = mlp_params["net_arch"][i]
            x = keras.layers.TimeDistributed(keras.layers.Dense(units, activation="relu", name="fc_{0}".format(i)))(x)

        mask = keras.layers.Lambda(lambda x : tf.sequence_mask(x, maxlen=max_seq_len))(seq_in)

        ## LSTM layer
        lstm_out, h_out, c_out = keras.layers.LSTM(cell_size, return_sequences=True, return_state=True, stateful=False, name="lstm")(
            inputs=x,
            mask=mask,
            initial_state=[h_in, c_in]
        )

        ## output layer
        logits = keras.layers.TimeDistributed(keras.layers.Dense(action_shape[0]), name="logits")(lstm_out)

        self.model = keras.Model(inputs=[obs_in, seq_in, h_in, c_in], outputs=[logits, h_out, c_out])
    
    def multi_state_policy(self, states, agent_indices):
        """
        A function that takes in multiple OvercookedState instances and their respective agent indices and returns action probabilities.
        """
        # TODO: preprocess states

        return self.multi_obs_policy(states)

    def multi_obs_policy(self, states):
        """
        A function that takes in multiple preprocessed OvercookedState instatences and returns action probabilities.
        """
        return self.model.predict(states)


def get_lstm_agent(policy_params, new_model=True):
    if new_model:
        policy = LSTMPolicy(*policy_params)
        return AgentFromPolicy(policy)
    else:
        path = policy_params['model_path']
        # TODO: load pickle model.
        return None