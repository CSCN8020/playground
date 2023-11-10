class Policy:
    def select_action(self, state):
        pass


class DeterministicPolicy(Policy):
    def update(self, state, action):
        pass


class StochasticPolicy(Policy):
    def update(self, states, actions, rewards):
        pass

    def get_probability(self, state, action):
        pass
