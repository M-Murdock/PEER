import numpy as np

class BayesianPredictor:
    def __init__(self, policies, action_space_size=4, prior=None, tau=0.8, eps=1e-3):
        self.policies = policies
        self.N = len(policies)
        self.action_space_size = action_space_size
        
        # smoothing hyperparameter for posterior
        self.eps = eps  
        self.tau = tau  # likelihood temperature

        # log posterior
        if prior is None:
            prior = np.ones(self.N) / self.N
        
        self.log_post = np.log(prior + 1e-12)

    def log_likelihood(self, state, user_action, policy):
        """Return log P(u | pi)."""
        Q = np.array([policy.get_q_value(state, a) for a in range(self.action_space_size)])

        # softmax likelihood P(u | pi)
        logits = Q / self.tau
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        return np.log(probs[user_action] + 1e-12)

    def update(self, state, user_action):
        log_likes = np.zeros(self.N)
        
        for i, pi in enumerate(self.policies):
            log_likes[i] = self.log_likelihood(state, user_action, pi)

        # log posterior update
        self.log_post += log_likes
        
        # normalize in log-space
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        post /= np.sum(post)

        # add smoothing (prevents zeros)
        post = (1 - self.eps) * post + self.eps * (1.0 / self.N)

        # store back in log-space
        self.log_post = np.log(post + 1e-12)

        return post

    def get_prob(self):
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        return post / np.sum(post)


class MaxEntPredictor:
    def __init__(self, policies, action_space_size=4, tau=0.8, eps=1e-3):
        """
        policies: list of policies to evaluate
        action_space_size: number of discrete actions
        tau: softmax temperature
        eps: smoothing to prevent zeros
        """
        self.policies = policies
        self.N = len(policies)
        self.action_space_size = action_space_size
        self.tau = tau
        self.eps = eps

        # initialize uniform belief over policies
        self.log_post = np.log(np.ones(self.N) / self.N + 1e-12)

    def log_likelihood(self, state, user_action, policy):
        """MaxEnt IOC likelihood: P(u | pi) proportional to exp(Q(s,u)/tau)."""
        Q = np.array([policy.get_q_value(state, a) for a in range(self.action_space_size)])
        logits = Q / self.tau
        logits -= np.max(logits)  # for numerical stability
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.log(probs[user_action] + 1e-12)

    def update(self, state, user_action):
        """
        Update the belief over policies using MaxEnt likelihood.
        """
        log_likes = np.zeros(self.N)
        for i, pi in enumerate(self.policies):
            log_likes[i] = self.log_likelihood(state, user_action, pi)

        # log posterior update
        self.log_post += log_likes

        # normalize in log-space
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        post /= np.sum(post)

        # smoothing
        post = (1 - self.eps) * post + self.eps * (1.0 / self.N)

        # store back in log-space
        self.log_post = np.log(post + 1e-12)

        return post

    def get_prob(self):
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        return post / np.sum(post)
