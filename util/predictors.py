import numpy as np

class BayesianPredictor:
    def __init__(self, policies, action_space_size=4, prior=None, tau=0.8, eps=0.2):
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



class CRFPredictor:
    """
    Linear-chain CRF predictor over policy hypotheses.
    Maintains P(pi | user action sequence) using CRF potentials.
    """

    def __init__(
        self,
        policies,
        action_space_size=4,
        eps=1e-3
    ):
        self.policies = policies
        self.N = len(policies)
        self.action_space_size = action_space_size
        self.eps = eps

        # posterior over policies (log space)
        self.log_post = np.log(np.ones(self.N) / self.N + 1e-12)

        # store last action for pairwise potentials
        self.prev_action = None

    def log_likelihood(self, state, user_action, policy):
        """
        CRF likelihood for a single step:
            φ(s_t, u_t | π) + ψ(u_{t-1}, u_t | π)
        normalized internally.
        """

        # compute all φ + ψ for all actions, because we must normalize
        logits = np.zeros(self.action_space_size)

        for a in range(self.action_space_size):
            unary = self.unary_fn(policy, state, a)
            pair = 0.0
            if self.prev_action is not None:
                pair = self.pairwise_fn(policy, self.prev_action, a)
            logits[a] = unary + pair

        # CRF softmax normalization
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return np.log(probs[user_action] + 1e-12)

    def update(self, state, user_action):
        """
        Update belief over policies using CRF potentials.
        Mirrors BayesianPredictor and MaxEntPredictor.
        """

        log_likes = np.zeros(self.N)

        for i, pi in enumerate(self.policies):
            log_likes[i] = self.log_likelihood(state, user_action, pi)

        # posterior update in log-space
        self.log_post += log_likes

        # normalize
        max_log = np.max(self.log_post)
        post = np.exp(self.log_post - max_log)
        post /= np.sum(post)

        # smoothing
        post = (1 - self.eps) * post + self.eps * (1.0 / self.N)
        self.log_post = np.log(post + 1e-12)

        # update CRF chain memory
        self.prev_action = user_action

        return post

    def get_prob(self):
        max_log = np.max(self.log_post)
        post = np.exp(self.log_post - max_log)
        return post / np.sum(post)
    
    def unary_fn(self, policy, state, action):
        Q = policy.get_q_value(state, action)
        return Q / 0.8   # temperature τ

    def pairwise_fn(self, policy, prev_a, a):
        return 1.0 if prev_a == a else -0.5
