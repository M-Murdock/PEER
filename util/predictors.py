import numpy as np

class BayesianPredictor:
    # def __init__(self, policies, action_space_size=4, prior=None, tau=0.8, eps=1e-3):
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
        
        # return np.log(probs[user_action] + 1e-12)
        return np.log(probs[user_action] + 1e-8)

    def update( 
            self,
            state,
            user_action,
            alpha=0.05,        # forgetting / adaptation rate
            p_switch=0.02,     # goal-switch prior
            beta=1           # posterior temperature (>1 = smoother)
        ):
        """
        Bayesian update with forgetting, goal persistence, and posterior smoothing.
        """

        # 1. Compute log-likelihoods
        log_likes = np.zeros(self.N)
        for i, pi in enumerate(self.policies):
            log_likes[i] = self.log_likelihood(state, user_action, pi)

        # 2. Exponential forgetting (key fix)
        # Blends past belief with current evidence
        self.log_post = (1 - alpha) * self.log_post + alpha * log_likes

        # 3. Normalize in probability space
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        post /= np.sum(post)

        # 4. Goal-switch prior (intent persistence)
        if p_switch > 0:
            post = (1 - p_switch) * post + p_switch * (1.0 / self.N)

        # 5. Posterior temperature (optional but useful)
        if beta != 1.0:
            post = post ** (1.0 / beta)
            post /= np.sum(post)

        # 6. Light smoothing (numerical safety only)
        post = (1 - self.eps) * post + self.eps * (1.0 / self.N)

        # 7. Store back in log space
        self.log_post = np.log(post + 1e-12)

        return post


    def get_prob(self):
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        return post / np.sum(post)


class MaxEntPredictor:
    def __init__(self, policies, action_space_size=4, tau=0.8, eps=1e-2):
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

    def update(self, state, user_action, alpha=0.5):
        """
        Update the belief over policies using MaxEnt likelihood.
        alpha: learning rate for smoothing the posterior
        """
        log_likes = np.zeros(self.N)
        for i, pi in enumerate(self.policies):
            log_likes[i] = self.log_likelihood(state, user_action, pi)

        # Convert to probability
        likes = np.exp(log_likes - np.max(log_likes))
        likes /= np.sum(likes)

        # Current posterior in probability space
        post = np.exp(self.log_post - np.max(self.log_post))
        post /= np.sum(post)

        # Exponentially weighted update
        post = (1 - alpha) * post + alpha * likes

        # smoothing
        post = (1 - self.eps) * post + self.eps * (1.0 / self.N)

        # store back in log-space
        self.log_post = np.log(post + 1e-12)

        return post


    def get_prob(self):
        max_logp = np.max(self.log_post)
        post = np.exp(self.log_post - max_logp)
        return post / np.sum(post)



import numpy as np


class CRFPredictor:
    """
    Linear-chain CRF predictor over policy hypotheses.
    Designed to avoid action-lock and belief collapse.
    """

    def __init__(
        self,
        policies,
        action_space_size=4,
        eps=0.1,
        tau=3.0,
        pairwise_weight=0.1,
        max_log_like=5.0,
        min_entropy=0.6,
        memory_decay=0.7,
    ):
        self.policies = policies
        self.N = len(policies)
        self.action_space_size = action_space_size

        self.eps = eps
        self.tau = tau
        self.pairwise_weight = pairwise_weight
        self.max_log_like = max_log_like
        self.min_entropy = min_entropy
        self.memory_decay = memory_decay

        self.log_post = np.log(np.ones(self.N) / self.N + 1e-12)
        self.prev_action = None

    # --------------------------------------------------
    # CRF likelihood
    # --------------------------------------------------

    def log_likelihood(self, state, user_action, policy):
        logits = np.zeros(self.action_space_size)

        for a in range(self.action_space_size):
            unary = self.unary_fn(policy, state, a)

            pair = 0.0
            if self.prev_action is not None:
                pair = self.pairwise_fn(self.prev_action, a)

            logits[a] = unary + pair

        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

        logp = np.log(probs[user_action] + 1e-12)

        # CRITICAL: clip likelihood to prevent runaway certainty
        return np.clip(logp, -self.max_log_like, self.max_log_like)

    # --------------------------------------------------
    # Update
    # --------------------------------------------------

    def update(self, state, user_action):
        log_likes = np.array([
            self.log_likelihood(state, user_action, pi)
            for pi in self.policies
        ])

        self.log_post += log_likes

        # normalize
        max_log = np.max(self.log_post)
        post = np.exp(self.log_post - max_log)
        post /= np.sum(post)

        # entropy floor (prevents belief lock)
        entropy = -np.sum(post * np.log(post + 1e-12))
        if entropy < self.min_entropy:
            post = (1 - self.eps) * post + self.eps / self.N

        self.log_post = np.log(post + 1e-12)

        # deterministic CRF memory decay
        if self.prev_action is None:
            self.prev_action = user_action
        else:
            if np.random.rand() < self.memory_decay:
                self.prev_action = user_action
            else:
                self.prev_action = None

        return post

    # --------------------------------------------------
    # Accessors
    # --------------------------------------------------

    def get_prob(self):
        max_log = np.max(self.log_post)
        post = np.exp(self.log_post - max_log)
        return post / np.sum(post)

    def reset(self):
        self.log_post = np.log(np.ones(self.N) / self.N + 1e-12)
        self.prev_action = None

    # --------------------------------------------------
    # Potentials
    # --------------------------------------------------

    def unary_fn(self, policy, state, action):
        Q = policy.get_q_value(state, action)
        return Q / self.tau

    def pairwise_fn(self, prev_a, a):
        """
        Encourage *change* rather than repetition.
        This is the key difference.
        """
        if prev_a == a:
            return -self.pairwise_weight
        return self.pairwise_weight
