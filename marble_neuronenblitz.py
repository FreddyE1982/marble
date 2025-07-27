from marble_imports import *
from marble_core import Neuron, SYNAPSE_TYPES, NEURON_TYPES, perform_message_passing
from contrastive_learning import ContrastiveLearner
from imitation_learning import ImitationLearner
import threading
import multiprocessing as mp
import pickle
import random
import numpy as np
from datetime import datetime, timezone
from collections import deque
import math


def _wander_worker(state_bytes: bytes, input_value: float, seed: int) -> tuple[float, int]:
    nb = pickle.loads(state_bytes)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    output, _ = nb.dynamic_wander(input_value)
    return output, seed


def default_combine_fn(x: float, w: float) -> float:
    return max(x * w, 0)


def default_loss_fn(target: float, output: float) -> float:
    return target - output


def default_weight_update_fn(
    source: float | None, error: float | None, path_len: int
) -> float:
    if source is None:
        source = 0.0
    if error is None:
        error = 0.0
    return (error * source) / (path_len + 1)


def default_q_encoding(state: tuple[int, int], action: int) -> float:
    """Encode a state-action pair into a numeric value."""
    return float(state[0] * 10 + state[1] + action / 10)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Neuronenblitz:
    def __init__(
        self,
        core,
        backtrack_probability=0.3,
        consolidation_probability=0.2,
        consolidation_strength=1.1,
        route_potential_increase=0.5,
        route_potential_decay=0.9,
        route_visit_decay_interval=10,
        alternative_connection_prob=0.1,
        split_probability=0.2,
        merge_tolerance=0.01,
        combine_fn=None,
        loss_fn=None,
        loss_module=None,
        weight_update_fn=None,
        plasticity_threshold=10.0,
        continue_decay_rate=0.85,
        struct_weight_multiplier1=1.5,
        struct_weight_multiplier2=1.2,
        attention_decay=0.9,
        max_wander_depth=100,
        learning_rate=0.01,
        weight_decay=0.0,
        dropout_probability=0.0,
        dropout_decay_rate=1.0,
        exploration_decay=0.99,
        reward_scale=1.0,
        stress_scale=1.0,
        remote_fallback=False,
        noise_injection_std=0.0,
        dynamic_attention_enabled=True,
        backtrack_depth_limit=10,
        synapse_update_cap=1.0,
        structural_plasticity_enabled=True,
        backtrack_enabled=True,
        loss_scale=1.0,
        exploration_bonus=0.0,
        synapse_potential_cap=100.0,
        attention_update_scale=1.0,
        plasticity_modulation=1.0,
        wander_depth_noise=0.0,
        reward_decay=1.0,
        synapse_prune_interval=10,
        structural_learning_rate=0.1,
        remote_timeout=2.0,
        gradient_noise_std=0.0,
        min_learning_rate=0.0001,
        max_learning_rate=0.1,
        top_k_paths=5,
        parallel_wanderers=1,
        beam_width=1,
        synaptic_fatigue_enabled=True,
        fatigue_increase=0.05,
        fatigue_decay=0.95,
        lr_adjustment_factor=0.1,
        lr_scheduler="none",
        scheduler_steps=100,
        scheduler_gamma=0.99,
        epsilon_scheduler="none",
        epsilon_scheduler_steps=100,
        epsilon_scheduler_gamma=0.99,
        momentum_coefficient=0.0,
        reinforcement_learning_enabled=False,
        rl_discount=0.9,
        rl_epsilon=1.0,
        rl_epsilon_decay=0.95,
        rl_min_epsilon=0.1,
        shortcut_creation_threshold=5,
        use_echo_modulation=False,
        wander_cache_ttl=300,
        phase_rate=0.1,
        phase_adaptation_rate=0.05,
        chaotic_gating_enabled=False,
        chaotic_gating_param=3.7,
        chaotic_gate_init=0.5,
        context_history_size=10,
        context_embedding_decay=0.9,
        emergent_connection_prob=0.05,
        concept_association_threshold=5,
        concept_learning_rate=0.1,
        weight_limit=1e6,
        wander_cache_size=50,
        rmsprop_beta=0.99,
        grad_epsilon=1e-8,
        use_experience_replay=False,
        replay_buffer_size=1000,
        replay_alpha=0.6,
        replay_beta=0.4,
        replay_batch_size=32,
        exploration_entropy_scale=1.0,
        exploration_entropy_shift=0.0,
        gradient_score_scale=1.0,
        memory_gate_decay=0.99,
        memory_gate_strength=1.0,
        episodic_memory_size=50,
        episodic_memory_threshold=0.1,
        episodic_memory_prob=0.1,
        curiosity_strength=0.0,
        depth_clip_scaling=1.0,
        forgetting_rate=0.99,
        structural_dropout_prob=0.0,
        gradient_path_score_scale=1.0,
        use_gradient_path_scoring=True,
        activity_gate_exponent=1.0,
        subpath_cache_size=100,
        gradient_accumulation_steps=1,
        subpath_cache_ttl=300,
        use_mixed_precision=False,
        remote_client=None,
        torrent_client=None,
        torrent_map=None,
        metrics_visualizer=None,
    ):
        self.core = core
        self.backtrack_probability = backtrack_probability
        self.consolidation_probability = consolidation_probability
        self.consolidation_strength = consolidation_strength
        self.route_potential_increase = route_potential_increase
        self.route_potential_decay = route_potential_decay
        self.route_visit_decay_interval = route_visit_decay_interval
        self.alternative_connection_prob = alternative_connection_prob
        self.split_probability = split_probability
        self.merge_tolerance = merge_tolerance
        self.plasticity_threshold = plasticity_threshold
        self.continue_decay_rate = continue_decay_rate
        self.struct_weight_multiplier1 = struct_weight_multiplier1
        self.struct_weight_multiplier2 = struct_weight_multiplier2
        self.attention_decay = attention_decay
        self.max_wander_depth = max_wander_depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_probability = dropout_probability
        self.dropout_decay_rate = dropout_decay_rate
        self.exploration_decay = exploration_decay
        self.reward_scale = reward_scale
        self.stress_scale = stress_scale
        self.remote_fallback = remote_fallback
        self.noise_injection_std = noise_injection_std
        self.dynamic_attention_enabled = dynamic_attention_enabled
        self.backtrack_depth_limit = backtrack_depth_limit
        self.synapse_update_cap = synapse_update_cap
        self.structural_plasticity_enabled = structural_plasticity_enabled
        self.backtrack_enabled = backtrack_enabled
        self.loss_scale = loss_scale
        self.exploration_bonus = exploration_bonus
        self.synapse_potential_cap = synapse_potential_cap
        self.attention_update_scale = attention_update_scale
        self.plasticity_modulation = plasticity_modulation
        self.wander_depth_noise = wander_depth_noise
        self.reward_decay = reward_decay
        self.synapse_prune_interval = synapse_prune_interval
        self.structural_learning_rate = structural_learning_rate
        self.remote_timeout = remote_timeout
        self.gradient_noise_std = gradient_noise_std
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.top_k_paths = top_k_paths
        self.parallel_wanderers = parallel_wanderers
        self.beam_width = max(1, int(beam_width))
        self.synaptic_fatigue_enabled = synaptic_fatigue_enabled
        self.fatigue_increase = fatigue_increase
        self.fatigue_decay = fatigue_decay
        self.lr_adjustment_factor = lr_adjustment_factor
        self.lr_scheduler = lr_scheduler
        self.scheduler_steps = int(scheduler_steps)
        self.scheduler_gamma = float(scheduler_gamma)
        self._scheduler_step = 0
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon_scheduler_steps = int(epsilon_scheduler_steps)
        self.epsilon_scheduler_gamma = float(epsilon_scheduler_gamma)
        self._epsilon_step = 0
        self.initial_epsilon = rl_epsilon
        self.momentum_coefficient = momentum_coefficient
        self.rl_enabled = reinforcement_learning_enabled
        self.rl_discount = rl_discount
        self.rl_epsilon = rl_epsilon
        self.rl_epsilon_decay = rl_epsilon_decay
        self.rl_min_epsilon = rl_min_epsilon
        self.shortcut_creation_threshold = int(shortcut_creation_threshold)
        self.use_echo_modulation = use_echo_modulation
        self.phase_rate = phase_rate
        self.phase_adaptation_rate = phase_adaptation_rate
        self.global_phase = 0.0
        self.chaos_state = 0.5
        self.chaotic_gating_enabled = chaotic_gating_enabled
        self.chaotic_gating_param = chaotic_gating_param
        self.chaotic_gate = chaotic_gate_init
        self.context_history_size = int(context_history_size)
        self.context_embedding_decay = float(context_embedding_decay)
        self.emergent_connection_prob = float(emergent_connection_prob)
        self.concept_association_threshold = int(concept_association_threshold)
        self.concept_learning_rate = float(concept_learning_rate)

        self._weight_limit = float(weight_limit)

        self.combine_fn = combine_fn if combine_fn is not None else default_combine_fn
        self.loss_fn = loss_fn if loss_fn is not None else default_loss_fn
        self.loss_module = loss_module
        self.weight_update_fn = (
            weight_update_fn
            if weight_update_fn is not None
            else default_weight_update_fn
        )

        self.training_history = []
        self.global_activation_count = 0
        self.last_context = {}
        self.type_attention = {nt: 0.0 for nt in NEURON_TYPES}
        self.type_speed_attention = {nt: 0.0 for nt in NEURON_TYPES}
        self.synapse_loss_attention = {st: 0.0 for st in SYNAPSE_TYPES}
        self.synapse_size_attention = {st: 0.0 for st in SYNAPSE_TYPES}
        self.synapse_speed_attention = {st: 0.0 for st in SYNAPSE_TYPES}
        self.remote_client = remote_client
        self.torrent_client = torrent_client
        self.torrent_map = torrent_map if torrent_map is not None else {}
        self.metrics_visualizer = metrics_visualizer
        self.last_message_passing_change = 0.0
        self.lock = threading.RLock()
        self.error_history = deque(maxlen=100)
        self._momentum = {}
        self._eligibility_traces = {}
        self.wander_cache = {}
        self._cache_order = deque()
        self._cache_max_size = int(wander_cache_size)
        self.wander_cache_ttl = wander_cache_ttl
        self.subpath_cache = {}
        self._subpath_order = deque()
        self._subpath_cache_size = int(subpath_cache_size)
        self.subpath_cache_ttl = float(subpath_cache_ttl)
        self.q_encoding = default_q_encoding
        self._grad_sq = {}
        self._rmsprop_beta = float(rmsprop_beta)
        self._grad_epsilon = float(grad_epsilon)
        # Store previous gradients per synapse for alignment-based gating
        self._prev_gradients = {}
        # Track path usage for shortcut creation
        self._path_usage = {}
        self.context_history = deque(maxlen=self.context_history_size)
        self._concept_pairs = {}
        self.use_experience_replay = bool(use_experience_replay)
        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_alpha = float(replay_alpha)
        self.replay_beta = float(replay_beta)
        self.replay_batch_size = int(replay_batch_size)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.replay_priorities = deque(maxlen=self.replay_buffer_size)
        self.exploration_entropy_scale = float(exploration_entropy_scale)
        self.exploration_entropy_shift = float(exploration_entropy_shift)
        self.gradient_score_scale = float(gradient_score_scale)
        self.memory_gate_decay = float(memory_gate_decay)
        self.memory_gate_strength = float(memory_gate_strength)
        self.memory_gates = {}
        self.episodic_memory_size = int(episodic_memory_size)
        self.episodic_memory_threshold = float(episodic_memory_threshold)
        self.episodic_memory_prob = float(episodic_memory_prob)
        self.episodic_memory = deque(maxlen=self.episodic_memory_size)
        self.curiosity_strength = float(curiosity_strength)
        self.depth_clip_scaling = float(depth_clip_scaling)
        self.forgetting_rate = float(forgetting_rate)
        self.structural_dropout_prob = float(structural_dropout_prob)
        self.gradient_path_score_scale = float(gradient_path_score_scale)
        self.use_gradient_path_scoring = bool(use_gradient_path_scoring)
        self.activity_gate_exponent = float(activity_gate_exponent)
        self.subpath_cache = {}
        self._subpath_order = deque()
        self._subpath_cache_size = int(subpath_cache_size)
        self.subpath_cache_ttl = float(subpath_cache_ttl)
        self.use_mixed_precision = bool(use_mixed_precision)
        self.gradient_accumulation_steps = int(max(1, gradient_accumulation_steps))
        self._accum_step = 0
        self._accum_updates = {}
        try:
            import n_plugin

            n_plugin.register(self)
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.RLock()

    def _compute_loss(self, target_value, output_value):
        """Return loss using either ``loss_module`` or ``loss_fn``."""
        if self.loss_module is not None:
            t = torch.tensor([output_value], dtype=torch.float32)
            tt = torch.tensor([target_value], dtype=torch.float32)
            return float(self.loss_module(t, tt))
        return self.loss_fn(target_value, output_value)

    def modulate_plasticity(self, context):
        """Adjust plasticity_threshold based on neuromodulatory context."""
        reward = context.get("reward", 0.0)
        stress = context.get("stress", 0.0)
        adjustment = reward - stress
        self.plasticity_threshold = max(0.5, self.plasticity_threshold - adjustment)
        self.last_context = context.copy()
        self.context_history.append(self.last_context.copy())

    def update_context(self, **kwargs):
        """Update the stored neuromodulatory context without modifying plasticity."""
        self.last_context.update(kwargs)
        self.context_history.append(self.last_context.copy())

    def get_context(self):
        """Return a copy of the most recently stored neuromodulatory context."""
        return self.last_context.copy()

    def get_context_embedding(self) -> np.ndarray:
        """Return a vector summarising the recent neuromodulatory history."""
        if not self.context_history:
            return np.zeros(3, dtype=float)
        weights = [
            self.context_embedding_decay**i
            for i in range(len(self.context_history) - 1, -1, -1)
        ]
        vec = np.zeros(3, dtype=float)
        total = 0.0
        for w, ctx in zip(weights, reversed(self.context_history)):
            vec[0] += w * float(ctx.get("arousal", 0.0))
            vec[1] += w * float(ctx.get("stress", 0.0))
            vec[2] += w * float(ctx.get("reward", 0.0))
            total += w
        vec /= max(total, 1e-6)
        return vec

    def active_forgetting(self) -> None:
        """Decay stored context history values to gradually forget."""
        if self.forgetting_rate >= 1.0 or not self.context_history:
            return
        size = int(self.context_history_size)
        new_hist = deque(maxlen=size)
        for ctx in self.context_history:
            decayed = {
                k: (v * self.forgetting_rate if isinstance(v, (int, float)) else v)
                for k, v in ctx.items()
            }
            if any(isinstance(v, (int, float)) and abs(v) > 1e-6 for v in decayed.values()):
                new_hist.append(decayed)
        self.context_history = new_hist

    def reset_neuron_values(self):
        for neuron in self.core.neurons:
            neuron.value = None

    def decay_fatigues(self) -> None:
        """Apply fatigue decay to all synapses."""
        if not self.synaptic_fatigue_enabled:
            return
        for syn in self.core.synapses:
            if hasattr(syn, "update_fatigue"):
                syn.update_fatigue(0.0, self.fatigue_decay)

    def decay_visit_counts(self, decay: float = 0.95) -> None:
        """Apply exponential decay to synapse visit counters."""
        for syn in self.core.synapses:
            syn.visit_count *= decay

    def add_to_replay(self, input_value: float, target_value: float, error: float) -> None:
        """Store an experience and its priority for replay."""
        if not self.use_experience_replay:
            return
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.popleft()
            self.replay_priorities.popleft()
        self.replay_buffer.append((float(input_value), float(target_value)))
        self.replay_priorities.append(abs(float(error)) + 1e-6)

    def sample_replay_indices(self, batch_size: int) -> list[int]:
        """Return indices sampled proportionally to stored priorities."""
        pri = np.array(self.replay_priorities, dtype=float)
        if pri.sum() == 0:
            probs = np.ones_like(pri) / len(pri)
        else:
            probs = pri ** self.replay_alpha
            probs = probs / probs.sum()
        idx = np.random.choice(len(pri), size=batch_size, p=probs)
        return list(map(int, idx))

    def compute_path_entropy(self) -> float:
        """Compute entropy of synapse visit distribution."""
        counts = np.array([s.visit_count for s in self.core.synapses], dtype=float)
        total = counts.sum()
        if total <= 0:
            return 0.0
        probs = counts / total
        entropy = -float(np.sum([p * math.log(max(p, 1e-12)) for p in probs]))
        return entropy

    def compute_path_gradient_score(self, path) -> float:
        """Return cumulative absolute gradient magnitude for ``path``."""
        score = 0.0
        for _, syn in path:
            if syn is None:
                continue
            score += abs(self._prev_gradients.get(syn, 0.0))
        return score

    def update_exploration_schedule(self) -> None:
        """Adapt dropout probability based on visit entropy."""
        ent = self.compute_path_entropy()
        norm = math.log(len(self.core.synapses) + 1e-12)
        if norm > 0:
            ent /= norm
        val = self.exploration_entropy_scale * ent + self.exploration_entropy_shift
        self.dropout_probability = float(max(0.0, min(1.0, val)))

    def decay_memory_gates(self) -> None:
        """Decay memory gate strengths over time."""
        for syn in list(self.memory_gates.keys()):
            self.memory_gates[syn] *= self.memory_gate_decay
            if self.memory_gates[syn] < 1e-6:
                del self.memory_gates[syn]

    def _update_traces(self, path, decay: float = 0.9) -> None:
        """Update eligibility traces for the given path."""
        for syn in list(self._eligibility_traces.keys()):
            self._eligibility_traces[syn] *= decay
            if self._eligibility_traces[syn] < 1e-6:
                del self._eligibility_traces[syn]
        for syn in path:
            self._eligibility_traces[syn] = self._eligibility_traces.get(syn, 0.0) + 1.0

    def adjust_learning_rate(self) -> None:
        """Adapt learning rate based on recent error trends."""
        if len(self.error_history) < 4:
            return
        half = len(self.error_history) // 2
        hist = list(self.error_history)
        recent = np.mean(hist[-half:])
        previous = np.mean(hist[:-half]) if half > 0 else recent
        if previous <= 0:
            return
        ratio = recent / previous
        if ratio > 1.05:
            self.learning_rate = min(
                self.learning_rate * (1 + self.lr_adjustment_factor),
                self.max_learning_rate,
            )
        elif ratio < 0.95:
            self.learning_rate = max(
                self.learning_rate * (1 - self.lr_adjustment_factor),
                self.min_learning_rate,
            )

    def step_lr_scheduler(self) -> None:
        """Update ``learning_rate`` according to the configured scheduler."""
        if self.lr_scheduler == "none":
            return
        if self.lr_scheduler == "cosine":
            progress = min(1.0, self._scheduler_step / max(1, self.scheduler_steps))
            cos_out = (1 + math.cos(math.pi * progress)) / 2
            self.learning_rate = (
                self.min_learning_rate
                + (self.max_learning_rate - self.min_learning_rate) * cos_out
            )
        elif self.lr_scheduler == "exponential":
            self.learning_rate = max(
                self.min_learning_rate,
                self.max_learning_rate
                * (self.scheduler_gamma ** (self._scheduler_step + 1)),
            )
        elif self.lr_scheduler == "cyclic":
            steps = max(1, self.scheduler_steps)
            cycle_pos = (self._scheduler_step % steps) / steps
            if cycle_pos <= 0.5:
                scale = cycle_pos * 2
            else:
                scale = 2 - 2 * cycle_pos
            self.learning_rate = (
                self.min_learning_rate
                + (self.max_learning_rate - self.min_learning_rate) * scale
            )
        self._scheduler_step += 1

    def step_epsilon_scheduler(self) -> None:
        """Update ``rl_epsilon`` according to the configured scheduler."""
        if self.epsilon_scheduler == "none":
            return
        if self.epsilon_scheduler == "cosine":
            progress = min(
                1.0, self._epsilon_step / max(1, self.epsilon_scheduler_steps)
            )
            cos_out = (1 + math.cos(math.pi * progress)) / 2
            self.rl_epsilon = (
                self.rl_min_epsilon
                + (self.initial_epsilon - self.rl_min_epsilon) * cos_out
            )
        elif self.epsilon_scheduler == "exponential":
            self.rl_epsilon = max(
                self.rl_min_epsilon,
                self.initial_epsilon
                * (self.epsilon_scheduler_gamma ** (self._epsilon_step + 1)),
            )
        elif self.epsilon_scheduler == "linear":
            progress = min(
                1.0, self._epsilon_step / max(1, self.epsilon_scheduler_steps)
            )
            self.rl_epsilon = max(
                self.rl_min_epsilon,
                self.initial_epsilon * (1 - progress),
            )
        elif self.epsilon_scheduler == "cyclic":
            steps = max(1, self.epsilon_scheduler_steps)
            cycle_pos = (self._epsilon_step % steps) / steps
            if cycle_pos <= 0.5:
                scale = 1 - 2 * cycle_pos
            else:
                scale = 2 * (cycle_pos - 0.5)
            self.rl_epsilon = (
                self.rl_min_epsilon
                + (self.initial_epsilon - self.rl_min_epsilon) * scale
            )
        self._epsilon_step += 1

    def clip_gradient(self, value: float) -> float:
        """Return ``value`` clipped using ``core.gradient_clip_value``."""
        clip = getattr(self.core, "gradient_clip_value", None)
        if clip is None:
            return float(value)
        return float(np.clip(value, -clip, clip))

    def adjust_dropout_rate(self, avg_error: float) -> None:
        """Dynamically adapt dropout probability based on training error."""
        self.dropout_probability *= self.dropout_decay_rate
        self.dropout_probability += 0.1 * (avg_error - 0.5)
        self.dropout_probability = float(max(0.0, min(1.0, self.dropout_probability)))

    def _cache_subpaths(self, path, value):
        """Store all prefixes of ``path`` in ``subpath_cache``."""
        ids = [n.id for n, _ in path]
        now = datetime.now(timezone.utc)
        for i in range(1, len(ids) + 1):
            key = tuple(ids[:i])
            if key in self._subpath_order:
                self._subpath_order.remove(key)
            self._subpath_order.append(key)
            self.subpath_cache[key] = (value, path, now)
        while len(self._subpath_order) > self._subpath_cache_size:
            old = self._subpath_order.popleft()
            self.subpath_cache.pop(old, None)

    def _get_cached_subpath(self, path):
        """Return cached result for ``path`` if still valid."""
        key = tuple(n.id for n, _ in path)
        if key not in self.subpath_cache:
            return None
        value, cached_path, ts = self.subpath_cache[key]
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > self.subpath_cache_ttl:
            self.subpath_cache.pop(key, None)
            if key in self._subpath_order:
                self._subpath_order.remove(key)
            return None
        return [(self.core.neurons[n.id], s) for n, s in cached_path], value

    def check_finite_state(self) -> None:
        """Raise ``ValueError`` if any neuron value or synapse weight is NaN/Inf."""
        for n in self.core.neurons:
            if n.value is not None and not np.isfinite(n.value):
                raise ValueError("NaN or Inf encountered in neuron value")
        for s in self.core.synapses:
            if not np.isfinite(s.weight):
                raise ValueError("NaN or Inf encountered in synapse weight")

    # Reinforcement learning utilities
    def enable_rl(self) -> None:
        """Enable built-in reinforcement learning."""
        self.rl_enabled = True

    def disable_rl(self) -> None:
        """Disable built-in reinforcement learning."""
        self.rl_enabled = False

    def rl_select_action(self, state: tuple[int, int], n_actions: int) -> int:
        """Return an action using epsilon-greedy selection."""
        if not self.rl_enabled:
            raise RuntimeError("reinforcement learning disabled")
        if random.random() < self.rl_epsilon:
            return random.randrange(n_actions)
        q_vals = [
            self.dynamic_wander(self.q_encoding(state, a))[0] for a in range(n_actions)
        ]
        return int(np.argmax(q_vals))

    def rl_update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
        n_actions: int = 4,
    ) -> None:
        """Perform a Q-learning update using ``dynamic_wander``."""
        if not self.rl_enabled:
            return
        next_q = 0.0
        if not done:
            next_q = max(
                self.dynamic_wander(self.q_encoding(next_state, a))[0]
                for a in range(n_actions)
            )
        target = reward + self.rl_discount * next_q
        self.train([(self.q_encoding(state, action), target)], epochs=1)
        self.rl_epsilon = max(
            self.rl_min_epsilon, self.rl_epsilon * self.rl_epsilon_decay
        )

    def weighted_choice(self, synapses):
        """Select a synapse using fatigue- and attention-aware softmax."""
        if len(synapses) == 1:
            return synapses[0]

        scores = []
        ctx_vec = self.get_context_embedding()
        for syn in synapses:
            fatigue = (
                getattr(syn, "fatigue", 0.0) if self.synaptic_fatigue_enabled else 0.0
            )
            fatigue_factor = max(0.0, 1.0 - fatigue)
            attention = 1.0 + self.core.neurons[syn.target].attention_score
            novelty_penalty = 1.0 / (1.0 + getattr(syn, "visit_count", 0))
            curiosity_factor = 1.0 + self.curiosity_strength * novelty_penalty
            tgt_rep = self.core.neurons[syn.target].representation[: len(ctx_vec)]
            sim = _cosine_similarity(tgt_rep, ctx_vec)
            context_factor = 1.0 + sim
            mem_factor = 1.0 + self.memory_gates.get(syn, 0.0)
            scores.append(
                syn.potential
                * fatigue_factor
                * attention
                * novelty_penalty
                * curiosity_factor
                * context_factor
                * mem_factor
            )

        scores_arr = np.array(scores, dtype=float)
        if np.all(scores_arr == 0.0):
            return random.choice(synapses)
        max_score = np.max(scores_arr)
        exp_scores = np.exp(scores_arr - max_score)
        sum_exp = np.sum(exp_scores)
        if sum_exp == 0 or np.isnan(sum_exp):
            return random.choice(synapses)
        probs = exp_scores / sum_exp
        idx = np.random.choice(len(synapses), p=probs)
        return synapses[idx]

    def _wander(self, current_neuron, path, current_continue_prob, depth_remaining):
        cached = self._get_cached_subpath(path)
        if cached is not None:
            cached_path, val = cached
            neuron = cached_path[-1][0]
            neuron.value = val
            # Apply side effects that would occur during normal traversal
            for _, syn in cached_path[1:]:
                if syn is None:
                    continue
                if self.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                    syn.update_fatigue(self.fatigue_increase, self.fatigue_decay)
                syn.visit_count += 1
            return [(neuron, cached_path)]

        results = []
        synapses = current_neuron.synapses
        if self.dropout_probability > 0.0:
            synapses = [
                s for s in synapses if random.random() > self.dropout_probability
            ]
        if (
            depth_remaining <= 0
            or not synapses
            or random.random() > current_continue_prob
        ):
            results.append((current_neuron, path))
            self._cache_subpaths(path, current_neuron.value)
            return results
        if len(synapses) > 1 and random.random() < self.split_probability:
            for syn in synapses:
                next_neuron = self.core.neurons[syn.target]
                w = (
                    syn.effective_weight(self.last_context, self.global_phase)
                    if hasattr(syn, "effective_weight")
                    else syn.weight
                )
                transmitted_value = self.combine_fn(current_neuron.value, w)
                if hasattr(syn, "apply_side_effects"):
                    syn.apply_side_effects(self.core, current_neuron.value)
                if hasattr(syn, "update_echo"):
                    syn.update_echo(current_neuron.value, self.core.synapse_echo_decay)
                if self.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                    syn.update_fatigue(self.fatigue_increase, self.fatigue_decay)
                inc = self.route_potential_increase
                if syn.potential < 1.0:
                    inc += self.exploration_bonus
                syn.potential = min(self.synapse_potential_cap, syn.potential + inc)
                syn.visit_count += 1
                if hasattr(next_neuron, "process"):
                    next_neuron.value = next_neuron.process(transmitted_value)
                else:
                    next_neuron.value = transmitted_value
                new_path = path + [(next_neuron, syn)]
                new_continue_prob = current_continue_prob * self.continue_decay_rate
                if next_neuron.tier == "remote" and self.remote_client is not None:
                    remote_out = self.remote_client.process(
                        transmitted_value, timeout=self.remote_timeout
                    )
                    next_neuron.value = remote_out
                    results.append((next_neuron, new_path))
                elif next_neuron.tier == "torrent" and self.torrent_client is not None:
                    part = self.torrent_map.get(next_neuron.id)
                    remote_out = self.torrent_client.process(transmitted_value, part)
                    next_neuron.value = remote_out
                    results.append((next_neuron, new_path))
                else:
                    results.extend(
                        self._wander(
                            next_neuron,
                            new_path,
                            new_continue_prob,
                            depth_remaining - 1,
                        )
                    )
        else:
            syn = self.weighted_choice(synapses)
            next_neuron = self.core.neurons[syn.target]
            w = (
                syn.effective_weight(self.last_context, self.global_phase)
                if hasattr(syn, "effective_weight")
                else syn.weight
            )
            transmitted_value = self.combine_fn(current_neuron.value, w)
            if hasattr(syn, "apply_side_effects"):
                syn.apply_side_effects(self.core, current_neuron.value)
            if hasattr(syn, "update_echo"):
                syn.update_echo(current_neuron.value, self.core.synapse_echo_decay)
            if self.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                syn.update_fatigue(self.fatigue_increase, self.fatigue_decay)
            inc = self.route_potential_increase
            if syn.potential < 1.0:
                inc += self.exploration_bonus
            syn.potential = min(self.synapse_potential_cap, syn.potential + inc)
            syn.visit_count += 1
            if hasattr(next_neuron, "process"):
                next_neuron.value = next_neuron.process(transmitted_value)
            else:
                next_neuron.value = transmitted_value
            new_path = path + [(next_neuron, syn)]
            new_continue_prob = current_continue_prob * self.continue_decay_rate
            if next_neuron.tier == "remote" and self.remote_client is not None:
                remote_out = self.remote_client.process(
                    transmitted_value, timeout=self.remote_timeout
                )
                next_neuron.value = remote_out
                results.append((next_neuron, new_path))
            elif next_neuron.tier == "torrent" and self.torrent_client is not None:
                part = self.torrent_map.get(next_neuron.id)
                remote_out = self.torrent_client.process(transmitted_value, part)
                next_neuron.value = remote_out
                results.append((next_neuron, new_path))
            else:
                results.extend(
                    self._wander(
                        next_neuron,
                        new_path,
                        new_continue_prob,
                        depth_remaining - 1,
                    )
                )
        for n, p in results:
            self._cache_subpaths(p, n.value)
        return results

    def _merge_results(self, results):
        groups = {}
        for neuron, path in results:
            groups.setdefault(neuron.id, []).append((neuron, path))
        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                avg_value = sum(n.value for n, _ in group) / len(group)
                if self.use_gradient_path_scoring:
                    rep_path = max(
                        group, key=lambda tup: self.compute_path_gradient_score(tup[1])
                    )[1]
                else:
                    rep_path = max(group, key=lambda tup: len(tup[1]))[1]
                neuron = self.core.neurons[key]
                neuron.value = avg_value
                merged.append((neuron, rep_path))
        def _score(tup):
            neuron, path = tup
            base = neuron.value
            if self.use_gradient_path_scoring:
                base += self.gradient_path_score_scale * self.compute_path_gradient_score(path)
            return base

        final_neuron, final_path = max(merged, key=_score)
        if final_path and hasattr(final_path[0][0], "created_at"):
            pathway_start_time = final_path[0][0].created_at
            pathway_age = (datetime.now() - pathway_start_time).total_seconds()
            print(f"Partial pathway age: {pathway_age:.2f} sec")
        return final_neuron, final_path

    def _select_entry_neuron(self):
        """Return an entry neuron biased by ``attention_score`` if enabled."""
        candidates = [n for n in self.core.neurons if n.synapses]
        if not candidates:
            return random.choice(self.core.neurons)
        if self.dynamic_attention_enabled:
            scores = np.array([abs(n.attention_score) + 1e-3 for n in candidates])
            probs = scores / np.sum(scores)
            idx = np.random.choice(len(candidates), p=probs)
            return candidates[idx]
        return random.choice(candidates)

    def _beam_wander(self, start_neuron, depth_limit):
        beams = [(start_neuron, [(start_neuron, None)], 0.0)]
        for _ in range(depth_limit):
            candidates = []
            for neuron, path, score in beams:
                if not neuron.synapses:
                    candidates.append((neuron, path, score))
                    continue
                for syn in neuron.synapses:
                    next_neuron = self.core.neurons[syn.target]
                    w = (
                        syn.effective_weight(self.last_context, self.global_phase)
                        if hasattr(syn, "effective_weight")
                        else syn.weight
                    )
                    val = self.combine_fn(neuron.value, w)
                    if hasattr(syn, "apply_side_effects"):
                        syn.apply_side_effects(self.core, neuron.value)
                    if hasattr(next_neuron, "process"):
                        next_neuron.value = next_neuron.process(val)
                    else:
                        next_neuron.value = val
                    new_path = path + [(next_neuron, syn)]
                    fatigue_factor = 1.0
                if self.synaptic_fatigue_enabled:
                    fatigue_factor -= getattr(syn, "fatigue", 0.0)
                    fatigue_factor = max(0.0, fatigue_factor)
                novelty_penalty = 1.0 / (1.0 + getattr(syn, "visit_count", 0))
                new_score = score + syn.potential * fatigue_factor * novelty_penalty
                syn.visit_count += 1
                candidates.append((next_neuron, new_path, new_score))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[: self.beam_width]
        best = max(beams, key=lambda x: x[2])
        return best[0], best[1]

    def dynamic_wander(self, input_value, apply_plasticity=True):
        with self.lock:
            self.global_phase = (self.global_phase + self.phase_rate) % (2 * math.pi)
            if not apply_plasticity and input_value in self.wander_cache:
                out, path, ts = self.wander_cache[input_value]
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age <= self.wander_cache_ttl:
                    if input_value in self._cache_order:
                        self._cache_order.remove(input_value)
                    self._cache_order.append(input_value)
                    return out, list(path)
                else:
                    self.wander_cache.pop(input_value, None)
                    if input_value in self._cache_order:
                        self._cache_order.remove(input_value)

            for neuron in self.core.neurons:
                neuron.value = None
            self.active_forgetting()
            self.decay_memory_gates()
            self.decay_fatigues()
            self.decay_visit_counts()
            entry_neuron = self._select_entry_neuron()
            entry_neuron.value = input_value
            initial_path = [(entry_neuron, None)]
            depth_limit = int(
                max(
                    1,
                    round(
                        self.max_wander_depth
                        + np.random.normal(0.0, self.wander_depth_noise)
                    ),
                )
            )
            if self.beam_width > 1:
                final_neuron, final_path = self._beam_wander(entry_neuron, depth_limit)
            else:
                results = self._wander(entry_neuron, initial_path, 1.0, depth_limit)
                final_neuron, final_path = self._merge_results(results)
            if not final_path or all(s is None for _, s in final_path):
                if entry_neuron.synapses and self.dropout_probability < 1.0:
                    syn = self.weighted_choice(entry_neuron.synapses)
                    next_neuron = self.core.neurons[syn.target]
                    w = (
                        syn.effective_weight(self.last_context, self.global_phase)
                        if hasattr(syn, "effective_weight")
                        else syn.weight
                    )
                    raw_val = self.combine_fn(entry_neuron.value, w)
                    if hasattr(next_neuron, "process"):
                        next_neuron.value = next_neuron.process(raw_val)
                    else:
                        next_neuron.value = raw_val
                    if hasattr(syn, "apply_side_effects"):
                        syn.apply_side_effects(self.core, entry_neuron.value)
                    if hasattr(syn, "update_echo"):
                        syn.update_echo(
                            entry_neuron.value, self.core.synapse_echo_decay
                        )
                    if self.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                        syn.update_fatigue(self.fatigue_increase, self.fatigue_decay)
                    syn.visit_count += 1
                    final_path = [(entry_neuron, None), (next_neuron, syn)]
                else:
                    final_path = initial_path
            self.global_activation_count += 1
            if self.global_activation_count % self.route_visit_decay_interval == 0:
                for syn in self.core.synapses:
                    syn.potential *= self.route_potential_decay
            if self.global_activation_count % self.synapse_prune_interval == 0:
                self.prune_low_potential_synapses()
            if apply_plasticity:
                self.apply_structural_plasticity(final_path)
                self._record_path_usage([s for (_, s) in final_path if s is not None])
                self.maybe_create_emergent_synapse()
                self._update_concept_pairs(
                    [s for (_, s) in final_path if s is not None]
                )
                self.apply_concept_associations()
            result_path = [s for (_, s) in final_path if s is not None]
            self._cache_subpaths(final_path, final_neuron.value)
            if not apply_plasticity:
                now = datetime.now(timezone.utc)
                expired = [
                    key
                    for key, (_, _, ts) in self.wander_cache.items()
                    if (now - ts).total_seconds() > self.wander_cache_ttl
                ]
                for key in expired:
                    self.wander_cache.pop(key, None)
                    if key in self._cache_order:
                        self._cache_order.remove(key)
                if len(self._cache_order) >= self._cache_max_size:
                    old = self._cache_order.popleft()
                    self.wander_cache.pop(old, None)
                self.wander_cache[input_value] = (
                    final_neuron.value,
                    result_path,
                    datetime.now(timezone.utc),
                )
                self._cache_order.append(input_value)
            self.check_finite_state()
            return final_neuron.value, result_path

    def dynamic_wander_parallel(self, input_value, num_processes=None):
        """Run ``dynamic_wander`` in multiple processes.

        Each process receives a temporary copy of ``self`` and performs its own
        wandering starting from a different random seed.  The seeds used are
        returned so the main process can replay each path and apply weight
        updates.  This avoids modifying stale copies of the core while still
        allowing the expensive wandering phase to execute in parallel.
        """

        num = num_processes if num_processes is not None else self.parallel_wanderers
        num = int(max(1, round(num)))
        if num <= 1:
            output, _ = self.dynamic_wander(input_value)
            seed = random.randint(0, 2**32 - 1)
            return [(output, seed)]

        state = pickle.dumps(self)
        seeds = [random.randint(0, 2**32 - 1) for _ in range(num)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num) as pool:
            outputs = pool.starmap(
                _wander_worker, [(state, input_value, s) for s in seeds]
            )

        return outputs

    def chaotic_memory_replay(
        self,
        input_value: float,
        chaos_param: float = 3.7,
        iterations: int = 5,
    ) -> tuple[float, list]:
        """Generate a replay sequence using chaotic perturbations.

        A logistic map drives noise injections between successive
        ``dynamic_wander`` calls.  The resulting paths are concatenated
        and returned alongside the final output value.
        """

        output = float(input_value)
        path = []
        state = self.chaos_state
        for _ in range(int(iterations)):
            state = chaos_param * state * (1.0 - state)
            noise = state - 0.5
            out, p = self.dynamic_wander(output + noise, apply_plasticity=False)
            output = float(out)
            path.extend(p)
        self.chaos_state = state
        return output, path

    def update_chaotic_gate(self) -> float:
        """Advance and return the logistic gate value used for update scaling."""
        self.chaotic_gate = (
            self.chaotic_gating_param * self.chaotic_gate * (1.0 - self.chaotic_gate)
        )
        return self.chaotic_gate

    def get_current_gate(self) -> float:
        """Return the current chaotic gate value."""
        return float(self.chaotic_gate)

    def apply_structural_plasticity(self, path):
        ctx = self.last_context
        mod = 1.0 + ctx.get("reward", 0.0) - ctx.get("stress", 0.0)
        for _, syn in path:
            if (
                syn is not None
                and syn.potential >= self.plasticity_threshold
                and random.random() > self.structural_dropout_prob
            ):
                source = self.core.neurons[syn.source]
                target = self.core.neurons[syn.target]
                if source.tier == "vram":
                    new_tier = "ram"
                elif source.tier == "ram":
                    new_tier = "disk"
                else:
                    new_tier = "disk"
                new_id = len(self.core.neurons)
                new_neuron = Neuron(new_id, value=target.value, tier=new_tier)
                rep_sim = _cosine_similarity(
                    source.representation, target.representation
                )
                new_neuron.representation = (
                    source.representation + target.representation
                ) / 2.0
                self.core.neurons.append(new_neuron)
                new_weight1 = (
                    syn.weight
                    * self.struct_weight_multiplier1
                    * mod
                    * self.structural_learning_rate
                    * (1.0 + rep_sim)
                )
                if new_weight1 > self._weight_limit:
                    new_weight1 = self._weight_limit
                elif new_weight1 < -self._weight_limit:
                    new_weight1 = -self._weight_limit
                new_syn1 = self.core.add_synapse(
                    source.id,
                    new_id,
                    weight=new_weight1,
                    synapse_type=random.choice(SYNAPSE_TYPES),
                )
                new_weight2 = (
                    syn.weight
                    * self.struct_weight_multiplier2
                    * mod
                    * self.structural_learning_rate
                    * (1.0 + rep_sim)
                )
                if new_weight2 > self._weight_limit:
                    new_weight2 = self._weight_limit
                elif new_weight2 < -self._weight_limit:
                    new_weight2 = -self._weight_limit
                new_syn2 = self.core.add_synapse(
                    new_id,
                    target.id,
                    weight=new_weight2,
                    synapse_type=random.choice(SYNAPSE_TYPES),
                )
                source.synapses = [s for s in source.synapses if s != syn]
                self.core.synapses = [s for s in self.core.synapses if s != syn]
                print(
                    f"Structural plasticity: Replaced synapse from {source.id} (tier {source.tier}) to {target.id} with new neuron {new_id} in tier {new_tier}."
                )

    def update_attention(self, path, error):
        path_len = len(path)
        for syn in path:
            n_type = self.core.neurons[syn.target].neuron_type
            score = abs(error) / max(path_len, 1)
            self.type_attention[n_type] += score
            speed_score = 1.0 / max(path_len, 1)
            self.type_speed_attention[n_type] += speed_score

    def get_preferred_neuron_type(self):
        if not any(self.type_attention.values()):
            return "standard"
        best = max(self.type_attention.items(), key=lambda x: x[1])[0]
        for k in self.type_attention:
            self.type_attention[k] *= self.attention_decay
        return best

    def get_combined_preferred_neuron_type(self):
        combined = {
            nt: self.type_attention[nt] + self.type_speed_attention[nt]
            for nt in NEURON_TYPES
        }
        if not any(combined.values()):
            return "standard"
        best = max(combined.items(), key=lambda x: x[1])[0]
        for d in (self.type_attention, self.type_speed_attention):
            for k in d:
                d[k] *= self.attention_decay
        return best

    def apply_weight_updates_and_attention(self, path, error):
        # Decay stored momentum values to prevent unbounded growth over time
        for syn in list(self._momentum.keys()):
            self._momentum[syn] *= 0.9
            if abs(self._momentum[syn]) < 1e-8:
                del self._momentum[syn]

        self._update_traces(path)
        path_length = len(path)
        for syn in path:
            if getattr(syn, "frozen", False):
                continue
            source_value = self.core.neurons[syn.source].value
            delta = self.weight_update_fn(source_value, error, path_length)
            prev_delta = self._prev_gradients.get(syn)
            if prev_delta is not None and prev_delta * delta < 0:
                delta *= 0.5
            self._prev_gradients[syn] = delta
            delta *= self._eligibility_traces.get(syn, 1.0)
            if self.use_echo_modulation and hasattr(syn, "get_echo_average"):
                delta *= syn.get_echo_average()
            if self.gradient_noise_std > 0:
                delta += np.random.normal(0.0, self.gradient_noise_std)
            delta = self.clip_gradient(delta)
            prev_v = self._grad_sq.get(syn, 1.0)
            v = self._rmsprop_beta * prev_v + (1 - self._rmsprop_beta) * (delta**2)
            self._grad_sq[syn] = v
            scaled_delta = delta / math.sqrt(v + self._grad_epsilon)
            mom_prev = self._momentum.get(syn, 0.0)
            mom = self.momentum_coefficient * mom_prev + scaled_delta
            self._momentum[syn] = mom
            update = self.learning_rate * (
                self.momentum_coefficient * mom + scaled_delta
            )
            phase_factor = math.cos(syn.phase - self.global_phase)
            update *= phase_factor
            if self.chaotic_gating_enabled:
                update *= self.update_chaotic_gate()
            syn.phase = (syn.phase + self.phase_adaptation_rate * error) % (2 * math.pi)
            if self.synaptic_fatigue_enabled:
                fatigue_factor = 1.0 - getattr(syn, "fatigue", 0.0)
                update *= max(0.0, fatigue_factor)
            activity_factor = 1.0 / (1.0 + syn.visit_count ** self.activity_gate_exponent)
            update *= activity_factor
            depth_factor = 1.0 + self.depth_clip_scaling * (
                path_length / max(1, self.max_wander_depth)
            )
            cap = self.synapse_update_cap / depth_factor
            if abs(update) > cap:
                update = math.copysign(cap, update)
            self._accum_updates[syn] = self._accum_updates.get(syn, 0.0) + update
            syn.potential = min(
                self.synapse_potential_cap,
                syn.potential + abs(scaled_delta) * self.gradient_score_scale,
            )
            if random.random() < self.consolidation_probability:
                syn.weight *= self.consolidation_strength
            if syn.weight > self._weight_limit:
                syn.weight = self._weight_limit
            elif syn.weight < -self._weight_limit:
                syn.weight = -self._weight_limit
            score = abs(error) * abs(syn.weight) / max(path_length, 1)
            self.core.neurons[syn.target].attention_score += score
        self._accum_step += 1
        if self._accum_step >= self.gradient_accumulation_steps:
            for syn, upd in self._accum_updates.items():
                syn.weight += upd
                if syn.weight > self._weight_limit:
                    syn.weight = self._weight_limit
                elif syn.weight < -self._weight_limit:
                    syn.weight = -self._weight_limit
            if self.weight_decay:
                for syn in self.core.synapses:
                    if getattr(syn, "frozen", False):
                        continue
                    syn.weight *= 1.0 - self.weight_decay
            self._accum_updates.clear()
            self._accum_step = 0
        if path:
            last_neuron = self.core.neurons[path[-1].target]
            last_neuron.attention_score += abs(error)
            self.update_attention(path, error)
            self.update_synapse_type_attentions(
                path,
                error,
                len(path),
                len(self.core.synapses),
            )
            if abs(error) < self.episodic_memory_threshold:
                mem_path = []
                for syn in path:
                    self.memory_gates[syn] = self.memory_gates.get(syn, 0.0) + self.memory_gate_strength
                    mem_path.append(syn)
                self.episodic_memory.append(mem_path)
        if self.weight_decay:
            for syn in self.core.synapses:
                if getattr(syn, "frozen", False):
                    continue
                syn.weight *= 1.0 - self.weight_decay
        return path_length

    def train_example(self, input_value, target_value):
        with self.lock:
            if self.parallel_wanderers > 1:
                results = self.dynamic_wander_parallel(
                    input_value, num_processes=self.parallel_wanderers
                )
                metrics = []
                for _, seed in results:
                    random.seed(seed)
                    np.random.seed(seed % (2**32 - 1))
                    out_val, path = self.dynamic_wander(
                        input_value, apply_plasticity=False
                    )
                    err = self._compute_loss(target_value, out_val)
                    pred_size = len(self.core.synapses) + sum(
                        1 for syn in path if syn.potential >= self.plasticity_threshold
                    )
                    metrics.append(
                        (
                            abs(err),
                            len(path),
                            pred_size,
                            seed,
                        )
                    )
                best = min(metrics, key=lambda m: (m[0], m[1], m[2]))
                best_seed = best[3]
                random.seed(best_seed)
                np.random.seed(best_seed % (2**32 - 1))
                if self.use_mixed_precision and torch.cuda.is_available():
                    device = "cuda"
                    with torch.autocast(device_type=device):
                        output_value, path = self.dynamic_wander(input_value)
                        error = self._compute_loss(target_value, output_value)
                else:
                    output_value, path = self.dynamic_wander(input_value)
                    error = self._compute_loss(target_value, output_value)
                path_length = self.apply_weight_updates_and_attention(path, error)
            else:
                if self.use_mixed_precision and torch.cuda.is_available():
                    device = "cuda"
                    with torch.autocast(device_type=device):
                        output_value, path = self.dynamic_wander(input_value)
                        error = self._compute_loss(target_value, output_value)
                else:
                    output_value, path = self.dynamic_wander(input_value)
                    error = self._compute_loss(target_value, output_value)
                path_length = self.apply_weight_updates_and_attention(path, error)
            self.add_to_replay(input_value, target_value, error)
            self.error_history.append(abs(error))
            self.training_history.append(
                {
                    "input": input_value,
                    "target": target_value,
                    "output": output_value,
                    "error": error,
                    "path_length": path_length,
                }
            )
            return output_value, error, path

    def train(self, examples, epochs=1):
        for epoch in range(epochs):
            epoch_errors = []
            for input_val, target_val in examples:
                output, error, _ = self.train_example(input_val, target_val)
                epoch_errors.append(
                    abs(error) if isinstance(error, (int, float)) else 0
                )
            avg_error = sum(epoch_errors) / len(epoch_errors) if epoch_errors else 0
            print(f"Epoch {epoch+1}/{epochs} - Average error: {avg_error:.4f}")
            if avg_error > 0.1:
                self.core.expand(
                    num_new_neurons=10,
                    num_new_synapses=15,
                    alternative_connection_prob=self.alternative_connection_prob,
                )
            self.core.synapses = [
                s for s in self.core.synapses if abs(s.weight) >= 0.05
            ]
            change = perform_message_passing(
                self.core,
                metrics_visualizer=self.metrics_visualizer,
                attention_module=self.core.attention_module,
            )
            self.last_message_passing_change = change
            self.decide_synapse_action()
            self.adjust_learning_rate()
            self.step_lr_scheduler()
            self.step_epsilon_scheduler()
            if (
                self.use_experience_replay
                and len(self.replay_buffer) >= self.replay_batch_size
            ):
                idxs = self.sample_replay_indices(self.replay_batch_size)
                for i in idxs:
                    inp, tgt = self.replay_buffer[i]
                    _, err, _ = self.train_example(inp, tgt)
                    self.replay_priorities[i] = abs(err) + 1e-6
            self.update_exploration_schedule()
            self.adjust_dropout_rate(avg_error)

    def get_training_history(self):
        return self.training_history

    def contrastive_train(
        self,
        inputs: list,
        epochs: int = 1,
        batch_size: int = 4,
        temperature: float = 0.5,
        augment_fn=None,
    ) -> float:
        """Run self-supervised contrastive learning on the given inputs."""
        learner = ContrastiveLearner(self.core, self, temperature, augment_fn)
        last_loss = 0.0
        for _ in range(int(epochs)):
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i : i + batch_size]
                last_loss = learner.train(batch)
        return last_loss

    def imitation_train(
        self,
        demonstrations: list[tuple[float, float]],
        epochs: int = 1,
        max_history: int = 100,
    ) -> float:
        """Train using behaviour cloning on demonstration pairs."""
        learner = ImitationLearner(self.core, self, max_history=max_history)
        for inp, act in demonstrations:
            learner.record(float(inp), float(act))
        last_loss = 0.0
        for _ in range(int(epochs)):
            for inp, act in demonstrations:
                last_loss = learner.train_step(float(inp), float(act))
        return last_loss

    def update_synapse_type_attentions(self, path, loss, speed, size):
        for syn in path:
            st = syn.synapse_type
            self.synapse_loss_attention[st] += abs(loss)
            self.synapse_speed_attention[st] += speed
            self.synapse_size_attention[st] += size

    def decide_synapse_action(self):
        if not self.core.synapses:
            return
        create_type = max(
            self.synapse_loss_attention, key=self.synapse_loss_attention.get
        )
        remove_type = max(
            self.synapse_size_attention, key=self.synapse_size_attention.get
        )
        if random.random() < 0.5:
            src = random.choice(self.core.neurons).id
            tgt = random.choice(self.core.neurons).id
            if src != tgt:
                self.core.add_synapse(
                    src,
                    tgt,
                    weight=self.core._init_weight(),
                    synapse_type=create_type,
                )
        else:
            for syn in list(self.core.synapses):
                if syn.synapse_type == remove_type:
                    self.core.neurons[syn.source].synapses.remove(syn)
                    self.core.synapses.remove(syn)
                    break

    def prune_low_potential_synapses(self, threshold=0.05):
        """Remove synapses with low potential or very small weights."""
        to_keep = []
        for syn in self.core.synapses:
            if abs(syn.weight) < threshold or syn.potential < threshold:
                self.core.neurons[syn.source].synapses.remove(syn)
                if syn in self._momentum:
                    del self._momentum[syn]
            else:
                to_keep.append(syn)
        self.core.synapses = to_keep

    def _record_path_usage(self, path):
        """Increment usage counter for ``path`` and create shortcuts if needed."""
        if len(path) < 2 or self.shortcut_creation_threshold <= 0:
            return
        key = tuple((syn.source, syn.target) for syn in path)
        count = self._path_usage.get(key, 0) + 1
        if count >= self.shortcut_creation_threshold:
            self._create_shortcut_synapse(path)
            count = 0
        self._path_usage[key] = count

    def _create_shortcut_synapse(self, path):
        """Create a direct synapse from first to last neuron of ``path``."""
        src = path[0].source
        tgt = path[-1].target
        if src == tgt:
            return
        weight = float(np.mean([syn.weight for syn in path]))
        existing = [s for s in self.core.neurons[src].synapses if s.target == tgt]
        if existing:
            syn = existing[0]
            syn.weight = min(syn.weight + weight, self._weight_limit)
        else:
            self.core.add_synapse(
                src,
                tgt,
                weight=min(weight, self._weight_limit),
                synapse_type=random.choice(SYNAPSE_TYPES),
            )
        print(f"Shortcut created from {src} to {tgt}")

    def maybe_create_emergent_synapse(self):
        """Randomly create a new synapse to encourage emergent structure."""
        if random.random() >= self.emergent_connection_prob:
            return None
        if len(self.core.neurons) < 2:
            return None
        src, tgt = random.sample(range(len(self.core.neurons)), 2)
        if src == tgt:
            return None
        weight = self.core._init_weight()
        syn = self.core.add_synapse(
            src,
            tgt,
            weight=weight,
            synapse_type=random.choice(SYNAPSE_TYPES),
        )
        return syn

    def _update_concept_pairs(self, path):
        """Record consecutive neuron pairs for concept association."""
        if not path:
            return
        prev = path[0].source
        for syn in path:
            pair = (prev, syn.target)
            self._concept_pairs[pair] = self._concept_pairs.get(pair, 0) + 1
            prev = syn.target

    def apply_concept_associations(self):
        """Create concept neurons when pair counts exceed threshold."""
        to_reset = []
        for (src, tgt), count in list(self._concept_pairs.items()):
            if count < self.concept_association_threshold:
                continue
            rep_a = self.core.neurons[src].representation
            rep_b = self.core.neurons[tgt].representation
            new_rep = np.tanh((rep_a + rep_b) / 2.0)
            new_id = len(self.core.neurons)
            tier = self.core.choose_new_tier()
            neuron = Neuron(new_id, value=0.0, tier=tier, rep_size=self.core.rep_size)
            neuron.representation = new_rep.astype(np.float32)
            self.core.neurons.append(neuron)
            self.core.add_synapse(src, new_id, weight=self.concept_learning_rate)
            self.core.add_synapse(new_id, tgt, weight=self.concept_learning_rate)
            to_reset.append((src, tgt))
        for pair in to_reset:
            self._concept_pairs[pair] = 0

    def genetic_algorithm(
        self,
        data,
        population_size: int = 10,
        generations: int = 5,
        selection_ratio: float = 0.5,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.05,
        device: str | None = None,
    ) -> None:
        """Evolve synapse weights using a genetic algorithm.

        Parameters
        ----------
        data:
            Iterable of ``(input, target)`` pairs used to evaluate fitness.
        population_size:
            Number of network copies to evolve each generation.
        generations:
            Number of generations to run.
        selection_ratio:
            Fraction of the population selected as parents for the next
            generation.
        crossover_rate:
            Probability that a child's synapse weight is taken from the second
            parent instead of the first.
        mutation_rate:
            Probability of mutating each synapse weight after crossover.
        mutation_strength:
            Magnitude of random weight change applied during mutation.
        device:
            ``"cuda"`` or ``"cpu"``.  If ``None``, uses ``CUDA`` when available.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not data:
            return

        population = [pickle.loads(pickle.dumps(self)) for _ in range(population_size)]

        for _ in range(generations):
            scores: list[float] = []
            for nb in population:
                loss_sum = 0.0
                for inp, tgt in data:
                    out, _ = nb.dynamic_wander(inp, apply_plasticity=False)
                    t = torch.tensor([float(out)], device=device)
                    tt = torch.tensor([float(tgt)], device=device)
                    loss_sum += torch.nn.functional.mse_loss(t, tt).item()
                scores.append(loss_sum / len(data))

            ranked = [
                nb for _, nb in sorted(zip(scores, population), key=lambda x: x[0])
            ]
            num_selected = max(1, int(population_size * selection_ratio))
            parents = ranked[:num_selected]

            children = []
            while len(children) + len(parents) < population_size:
                p1, p2 = random.sample(parents, 2)
                child = pickle.loads(pickle.dumps(p1))
                for s1, s2 in zip(child.core.synapses, p2.core.synapses):
                    if random.random() < crossover_rate:
                        s1.weight = s2.weight
                children.append(child)

            population = parents + children

            for nb in population:
                for syn in nb.core.synapses:
                    if random.random() < mutation_rate:
                        syn.weight += random.uniform(
                            -mutation_strength, mutation_strength
                        )

        best_nb = min(
            population,
            key=lambda nb: sum(
                torch.nn.functional.mse_loss(
                    torch.tensor(
                        [float(nb.dynamic_wander(inp, apply_plasticity=False)[0])],
                        device=device,
                    ),
                    torch.tensor([float(tgt)], device=device),
                ).item()
                for inp, tgt in data
            ),
        )

        best_state = pickle.dumps(best_nb)
        updated = pickle.loads(best_state)
        updated.lock = threading.RLock()
        self.__dict__.update(updated.__dict__)
