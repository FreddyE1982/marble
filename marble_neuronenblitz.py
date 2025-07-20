from marble_imports import *
from marble_core import Neuron, SYNAPSE_TYPES, NEURON_TYPES, perform_message_passing
from contrastive_learning import ContrastiveLearner
from marble_base import MetricsVisualizer
import threading
import multiprocessing as mp
import pickle
from collections import deque
import math


def _wander_worker(state_bytes, input_value, seed):
    nb = pickle.loads(state_bytes)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    output, _ = nb.dynamic_wander(input_value)
    return output, seed


def default_combine_fn(x, w):
    return max(x * w, 0)


def default_loss_fn(target, output):
    return target - output


def default_weight_update_fn(source, error, path_len):
    return (error * source) / (path_len + 1)


def default_q_encoding(state: tuple[int, int], action: int) -> float:
    """Encode a state-action pair into a numeric value."""
    return float(state[0] * 10 + state[1] + action / 10)


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
        synaptic_fatigue_enabled=True,
        fatigue_increase=0.05,
        fatigue_decay=0.95,
        lr_adjustment_factor=0.1,
        momentum_coefficient=0.0,
        reinforcement_learning_enabled=False,
        rl_discount=0.9,
        rl_epsilon=1.0,
        rl_epsilon_decay=0.95,
        rl_min_epsilon=0.1,
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
        self.synaptic_fatigue_enabled = synaptic_fatigue_enabled
        self.fatigue_increase = fatigue_increase
        self.fatigue_decay = fatigue_decay
        self.lr_adjustment_factor = lr_adjustment_factor
        self.momentum_coefficient = momentum_coefficient
        self.rl_enabled = reinforcement_learning_enabled
        self.rl_discount = rl_discount
        self.rl_epsilon = rl_epsilon
        self.rl_epsilon_decay = rl_epsilon_decay
        self.rl_min_epsilon = rl_min_epsilon

        self.combine_fn = combine_fn if combine_fn is not None else default_combine_fn
        self.loss_fn = loss_fn if loss_fn is not None else default_loss_fn
        self.loss_module = loss_module
        self.weight_update_fn = (
            weight_update_fn if weight_update_fn is not None else default_weight_update_fn
        )

        self._weight_limit = 1e6

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
        self.q_encoding = default_q_encoding

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

    def update_context(self, **kwargs):
        """Update the stored neuromodulatory context without modifying plasticity."""
        self.last_context.update(kwargs)

    def get_context(self):
        """Return a copy of the most recently stored neuromodulatory context."""
        return self.last_context.copy()

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
            self.dynamic_wander(self.q_encoding(state, a))[0]
            for a in range(n_actions)
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
        self.rl_epsilon = max(self.rl_min_epsilon, self.rl_epsilon * self.rl_epsilon_decay)

    def weighted_choice(self, synapses):
        total = sum(syn.potential for syn in synapses)
        r = random.uniform(0, total)
        upto = 0
        for syn in synapses:
            upto += syn.potential
            if upto >= r:
                return syn
        return random.choice(synapses)

    def _wander(self, current_neuron, path, current_continue_prob, depth_remaining):
        results = []
        synapses = current_neuron.synapses
        if self.dropout_probability > 0.0:
            synapses = [s for s in synapses if random.random() > self.dropout_probability]
        if (
            depth_remaining <= 0
            or not synapses
            or random.random() > current_continue_prob
        ):
            results.append((current_neuron, path))
            return results
        if (
            len(synapses) > 1
            and random.random() < self.split_probability
        ):
            for syn in synapses:
                next_neuron = self.core.neurons[syn.target]
                w = syn.effective_weight(self.last_context) if hasattr(syn, "effective_weight") else syn.weight
                transmitted_value = self.combine_fn(current_neuron.value, w)
                if hasattr(syn, "apply_side_effects"):
                    syn.apply_side_effects(self.core, current_neuron.value)
                if self.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                    syn.update_fatigue(self.fatigue_increase, self.fatigue_decay)
                inc = self.route_potential_increase
                if syn.potential < 1.0:
                    inc += self.exploration_bonus
                syn.potential = min(self.synapse_potential_cap, syn.potential + inc)
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
            w = syn.effective_weight(self.last_context) if hasattr(syn, "effective_weight") else syn.weight
            transmitted_value = self.combine_fn(current_neuron.value, w)
            if hasattr(syn, "apply_side_effects"):
                syn.apply_side_effects(self.core, current_neuron.value)
            if self.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                syn.update_fatigue(self.fatigue_increase, self.fatigue_decay)
            inc = self.route_potential_increase
            if syn.potential < 1.0:
                inc += self.exploration_bonus
            syn.potential = min(self.synapse_potential_cap, syn.potential + inc)
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
                rep_path = max(group, key=lambda tup: len(tup[1]))[1]
                neuron = self.core.neurons[key]
                neuron.value = avg_value
                merged.append((neuron, rep_path))
        final_neuron, final_path = max(merged, key=lambda tup: tup[0].value)
        if final_path and hasattr(final_path[0][0], "created_at"):
            pathway_start_time = final_path[0][0].created_at
            pathway_age = (datetime.now() - pathway_start_time).total_seconds()
            print(f"Partial pathway age: {pathway_age:.2f} sec")
        return final_neuron, final_path

    def dynamic_wander(self, input_value, apply_plasticity=True):
        with self.lock:
            for neuron in self.core.neurons:
                neuron.value = None
            self.decay_fatigues()
            candidates = [n for n in self.core.neurons if n.synapses]
            entry_neuron = (
                random.choice(candidates)
                if candidates
                else random.choice(self.core.neurons)
            )
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
            results = self._wander(entry_neuron, initial_path, 1.0, depth_limit)
            final_neuron, final_path = self._merge_results(results)
            if not final_path or all(s is None for _, s in final_path):
                if entry_neuron.synapses and self.dropout_probability < 1.0:
                    syn = self.weighted_choice(entry_neuron.synapses)
                    next_neuron = self.core.neurons[syn.target]
                    w = syn.effective_weight(self.last_context) if hasattr(syn, "effective_weight") else syn.weight
                    raw_val = self.combine_fn(entry_neuron.value, w)
                    if hasattr(next_neuron, "process"):
                        next_neuron.value = next_neuron.process(raw_val)
                    else:
                        next_neuron.value = raw_val
                    if hasattr(syn, "apply_side_effects"):
                        syn.apply_side_effects(self.core, entry_neuron.value)
                    if self.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                        syn.update_fatigue(self.fatigue_increase, self.fatigue_decay)
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
            return final_neuron.value, [s for (_, s) in final_path if s is not None]

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
            outputs = pool.starmap(_wander_worker, [(state, input_value, s) for s in seeds])

        return outputs

    def apply_structural_plasticity(self, path):
        ctx = self.last_context
        mod = 1.0 + ctx.get("reward", 0.0) - ctx.get("stress", 0.0)
        for _, syn in path:
            if syn is not None and syn.potential >= self.plasticity_threshold:
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
                self.core.neurons.append(new_neuron)
                new_weight1 = (
                    syn.weight
                    * self.struct_weight_multiplier1
                    * mod
                    * self.structural_learning_rate
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
        path_length = len(path)
        for syn in path:
            if getattr(syn, "frozen", False):
                continue
            source_value = self.core.neurons[syn.source].value
            delta = self.weight_update_fn(source_value, error, path_length)
            if self.gradient_noise_std > 0:
                delta += np.random.normal(0.0, self.gradient_noise_std)
            clip = getattr(self.core, "gradient_clip_value", None)
            if clip is not None:
                delta = float(np.clip(delta, -clip, clip))
            mom = self._momentum.get(syn, 0.0)
            mom = self.momentum_coefficient * mom + delta
            self._momentum[syn] = mom
            update = self.learning_rate * mom
            if abs(update) > self.synapse_update_cap:
                update = math.copysign(self.synapse_update_cap, update)
            syn.weight += update
            if random.random() < self.consolidation_probability:
                syn.weight *= self.consolidation_strength
            if syn.weight > self._weight_limit:
                syn.weight = self._weight_limit
            elif syn.weight < -self._weight_limit:
                syn.weight = -self._weight_limit
            score = abs(error) * abs(syn.weight) / max(path_length, 1)
            self.core.neurons[syn.target].attention_score += score
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
                        1
                        for syn in path
                        if syn.potential >= self.plasticity_threshold
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
                output_value, path = self.dynamic_wander(input_value)
                error = self._compute_loss(target_value, output_value)
                path_length = self.apply_weight_updates_and_attention(path, error)
            else:
                output_value, path = self.dynamic_wander(input_value)
                error = self._compute_loss(target_value, output_value)
                path_length = self.apply_weight_updates_and_attention(path, error)
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
            if self.dropout_decay_rate != 1.0:
                self.dropout_probability *= self.dropout_decay_rate
                self.dropout_probability = float(
                    max(0.0, min(1.0, self.dropout_probability))
                )

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

    def update_synapse_type_attentions(self, path, loss, speed, size):
        for syn in path:
            st = syn.synapse_type
            self.synapse_loss_attention[st] += abs(loss)
            self.synapse_speed_attention[st] += speed
            self.synapse_size_attention[st] += size

    def decide_synapse_action(self):
        if not self.core.synapses:
            return
        create_type = max(self.synapse_loss_attention, key=self.synapse_loss_attention.get)
        remove_type = max(self.synapse_size_attention, key=self.synapse_size_attention.get)
        if random.random() < 0.5:
            src = random.choice(self.core.neurons).id
            tgt = random.choice(self.core.neurons).id
            if src != tgt:
                self.core.add_synapse(
                    src,
                    tgt,
                    weight=random.uniform(0.1, 1.0),
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
