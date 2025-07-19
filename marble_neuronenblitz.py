from marble_imports import *
from marble_core import Neuron, SYNAPSE_TYPES, NEURON_TYPES, perform_message_passing
from marble_base import MetricsVisualizer
import threading


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

        self.combine_fn = (
            combine_fn if combine_fn is not None else (lambda x, w: max(x * w, 0))
        )
        self.loss_fn = (
            loss_fn if loss_fn is not None else (lambda target, output: target - output)
        )
        self.weight_update_fn = (
            weight_update_fn
            if weight_update_fn is not None
            else (lambda source, error, path_len: (error * source) / (path_len + 1))
        )

        self._weight_limit = 1e6

        self.training_history = []
        self.global_activation_count = 0
        self.last_context = {}
        self.type_attention = {nt: 0.0 for nt in NEURON_TYPES}
        self.synapse_loss_attention = {st: 0.0 for st in SYNAPSE_TYPES}
        self.synapse_size_attention = {st: 0.0 for st in SYNAPSE_TYPES}
        self.synapse_speed_attention = {st: 0.0 for st in SYNAPSE_TYPES}
        self.remote_client = remote_client
        self.torrent_client = torrent_client
        self.torrent_map = torrent_map if torrent_map is not None else {}
        self.metrics_visualizer = metrics_visualizer
        self.last_message_passing_change = 0.0
        self.lock = threading.RLock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.RLock()

    def modulate_plasticity(self, context):
        """Adjust plasticity_threshold based on neuromodulatory context."""
        reward = context.get("reward", 0.0)
        stress = context.get("stress", 0.0)
        adjustment = reward - stress
        self.plasticity_threshold = max(0.5, self.plasticity_threshold - adjustment)
        self.last_context = context.copy()

    def reset_neuron_values(self):
        for neuron in self.core.neurons:
            neuron.value = None

    def weighted_choice(self, synapses):
        total = sum(syn.potential for syn in synapses)
        r = random.uniform(0, total)
        upto = 0
        for syn in synapses:
            upto += syn.potential
            if upto >= r:
                return syn
        return random.choice(synapses)

    def _wander(self, current_neuron, path, current_continue_prob):
        results = []
        if not current_neuron.synapses or random.random() > current_continue_prob:
            results.append((current_neuron, path))
            return results
        if (
            len(current_neuron.synapses) > 1
            and random.random() < self.split_probability
        ):
            for syn in current_neuron.synapses:
                next_neuron = self.core.neurons[syn.target]
                transmitted_value = self.combine_fn(current_neuron.value, syn.weight)
                next_neuron.value = transmitted_value
                new_path = path + [(next_neuron, syn)]
                new_continue_prob = current_continue_prob * self.continue_decay_rate
                if next_neuron.tier == "remote" and self.remote_client is not None:
                    remote_out = self.remote_client.process(transmitted_value)
                    next_neuron.value = remote_out
                    results.append((next_neuron, new_path))
                elif next_neuron.tier == "torrent" and self.torrent_client is not None:
                    part = self.torrent_map.get(next_neuron.id)
                    remote_out = self.torrent_client.process(transmitted_value, part)
                    next_neuron.value = remote_out
                    results.append((next_neuron, new_path))
                else:
                    results.extend(
                        self._wander(next_neuron, new_path, new_continue_prob)
                    )
        else:
            syn = self.weighted_choice(current_neuron.synapses)
            next_neuron = self.core.neurons[syn.target]
            transmitted_value = self.combine_fn(current_neuron.value, syn.weight)
            next_neuron.value = transmitted_value
            new_path = path + [(next_neuron, syn)]
            new_continue_prob = current_continue_prob * self.continue_decay_rate
            if next_neuron.tier == "remote" and self.remote_client is not None:
                remote_out = self.remote_client.process(transmitted_value)
                next_neuron.value = remote_out
                results.append((next_neuron, new_path))
            elif next_neuron.tier == "torrent" and self.torrent_client is not None:
                part = self.torrent_map.get(next_neuron.id)
                remote_out = self.torrent_client.process(transmitted_value, part)
                next_neuron.value = remote_out
                results.append((next_neuron, new_path))
            else:
                results.extend(self._wander(next_neuron, new_path, new_continue_prob))
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
        if final_path:
            pathway_start_time = final_path[0][0].created_at
            pathway_age = (datetime.now() - pathway_start_time).total_seconds()
            print(f"Partial pathway age: {pathway_age:.2f} sec")
        return final_neuron, final_path

    def dynamic_wander(self, input_value):
        with self.lock:
            for neuron in self.core.neurons:
                neuron.value = None
            candidates = [n for n in self.core.neurons if n.synapses]
            entry_neuron = (
                random.choice(candidates)
                if candidates
                else random.choice(self.core.neurons)
            )
            entry_neuron.value = input_value
            initial_path = [(entry_neuron, None)]
            results = self._wander(entry_neuron, initial_path, 1.0)
            final_neuron, final_path = self._merge_results(results)
            if not final_path or all(s is None for _, s in final_path):
                if entry_neuron.synapses:
                    syn = self.weighted_choice(entry_neuron.synapses)
                    next_neuron = self.core.neurons[syn.target]
                    next_neuron.value = self.combine_fn(entry_neuron.value, syn.weight)
                    final_path = [(entry_neuron, None), (next_neuron, syn)]
                else:
                    final_path = initial_path
            self.global_activation_count += 1
            if self.global_activation_count % self.route_visit_decay_interval == 0:
                for syn in self.core.synapses:
                    syn.potential *= self.route_potential_decay
            self.apply_structural_plasticity(final_path)
            return final_neuron.value, [s for (_, s) in final_path if s is not None]

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
                new_weight1 = syn.weight * self.struct_weight_multiplier1 * mod
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
                new_weight2 = syn.weight * self.struct_weight_multiplier2 * mod
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

    def get_preferred_neuron_type(self):
        if not any(self.type_attention.values()):
            return "standard"
        best = max(self.type_attention.items(), key=lambda x: x[1])[0]
        for k in self.type_attention:
            self.type_attention[k] *= self.attention_decay
        return best

    def train_example(self, input_value, target_value):
        with self.lock:
            output_value, path = self.dynamic_wander(input_value)
            error = self.loss_fn(target_value, output_value)
            path_length = len(path)
            for syn in path:
                source_value = self.core.neurons[syn.source].value
                delta = self.weight_update_fn(source_value, error, path_length)
                syn.weight += delta
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
            if path:
                self.update_attention(path, error)
            if path:
                self.update_synapse_type_attentions(
                    path,
                    error,
                    len(path),
                    len(self.core.synapses),
                )
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

    def get_training_history(self):
        return self.training_history

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
