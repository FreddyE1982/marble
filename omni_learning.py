from marble_core import Core, perform_message_passing
from marble_neuronenblitz import Neuronenblitz
from contrastive_learning import ContrastiveLearner
from hebbian_learning import HebbianLearner
from adversarial_learning import AdversarialLearner
from autoencoder_learning import AutoencoderLearner
from semi_supervised_learning import SemiSupervisedLearner
from transfer_learning import TransferLearner
from continual_learning import ReplayContinualLearner
from curriculum_learning import CurriculumLearner
from federated_learning import FederatedAveragingTrainer
from reinforcement_learning import MarbleQLearningAgent, GridWorld
from imitation_learning import ImitationLearner
from harmonic_resonance_learning import HarmonicResonanceLearner
from synaptic_echo_learning import SynapticEchoLearner
from dream_reinforcement_learning import DreamReinforcementLearner
from quantum_flux_learning import QuantumFluxLearner
from fractal_dimension_learning import FractalDimensionLearner
from continuous_weight_field_learning import ContinuousWeightFieldLearner
from neural_schema_induction import NeuralSchemaInductionLearner


class OmniLearner:
    """Unified learner that orchestrates all paradigms simultaneously."""

    def __init__(self, core: Core, nb: Neuronenblitz) -> None:
        self.core = core
        self.nb = nb
        self.contrastive = ContrastiveLearner(core, nb)
        self.hebbian = HebbianLearner(core, nb)
        self.adv_discriminator = Neuronenblitz(core)
        self.adversarial = AdversarialLearner(core, nb, self.adv_discriminator)
        self.autoencoder = AutoencoderLearner(core, nb)
        self.semi_supervised = SemiSupervisedLearner(core, nb)
        self.transfer = TransferLearner(core, nb)
        self.continual = ReplayContinualLearner(core, nb)
        self.curriculum = CurriculumLearner(core, nb)
        self.federated = FederatedAveragingTrainer([(core, nb)])
        self.imitation = ImitationLearner(core, nb)
        self.harmonic = HarmonicResonanceLearner(core, nb)
        self.echo = SynapticEchoLearner(core, nb)
        self.dream_rl = DreamReinforcementLearner(core, nb)
        self.quantum = QuantumFluxLearner(core, nb)
        self.fractal = FractalDimensionLearner(core, nb)
        self.weight_field = ContinuousWeightFieldLearner(core, nb)
        self.schema = NeuralSchemaInductionLearner(core, nb)
        self.env = GridWorld()
        self.rl_agent = MarbleQLearningAgent(core, nb)

    def train_step(self, sample: tuple[float, float]) -> None:
        inp, target = sample
        # supervised
        self.imitation.train_step(inp, target)
        self.transfer.train_step(inp, target)
        self.adversarial.train_step(target)
        # semi-supervised / unsupervised
        self.contrastive.train([inp, inp])
        self.hebbian.train_step(inp)
        self.autoencoder.train_step(inp)
        self.semi_supervised.train_step((inp, target), inp)
        self.curriculum.train([(inp, target)], epochs=1)
        self.continual.train_step(inp, target)
        self.harmonic.train_step(inp, target)
        self.echo.train_step(inp, target)
        self.dream_rl.train_episode(inp, target)
        self.quantum.train_step(inp, target)
        self.fractal.train_step(inp, target)
        self.weight_field.train_step(inp, target)
        self.schema.train_step(inp)
        state = self.env.reset()
        action = self.rl_agent.select_action(state, self.env.n_actions)
        next_state, reward, done = self.env.step(action)
        self.rl_agent.update(state, action, reward, next_state, done)
        perform_message_passing(self.core)
        self.federated.aggregate()

    def train(self, data: list[tuple[float, float]], epochs: int = 1) -> None:
        for _ in range(epochs):
            for sample in data:
                self.train_step(sample)

