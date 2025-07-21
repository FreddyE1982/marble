# MARBLE ML Paradigms Handbook

This handbook explains every machine learning paradigm currently implemented in MARBLE. The material is presented in three versions tailored to different audiences: laymen, ML scientists and high school students. Each section lists all paradigms in the same order.

## Version for Laymen

### Numeric Regression
A basic approach where MARBLE learns to predict numbers from examples. You provide pairs of inputs and their correct outputs. MARBLE adjusts its connections until it can guess the numbers with minimal error.

### Image Classification
MARBLE is trained to recognise objects in pictures. Images are labeled, and the system learns the patterns that correspond to each category.

### Remote Offloading
Parts of the MARBLE brain can run on another computer. This lets large models use more memory or share computation across machines.

### Autograd and PyTorch Challenge
MARBLE can work alongside PyTorch. The challenge compares MARBLE's learning with a PyTorch model while they share training data.

### GPT Training
MARBLE includes a compact transformer that can be trained on text. It learns to predict the next character or token, eventually generating short passages on its own.

### Knowledge Distillation
A smaller MARBLE model can learn from a larger "teacher". Both see the same data, but the teacher's predictions guide the student, making it quicker and lighter.

### Reinforcement Learning
MARBLE can learn by trial and error, such as navigating a small grid world. Positive rewards encourage good actions and negative ones discourage mistakes.

### Contrastive Learning
Without any labels, MARBLE views two slightly different versions of the same input and tries to make their internal representations similar. This leads to features that are useful for later tasks.

### Hebbian Learning
Inspired by neuroscience, MARBLE strengthens connections when neurons activate together. The more often two neurons fire in sequence, the stronger their link becomes.

### Adversarial Learning
Two MARBLE networks compete: one generates data while the other tries to tell real from fake. Over time the generator improves until the discriminator struggles to spot the difference.

### Autoencoder Learning
MARBLE tries to recreate noisy inputs. By forcing the system to clean up the noise, it learns compressed internal representations of the original data.

### Semi-Supervised Learning
Here MARBLE mixes labeled and unlabeled data. Labeled examples provide guidance while unlabeled ones encourage the network to be consistent in its predictions.

### Federated Learning
Several MARBLE instances train on their own data sets. Periodically they average their knowledge, allowing them to learn collaboratively without sharing raw data.

### Curriculum Learning
Training examples are organised from easy to hard. MARBLE gradually tackles more difficult samples as it gets better, leading to stable learning.

### Meta Learning
MARBLE practices on many small tasks so it can adapt quickly to new ones. After learning each task briefly, it updates its main network toward the average of their solutions.

### Transfer Learning
A model trained on one task can be reused for another. MARBLE freezes part of the network and fine-tunes the rest on new data.

### Continual Learning
When tasks arrive one after another, MARBLE remembers previous ones by replaying a few stored examples while training on new data.

### Imitation Learning
MARBLE watches demonstrations and tries to mimic them. It stores pairs of inputs and desired actions, then adjusts itself to reproduce the behaviour.

### Harmonic Resonance Learning
An experimental method where inputs are encoded as sine waves. The frequency gradually changes, guiding MARBLE to capture periodic relationships.

### Synaptic Echo Learning
Each synapse maintains an echo buffer of recent activations. Weight updates scale the normal error term by the mean echo value: (Delta w = ta,	ext{echo}	imes	ext{error}). This links short-term memory with learning dynamics.

### Fractal Dimension Learning
MARBLE monitors the fractal dimension of activations and increases representation size when it grows too high.

### Quantum Flux Learning
Weights are updated with a sinusoidal phase factor that evolves over time, creating oscillatory plasticity.

### Dream Reinforcement Learning
After each real update the network performs short dream rollouts and learns from their errors.

### Continuous Weight Field Learning
Instead of a fixed weight vector, MARBLE learns a smooth function \(W(x)\). Each
input has its own weights, derived from radial basis functions. Neuronenblitz
provides features \(\phi(x)\) and the prediction is \(\phi(x) \cdot W(x)\). A
variational loss with a gradient regulariser ensures the field changes smoothly
across \(x\).
---

## Version for ML Scientists

### Numeric Regression
Given pairs \((x_i, y_i)\), MARBLE minimises mean squared error \(\frac{1}{n}\sum (y_i-\hat{y}_i)^2\) using dynamic wander paths through the Neuronenblitz graph. Gradients flow via custom weight update rules that support CPU and GPU execution.

### Image Classification
Images \(x\in\mathbb{R}^{H\times W\times C}\) are flattened and fed through MARBLE. Cross-entropy loss \(-\sum y\log \hat{y}\) trains the network, optionally with evolutionary pruning and mutation. Asynchronous updates allow inference during background training.

### Remote Offloading
High-attention lobes are serialized and transmitted to a `RemoteBrainServer`. The offload threshold triggers when attention exceeds a configured value. Communication latency is managed through retry and timeout parameters in `config.yaml`.

### Autograd and PyTorch Challenge
`MarbleAutogradLayer` wraps the brain so PyTorch's autograd computes gradients. A SqueezeNet baseline is trained alongside MARBLE while neuromodulatory stress adjusts plasticity when MARBLE underperforms. Comparative metrics include accuracy and energy usage.

### GPT Training
A lightweight transformer with `num_layers`, `num_heads` and `hidden_dim` specified in YAML implements the standard attention mechanism:
\[\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.\]
Sequences of length `block_size` are embedded and optimised using Adam with learning rate `gpt.learning_rate`. CuPy provides GPU acceleration when available.

### Knowledge Distillation
For examples \((x, y)\), the teacher produces \(t=f_T(x)\). The student trains on targets \((1-\alpha)y+\alpha t\). Loss becomes
\[L=(1-\alpha)\|y-\hat{y}\|^2+\alpha\|t-\hat{y}\|^2.\]
This blends ground truth and teacher output, enabling smaller models to approximate larger ones.

### Reinforcement Learning
In GridWorld, state-action values \(Q(s,a)\) are updated using
\[Q(s,a) \leftarrow Q(s,a)+\eta\bigl(r+\gamma \max_{a'}Q(s',a')-Q(s,a)\bigr).\]
States are encoded numerically for Neuronenblitz. Epsilon-greedy exploration decays from `epsilon_start` to `epsilon_min`.

### Contrastive Learning
Two augmented views \(v_1,v_2\) of each sample yield representations \(r_1,r_2\). The InfoNCE loss
\[L=-\frac{1}{N}\sum_i \log \frac{\exp(r_{2i} \cdot r_{2i+1}/\tau)}{\sum_j \exp(r_{2i} \cdot r_j/\tau)}\]
encourages positive pairs to be close while all others repel.

### Hebbian Learning
Weights along a path are adjusted by
\[\Delta w = \eta\,x_{\text{pre}}\,x_{\text{post}} - \lambda w,\]
where \(x_{\text{pre}}\) and \(x_{\text{post}}\) are neuron activations and \(\lambda\) is `weight_decay`. This unsupervised rule captures correlation structure.

### Adversarial Learning
Generator parameters \(\theta_G\) and discriminator parameters \(\theta_D\) are optimised in alternating fashion using
\[\min_{\theta_G}\max_{\theta_D} \;\mathbb{E}_{x\sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z\sim p_z}[\log (1-D(G(z)))] .\]
Both networks share MARBLE's Core for message passing during updates.

### Autoencoder Learning
For input \(x\) with noise \(\tilde{x}=x+\epsilon\), MARBLE minimises reconstruction loss \(\|x - f(\tilde{x})\|^2\). The learned representation resides in neuron embeddings updated via the custom weight update function.

### Semi-Supervised Learning
Supervised loss \(L_s\) on labeled pairs is combined with consistency loss \(L_u=\|f_u(x)-f_u'(x)\|^2\) from two stochastic passes over unlabeled input. Total loss is \(L_s + \lambda L_u\).

### Federated Learning
Each client trains locally for `local_epochs`, producing weight vector \(w_i\). Federated averaging computes \(\bar{w}=\frac{1}{N}\sum_i w_i\). The aggregator sets each client's weights to \(\bar{w}\), and communication repeats for `rounds` cycles.

### Curriculum Learning
Samples ranked by difficulty are introduced progressively. Epoch \(e\) uses the first \(k(e)\) samples where \(k(e)\) increases linearly or exponentially with `epochs`. Loss over gradually harder data stabilises optimisation.

### Meta Learning
For task set \(\{T_j\}\), each temporary clone trains for `inner_steps` producing weights \(w_j\). After all tasks, the main network weights update via
\[w \leftarrow w + \beta \frac{1}{m}\sum_j (w_j - w),\]
with meta step size \(\beta = \text{meta_lr}\). This approximates the Reptile algorithm.

### Transfer Learning
Given pretrained weights, a fraction `freeze_fraction` of synapses remains fixed. The rest fine-tune using standard supervised loss on the new dataset. This reuses prior knowledge while adapting to new data.

### Continual Learning
Experience replay stores a memory buffer of size `memory_size`. For each new sample, a random memory item is replayed. This approximates joint training to reduce catastrophic forgetting.

### Imitation Learning
Demonstration pairs \((s,a)\) are recorded. Loss is \(\|a - f(s)\|^2\). Training over stored pairs clones the expert policy and can be combined with other learning signals.

### Harmonic Resonance Learning
Inputs are represented as sinusoidal embeddings: \(r=[\sin(\omega x), \cos(\omega x)]\). After each step, frequency \(\omega\) decays by `decay`, allowing exploration of resonant structures in the data.

### Synaptic Echo Learning
### Fractal Dimension Learning
MARBLE monitors the fractal dimension of activations and increases representation size when it grows too high.

### Quantum Flux Learning
Weights are updated with a sinusoidal phase factor that evolves over time, creating oscillatory plasticity.

### Dream Reinforcement Learning
After each real update the network performs short dream rollouts and learns from their errors.
Each synapse maintains an echo buffer of recent activations. Weight updates scale the normal error term by the mean echo value: \(\Delta w = \eta\,\text{echo}\times\text{error}\). This links short-term memory with learning dynamics.

---

## Version for High School Students

### Numeric Regression
MARBLE learns to guess numbers by looking at lots of examples. It measures how far its guesses are from the real answers and keeps adjusting until the mistakes are tiny.

### Image Classification
The system studies many labeled pictures so it can say what is shown in new images. It's like teaching it to recognise cats, dogs and other objects.

### Remote Offloading
If a single computer is not enough, parts of MARBLE can run on a second machine. They communicate over the network so training can continue even with limited local resources.

### Autograd and PyTorch Challenge
MARBLE can connect to PyTorch, a popular deep learning library. The challenge compares how MARBLE learns with a regular PyTorch model so you can see the differences.

### GPT Training
A simplified transformer lets MARBLE generate text. After feeding it sentences, it predicts the next characters and slowly learns to write short stories.

### Knowledge Distillation
A large "teacher" model guides a smaller "student" by showing what it would predict. The student copies these hints, ending up almost as smart but much lighter.

### Reinforcement Learning
MARBLE can learn through rewards. In a grid world game, it gets points for reaching the goal and learns which moves work best through trial and error.

### Contrastive Learning
Without needing labels, MARBLE looks at two versions of the same example (like a rotated image) and makes their internal signals match. That way it learns useful features by itself.

### Hebbian Learning
This rule says that if two neurons activate together, their connection grows stronger. MARBLE uses this idea to find patterns without any labels at all.

### Adversarial Learning
Two networks play a game: one tries to create fake data while the other tries to spot the fake. They keep improving until the faker becomes very convincing.

### Autoencoder Learning
By adding noise to data and asking MARBLE to rebuild the original, it learns efficient ways to represent the information. This is helpful for compression or denoising.

### Semi-Supervised Learning
Sometimes we only label part of the data. MARBLE mixes the labeled pieces with the unlabeled ones, learning from both at once for better results.

### Federated Learning
Imagine several phones each training their own copy of MARBLE on local data. They share only their updated weights, not the raw data, and average them to improve together.

### Curriculum Learning
Just like schoolwork gets harder over time, MARBLE starts with easy examples and gradually moves to tougher ones. This helps it learn smoothly without getting stuck.

### Meta Learning
MARBLE practices on many small problems so it can adjust rapidly to a brand new one. It's a bit like learning how to learn.

### Transfer Learning
A model trained on one task can help with another. MARBLE keeps most of what it already knows but tweaks some connections to handle new data.

### Continual Learning
When tasks arrive one after the other, MARBLE saves a few old examples and reviews them during new training. That way it doesn't forget what it learned earlier.

### Imitation Learning
By watching an expert perform actions, MARBLE records those moves and practices until it can imitate them on its own.

### Harmonic Resonance Learning
This experimental idea turns numbers into sine waves before feeding them into MARBLE. The frequency slowly changes, helping the system notice repeating patterns.

### Synaptic Echo Learning
### Fractal Dimension Learning
MARBLE monitors the fractal dimension of activations and increases representation size when it grows too high.

### Quantum Flux Learning
Weights are updated with a sinusoidal phase factor that evolves over time, creating oscillatory plasticity.

### Dream Reinforcement Learning
After each real update the network performs short dream rollouts and learns from their errors.
MARBLE keeps short memories of recent neuron activity. These echoes influence how the connections change, giving the network a sense of its immediate past.

### Continuous Weight Field Learning
Each input has its own weights generated from a smooth field. MARBLE uses the
current neuron representations to compute a prediction and adjusts the field so
nearby inputs share similar weights while matching their targets.

