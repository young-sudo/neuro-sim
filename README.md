# Computational Neuroscience Simulations

*by Younginn Park*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003366?style=for-the-badge&logo=plotly&logoColor=white) 

Sample simulations of selected fundamental **Computational Neuroscience** models, which form the theoretical foundation for understanding biological computation and provided the initial inspiration for artificial neural networks (ANNs).

## Contents
### 1. [Integrate-and-Fire (IF) model](https://github.com/young-sudo/neuro-sim/tree/2ec869934b392e88559a2f1cdce41c3c4b3d508c/if_model)

A simplified neuron model that integrates incoming signals and generates spikes once a threshold potential is reached.

Integrate-and-fire neuron model with averaged excitatory ($_E$) and inhibitory ($_I$) inputs from chemical synapses. A simple model for neuronal activity (measured in membrane potential).

Definitions:
- $V$ - neuron activity/membrane potential ($V_r$ - resting potential)
- $S$ - synaptic activation function ($0 \leq S_i(t) \leq 1$)
- $w$ - weight for the synaptic activity
- $f$ - firing rate (opening rate of the receptor channel)
- $\tau$ - membrane time constant of the neuron

Equations:

$$
\begin{cases}
\tau \frac{dV}{dt} = -(V - V_r) - w_E S_E(t) V - w_I S_I(t)(V-V_I) \\
\frac{S_E}{dt}     = -\frac{S_E}{\tau_E} + (1-S_E)f_E \\
\frac{S_I}{dt}     = -\frac{S_I}{\tau_I} + (1-S_E)f_I
\end{cases}
$$

Simplifications:
- $f = f_E = f_I$
- $V_E=0$, 'reversal potential' - membrane potential at which the direction of current changes from positive to negative

<div align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/neuro-sim/main/img/if.png" alt="if" width="500"/>
    <br>
    <figcaption style="text-align:center;"><em>Figure 1. Example of IF model simulation output</em></figcaption>
  </figure>
</div>

### 2. [Hodgkin-Huxley (HH) model](https://github.com/young-sudo/neuro-sim/tree/2ec869934b392e88559a2f1cdce41c3c4b3d508c/hh_model)

A biophysical model describing how action potentials in neurons arise from ionic currents across the cell membrane.

Definitions:
- $V$ - membrane potential (voltage)
- $C$ - electric capacitance ($1\mu F/cm^2)$
- $g_{K}$, $g_{Na}$, $g_{L}$ - maximum conductance of ions ($K^+, Na^+$ and the rest, respectively) - in reality it varies through time (via e.g. chemical changes), but this simulation will be simplified and the value will be set to constant, let $g_K = 30 ms/cm^2$, $g_{Na} = 120 ms/cm^2$, $g_L = 0.1 ms/cm^2$
- $V_{K}$, $V_{Na}$, $V_{L}$ - reversal potential of ions ($K^+, Na^+$ and the rest, respectively), varies in time in reality, but practically constant, let $V_K = -80 mV$, $V_{Na} = 50 mV$, $V_L = -65 mV$
- $n, m, h$ - gating variables, they are dependent on time and voltage (membrane potential), they have values $0 \leq n,m,h \leq 1$
- $\alpha, \beta$ - coefficients that depend on voltage (membrane potential), for each of the gating variables
- $I_{app}$ - applied input current

Equations:

$$
\begin{cases}
C \frac{dV}{dt} = - g_K n^4(V - V_k) - g_{Na} m^3h(V - V_{Na}) - g_L(V - V_L) + I_{app} \\
\frac{dn}{dt} = \alpha_n (1 - n) - \beta_n n \\
\frac{dm}{dt} = \alpha_m (1 - m) - \beta_m m \\
\frac{dh}{dt} = \alpha_h (1 - h) - \beta_h h \\
\end{cases}
\begin{cases}
\alpha_n = \frac{0.1 (V + 55)}{1 - e^{-0.1 (V + 55)}} \\
\beta_n = 0.125 e ^{-0.0125 (V + 65)} \\
\alpha_m = \frac{0.1 (V + 40)}{1 - e^{-0.1 (V + 40)}} \\
\beta_m = 4 e ^{-0.05 (V + 65)} \\
\alpha_h =  0.07 e^{-0.05 (V + 65)} \\
\beta_h = \frac{1}{1 + e^{-0.1 (V + 35)}} \\
\end{cases}
$$

<div align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/neuro-sim/main/img/hh.png" alt="hh" width="500"/>
    <br>
    <img src="https://raw.githubusercontent.com/young-sudo/neuro-sim/main/img/random_spike.png" alt="spike" width="500"/>
    <br>
    <img src="https://raw.githubusercontent.com/young-sudo/neuro-sim/main/img/random_noise.png" alt="noise" width="500"/>
    <br>
    <figcaption style="text-align:center;"><em>Figure 2. Example of HH model simulation output with a constant input current, random spike train input, and random noise input</em></figcaption>
  </figure>
</div>

### 3. [Bienenstock-Cooper-Munro (BCM) model](https://github.com/young-sudo/neuro-sim/tree/2ec869934b392e88559a2f1cdce41c3c4b3d508c/bcm_model)

A synaptic plasticity model explaining how neurons adjust connection strengths based on activity history to support learning and memory. The fixed points represent stable or unstable synaptic states that determine whether a synapse weakens, remains unchanged, or strengthens over time.

Definitions:
- $f$ - firing rate of the presynaptic neuron (here, simplified: fixed)
- $r$ - firing rate of the postsynaptic neuron
- $w$ - synaptic weights
- $\theta$ - sliding threshold
- $\tau_w, \tau_\theta, \tau_r$ - time constants for weights, sliding threshold and firing rate of the postsynaptic neuron, respectively
- Constants:
    - $\epsilon$ - time constant of uniform decay (small value)
    - $\lambda$ - scaling constant for the change of weights (learning rate)
    - $\alpha$ - sensitivity constant (how sliding threshold is sensitive to firing rate)
    - $\beta$ - scaling constant for the efficacy of the presynaptic neuron on the postsynaptic neuron

$$
\begin{cases}
\tau_w \frac{dw}{dt} = \lambda f r (r - \theta) - \epsilon w \\
\tau_\theta \frac{d\theta}{dt} = - \theta + \alpha r^2 \\
\tau_r \frac{dr}{dt} = - r + \beta w f
\end{cases}
$$

<div align="center" style="margin-top: 50px; margin-bottom: 50px;">
  <figure>
    <img src="https://raw.githubusercontent.com/young-sudo/neuro-sim/main/img/bcm.png" alt="bcm" width="500"/>
    <br>
    <figcaption style="text-align:center;"><em>Figure 3. Example of BCM model simulation's phase portrait and its fixed points</em></figcaption>
  </figure>
</div>

<br>

_More details about the models and the parameters in each of the project folders_
