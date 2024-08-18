<!-- ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
``` -->
# Leveraging Koopman Operators and Deep Neural Networks for Parameter Estimation and Future Prediction
This repository contains the code and resources for the paper titled "Leveraging Koopman Operators and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators." Initially presented at the Iranian Conference on Sound and Vibration (ISAV), the work was selected by David Younesian as a leading paper and subsequently invited for publication in the Journal of Theoretical and Applied Vibration and Acoustics (TAVA).

You can also find the first iteration of this work in the [Koopman-Operator-and-Deep-Neural-Networks-ISAV2023](https://github.com/yriyazi/Koopman-Operator-and-Deep-Neural-Networks-ISAV2023) repository.


## Overview
In this research, we present a novel methodology that combines the Koopman operator with deep neural networks to create a linear representation of the Duffing oscillator. Our approach facilitates effective parameter estimation and accurate future behavior prediction of the oscillator. A modified loss function enhances the training process, making this method a powerful tool for analyzing nonlinear systems, advancing predictive modeling, and exploring diverse applications in science and engineering.
Key Contributions:

### Key Contributions
  * Koopman Operators & DNN Integration: Combines the strengths of Koopman operators with deep neural networks for linear representation of nonlinear systems.
  * Modified Training Structure: Proposes a new data-feeding approach to train the neural network on diverse behaviors.
  * Robustness & Accuracy: Demonstrates the method's robustness against noise and its capability to predict complex behaviors like period-doubling bifurcation.

## Abstract
Nonlinear dynamical systems play a crucial role in scientific and engineering applications. Traditional methods for analyzing these systems often rely on predefined models or costly simulations, limiting their generalizability. In this research, we propose a new methodology that builds a linear representation of the Duffing oscillator by integrating Koopman operators with deep neural networks, without requiring prior knowledge of the system. This approach improves interpretability, reduces training complexity, and provides robust parameter estimation and future behavior prediction. Notably, the method captures the spectrum of Duffing equation behaviors, including periodic oscillations and bifurcations, while maintaining computational efficiency.

```diff
+ Keywords: Koopman operator, parameter estimation, nonlinear dynamical systems, neural networks, period-doubling.
```

## Repository Structure


plaintext

│   config.yaml
│   LICENSE
│   Loader.ipynb
│   test.py
│   Train.ipynb
│
├───dataset
│   │   local_equations.py
│   │   README.md
│   │   test.ipynb
│
├───deeplearning
│   │   Base.py
│   │   README.md
│

├───loss
│   │   Koopman_repeat.py
│   │   loss_function.py
│   │   README.md
│
├───model
│   │   decoder.py
│   │   encoder.py
│   │   README.md
│   │   structure.py
│
└───utils

## Getting Started
To get started with using the code and reproducing the results presented in the paper, please refer to the `Train.ipynb` file.


## Citation

If you find this work useful, please consider citing our paper:

```
[Yassin Riyazi, Navidreza Ghanbari, Arash Bahrami* 2024. Leveraging Koopman Operator and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators. Journal of Theoretical and Applied Vibration and Acoustics (TAVA)]
```

```
@article{riyazi2024leveraging,
  title={Leveraging Koopman Operator and Deep Neural Networks for Parameter Estimation and Future Prediction of Duffing Oscillators},
  author={Yassin Riyazi, Navidreza Ghanbari, Arash Bahrami*},
  journal={Journal of Theoretical and Applied Vibration and Acoustics (TAVA)},
  year={2024}
}
```

## Contact
For any inquiries, issues, or collaboration opportunities, please reach out via email at [iyasiniyasin98@gmail.com].
We hope this work will inspire further advancements in nonlinear dynamical systems analysis and prediction. Happy researching!