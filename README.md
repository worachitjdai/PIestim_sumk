# Large width penalization for neural network-based prediction interval estimation
Forecasting accuracy in highly uncertain environments is challenging due to the stochastic nature of such systems. Deterministic forecasting, which provides only point estimates, is insufficient to capture potential outcomes. Therefore, probabilistic forecasting has gained significant attention due to its ability to quantify uncertainty, where one of the approaches is to express it as a prediction interval (PI), that explicitly shows upper and lower bounds of predictions associated with a confidence level. High-quality PI is characterized by a high PI coverage probability (PICP) and a narrow PI width. In many real-world applications, the PI width is generally used in risk management to prepare resources that improve reliability and effectively manage uncertainty. A wider PI width results in higher costs for backup resources as decision-making processes generally focus on the worst-case scenarios arising with large PI widths under extreme conditions. This study aims to reduce the large PI width from the PI estimation method by proposing a new PI loss function that penalizes the average of the large PI widths more heavily. Additionally, the proposed formulation is compatible with gradient-based algorithms, the standard approach to training neural networks (NNs), and integrating state-of-the-art NNs and existing deep learning techniques. Experiments with the synthetic dataset reveal that our formulation significantly reduces the large PI width while effectively maintaining the PICP to achieve the desired probability. The practical implementation of our proposed loss function is demonstrated in solar irradiance forecasting, highlighting its effectiveness in minimizing the large PI width in data with high uncertainty and showcasing its compatibility with more complex neural network models. Therefore, reducing large PI widths from our method can lead to significant cost savings by over-allocation of reserve resources.

The proposed formulation is called the Sum-$k$ formulation, which can be shown as

```math
\begin{equation}
	\mathcal{L}_{\text{sum}-k}(\theta) = \max(0, (1-\delta) - \text{PICP}(\theta)) + \gamma \mathcal{W}(\theta),
\end{equation}
```
where
```math
\begin{equation}
\mathcal{W}(\theta) := \mathcal{W}(\theta | K, \lambda) = \frac{1}{R_{\text{quantile}}} \left [ \frac{1}{K} \sum_{i=1}^{K} w_{[i]}(\theta) + \lambda \cdot \frac{1}{N-K} \sum_{K+1}^{N} w_{[i]}(\theta) \right ]
\end{equation}
```
<p align="center">
  <img src="piresult_example_new.png" alt="The example of the PI result" width="600">
</p>

This repository consists of the following folders.
- **codes** consist of an experiment for the simulated and solar datasets. 
  - **Formulation** consists of .py file for P1 and P3 formulation which reformulated as linear programming.
  are utilized for downloading data, cleaning it, and generating datasets.
- **data** contains solar and simulated data used in the experiment training and testing process.
- **figures** contains all figures and a Python notebook file for visualization.ipynb generates all figures in the conference paper.
- **results** store all the CSV, pkl, npy files used in this project
```
|-- experiment
|   |-- pi_characteristics
|   |-- benchmark
|   |-- input_data
|   |-- hyperparameter
|-- paper_figure
|   |-- visualization.ipynb
|   |-- saved_figures
|-- utils
|   |-- formulations.py
|   |-- networks.py
|   |-- trainer.py
|-- demo.ipynb
```
