## Uncertainty Estimation in CycleGAN
Official PyTorch Code for Uncertainty Estimation in Unsupervised MR-CT Synthesis of Scoliotic Spines published at the IEEE Open Journal of Engineering in Medicine and Biology (OJEMB). The paper can be found [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10086579). 

### Abstract
Uncertainty estimations through approximate Bayesian inference provide interesting insights to deep neural networks' behavior. In unsupervised learning tasks, where expert labels are unavailable, it becomes ever more important to critique the model through uncertainties. 
This paper presents a proof-of-concept for generalizing the aleatoric and epistemic uncertainties in unsupervised MR-CT synthesis of scoliotic spines. A novel adaptation of the cycle-consistency constraint in CycleGAN is proposed such that the model predicts the aleatoric uncertainty maps in addition to the standard volume-to-volume translation between Magnetic Resonance (MR) and Computed Tomography (CT) data. 
Ablation experiments were performed to understand uncertainty estimation as an implicit regularizer and a measure of the model's confidence. The aleatoric uncertainty helps in distinguishing between the bone and soft-tissue regions in CT and MR data during translation, while the epistemic uncertainty provides interpretable information to the user for downstream tasks.


<p align="center">
  <img width="600" alt="flowchart_updated" src="https://user-images.githubusercontent.com/53445351/228790080-cfad9d7f-2b12-492d-9b6e-e3ffab49f390.png">
</p>

### Structure of the Repository

1. `datasets3D_pl.py` - code for loading the data from the respective folders. 
2. `losses_pl.py` - contains the loss functions used in the main training script.
3. `models3D_dropout_pl.py` - contains the generator and discriminator architectures for 3D CycleGAN with uncertainty estimations.
4. `utils3D_pl.py` - code for all the necessary utility functions.
5. `train3D_pl.py` - main code file for training 3D CycleGAN, calls the above-mentioned files for various functions. 
6. `test3D_pl.py` - main code for testing the CycleGAN model and getting the uncertainty estimates. 


### Citation

If you find this code useful in your research, please consider citing:

TODO:


