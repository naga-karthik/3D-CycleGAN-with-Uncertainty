## 3D-CycleGAN-with-Uncertainty
Official PyTorch Code for Uncertainty Estimation in Unsupervised MR-CT Synthesis of Scoliotic Spines (IEEE OJEMB)

### Abstract
TODO


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


