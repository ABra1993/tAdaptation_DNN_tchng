# CNN_adaptation
Studying temporal dynamics in Convolutional Neural Networks (CNNs). Model architectures include feedforward and recurrent CNNs as well as an adpatation of the feedforward CNNs with intrinsic adaptation.

- cnn_sat.py: this script simulates (for n timepoints) a feedforward CNN, that has been previously developed by Spoerer and colleagues (2020). Different model architectures can be chosen which differ in either depth, feature maps or kernels.
- cnn_sat_adapt.py: this script simulates (for n timepoints) a feedforward CNN - same architectures as for the cnn_sat.py - implemented with intrinsic adaptation, as has been done previously by Vinken et al. (2020).
- cnn_sat_rnn.py: this script simulates a recurrent CNN, that has been previously developed by Spoerer and colleagues (2020).

Pretrained weights area available for all networks (link to the OSF page can be found in Spoerer et al, 2020), where for the recurrent CNN, a number of 8 timesteps has been used. Results of the simulations can be found in the visualizations folder.

# References
Spoerer, C. J., Kietzmann, T. C., Mehrer, J., Charest, I., & Kriegeskorte, N. (2020). Recurrent neural networks can explain flexible trading of speed and accuracy in biological vision. PLoS computational biology, 16(10), e1008215.

Vinken, K., Boix, X., & Kreiman, G. (2020). Incorporating intrinsic suppression in deep neural networks captures dynamics of adaptation in neurophysiology and perception. Science advances, 6(42), eabd4205.
