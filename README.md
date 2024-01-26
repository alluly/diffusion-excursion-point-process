## Inference and Sampling of Point Processes from Diffusion Excursions

Code for the paper [_Inference and Sampling of Point Processes from Diffusion Excursions_](https://proceedings.mlr.press/v216/hasan23a.html) accepted in UAI 2023 (spotlight). 

Please cite the following paper if the repository was helpful:

```
@InProceedings{pmlr-v216-hasan23a,
  title = 	 {Inference and sampling of point processes from diffusion excursions},
  author =       {Hasan, Ali and Chen, Yu and Ng, Yuting and Abdelghani, Mohamed and Schneider, Anderson and Tarokh, Vahid},
  booktitle = 	 {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {839--848},
  year = 	 {2023},
  editor = 	 {Evans, Robin J. and Shpitser, Ilya},
  volume = 	 {216},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {31 Jul--04 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v216/hasan23a/hasan23a.pdf},
  url = 	 {https://proceedings.mlr.press/v216/hasan23a.html},
  abstract = 	 {Point processes often have a natural interpretation with respect to a continuous process. We propose a point process construction that describes arrival time observations in terms of the state of a latent diffusion process. In this framework, we relate the return times of a diffusion in a continuous path space to new arrivals of the point process. This leads to a continuous sample path that is used to describe the underlying mechanism generating the arrival distribution. These models arise in many disciplines, such as financial settings where actions in a market are determined by a hidden continuous price or in neuroscience where a latent stimulus generates spike trains. Based on the developments in Itô’s excursion theory, we propose methods for inferring and sampling from the point process derived from the latent diffusion process. We illustrate the approach with numerical examples using both simulated and real data. The proposed methods and framework provide a basis for interpreting point processes through the lens of diffusions.}
}
```

### Running the code

To run the multi dimensional point process experiments, please run `train_fp_nd.py'

To run the renewal process experiments, please run `train_fp_args.py $exp` where `$exp` is one of the following:
1. `lognormal`
2. `weibull`
3. `gamma`
4. `exp`
which correspond to the related renewal density. 

To run the neuron experiment, please run `train_fp_args.py neuron-nohistp`.

To run the history and exogenous signal experiments, please refer to history-dep/train_fp_hist.py and run the script.

To plot Figure 1, please run `pathdecomp.py`.
To plot Figure 2, please run `twod_fig.py`.
To plot Figure 3, please run `train_fp_args.py lognormal-nt` and use the figures named `reached.pdf` and `p_t_reg-mle.pdf`.

To plot the multi dimensional point process results, please run `plot_nd.py` after running `train_fp_nd.py`.


