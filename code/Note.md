# Note

We have only one version of the model v1, with multiple variants for WLASL100, WLASL300, WLASL1000 and WLASL2000 of the WLASL dataset.
I wanted to streamline into one file and make it configurable for ease of use, but we used different optimization techniques for variants to improve the training efficiency and also the model. 

One of the examples is that, we implemented weighted sampler in the wlasl100 to improve its results, but did not get a chance to try the same for other variants. We used scheduler and optimizer for all variants except for 2000 variant due to time constraints. 

To allow new developers to build on our work, we are providing the files as they are. I will re-factor the file structures in the future as I get time to do so. Feel free to explore!