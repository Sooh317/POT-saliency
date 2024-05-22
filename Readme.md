# Understanding Parameter Saliency via Extreme Value Theory

## Abstract
The concept of parameter saliency was proposed to diagnose convolutional neural networks (CNNs) by ranking convolution filters that may have caused misclassification for an image. Identifying sensitive parameters that are unique to an incorrectly classified image gives us information to explain model behaviors from a different perspective of saliency map. However, we found that the original parameter saliency ranking method has a bias when gradient norm distributions are heavy-tailed, which is often the case. To mitigate this bias, we revisit the original ranking method from a statistical perspective and show that the method can be reformulated as statistical anomaly detection. This reformulation enables us to use a method that is less biased and more robust to heavy-tailed distributions on the basis of extreme value theory (EVT). Through experiments, we confirmed that the bias inherent in the original ranking method induces a problem in domain shift settings and that the bias is mitigated by the reformulation based on EVT.

## Code Explanation

### Installation
To run the code, you need to install the required Python libraries. You can do this by running:

```bash
pip install -r requirements.txt
```


### Function Example: `imagenet(args, net, dataset_path, model_helpers_root_path, algo='POT', do_stats=True, do_finetuning=True)`

This function is used to conduct the reproducibility experiments. Below is a detailed explanation of its parameters:

- `args`: The arguments needed for the experiment.
- `net`: The neural network model to be used.
- `dataset_path`: The path to the ImageNet dataset.
- `model_helpers_root_path`: The root path where model helper files are stored.
- `algo`: The algorithm to be used. Options are:
  - `baseline`: The baseline method as described in Levin et al., 2022.
  - `POT`: The method proposed in this work.
  - `conv5`: Another comparative method.
- `do_stats`: If set to `True`, this performs pre-computation for estimators.
- `do_finetuning`: If set to `True`, this checks for selective parameter correction.

