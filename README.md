# Data- and Query-Efficient Black-box Universal Adversarial Attacks with Bayesian Optimization
Makoto Yuito, Kazuki Yoneyama <br>
[Paper Info (JA)](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202302248215714886)

<p align="center">
  <a href="https://github.com/myuito3/universal-bayesopt-attack/blob/main/assets/test_and_adv.jpg">
    <img alt="test_and_adv" src="assets/test_and_adv.jpg" width="100%">
  </a>
</p>

We propose a black-box universal adversarial attack method using Bayesian optimization for efficiently generating universal adversarial perturbations. In our experiments on ImageNet, the proposed method achieves an attack success rate comparable to existing methods (up to 81%) despite using a smaller amount of training data and fewer queries.

## Installation
### Prerequisites
Our code has been tested on an Ubuntu 22.04 system with CUDA 11.8 installed. The Bayesian optimisation process we use to search for universal adversarial perturbations relies on [BayesOpt Attack by Ru et al](https://github.com/rubinxin/BayesOpt_Attack).

### Setup the environment
First, clone the repository with the following command:
```bash
git clone https://github.com/myuito3/universal-bayesopt-attack.git --recursive
cd universal-bayesopt-attack/
```

Next, install the required libraries:
```bash
pip install -r requirements.txt
```

**Optional**: If you get a cv2 import error when you run the codes, try the following command:
```bash
apt -y update && apt -y upgrade
apt -y install libopencv-dev
```

### Run
To quickly try it out, you can run `python attack.py`.

## Usage
You can set several options in the execution command:
- `--data`: The target model for the attack. Currently supported models are "mnist" and "cifar10" (default "mnist").
- `--setting`: Whether to use logits ("score") or classification classes ("decision") for loss calculation (default "score").
- `--max_iters`: The number of queries to the model (default 2000).
- `--num_train_images`: The number of training images used to generate the universal adversarial perturbation (default 10).
- `--num_test_images`: The number of test images (default 1000).

## License
Unless otherwise credited or noted, this repository is licensed under MIT. Of course, the contents within the submodules folder belong to the respective developers.
