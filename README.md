# Data- and Query-Efficient Black-box Universal Adversarial Attacks with Bayesian Optimization
Makoto Yuito, Kazuki Yoneyama <br>
[Paper Info (JA)](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202302248215714886)

<p align="center">
  <a href="https://github.com/myuito3/universal-bayesopt-attack/blob/main/assets/test_and_adv.jpg">
    <img alt="test_and_adv" src="assets/test_and_adv.jpg" width="100%">
  </a>
</p>

We propose a black-box universal adversarial attack method using Bayesian optimization for efficiently generating an universal adversarial perturbation. In our experiments on ImageNet, proposed method achieves an attack success rate comparable to existing methods (up to 81%) despite using a smaller amount of training data and fewer queries.

## Installation
### Prerequisites
Our code has been tested on Ubuntu 22.04 system with CUDA 11.8 installed. The Bayesian optimisation process we use to search for universal adversarial perturbations relies on [BayesOpt Attack by Ru et al](https://github.com/rubinxin/BayesOpt_Attack).

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
comming soon..
