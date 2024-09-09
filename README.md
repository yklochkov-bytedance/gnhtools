# Gauss-Newton Hessian tools

This repository presents tools for vector manipulations associated with Gauss-Newton Hessian (GNH). Everywhere below Gauss-Newton Hessian is called GNH or simply Hessian. The tools include: 
- calculation of Hessian-vector products (HVP)
- Hessian sketching for analysis of top eigenvalues, estimation of trace and squared trace
- estimation inverse-Hessian-vector products (iHVP) with LiSSA (Linear time Stochastic Second order Algorithm, Koh and Liang (2016)[^1])


## GNH definition

Suppose, we have the classification problem $(x, y)\in\mathcal{X}\times[1..K]$, where $K$ is the number of classes. Suppose, our model produces logits $h(x, \theta)$, which we then pass to CrossEntropyLoss to calculate the loss $ \ell((x, y); \theta) = \ell_{CE}(h(x, \theta), y)$.

Then, the GNH is defined as follows,

$$H=\frac{1}{|\mathcal{D}|}\sum_{(x, y)\in\mathcal{D}}(J_{\theta}h(x,\theta))\nabla_{h}^{2}\ell_{CE}(h(x;\theta),y)(J_{\theta} h(x,\theta))^{\top}$$

Here, $J_{\theta}h(x;\theta)=\left(\frac{{\partial}h_j}{{\partial}\theta_i}\right)_{ij}\in\mathbb{R}^{\mathrm{dim}(\theta){\times}K}$.

For elaborate discussion see Martens (2020)[^2]

## HVP at batch

For a given batch $B\subset\mathcal{D}$, $\tilde{H}$ denotes the GNH calculated on instances from the batch, i.e. the sampled version.

We calculate the in-batch HVP as follows

$$\tilde{H}v =\nabla_{\theta}\frac{1}{|B|}\sum_{(x, y){\in}B}h(x;\theta)^{\top}\color{red}{\left[\nabla_{h}^{2} \ell_{CE}\frac{h(x;\theta+\delta{v})-h(x;\theta-{\delta}v)}{2\delta}\right]}$$

With small fixed $\delta=0.01$. The red part is not backpropagated into. One HVP computation costs 3 forward propagations + 1 backward propagation.

The finite differences trick is similar to HF approach of Martens (2010)[^3]

## Inverse Hessian-vector product

The estimation of iHVP $u=(H+\lambda)^{-1}v$ involves iterating over updates
$$u^t= u^{t-1}-\eta(\tilde{H}_tu^{t-1}+{\lambda}u^{t-1}-v),{\qquad}t=1,\dots,T,$$
where each $\tilde{H}_tu^{t-1}$ computes in-batch HVP with batch sampled independently for each step.

Our paper [TBD] explains the choice of hyperparameters for this procedure $\eta$-learning rate, $T$- number of steps, $|B|$ -batch size.

Some examples below for $\lambda = 5.0$

| model | size | data | eta | T | batch size[^4] |
|---|---|---|---|---|---|
|ResNet18| 11M | ImageNet | 0.003 | 150 | 100 |
|ResNet50| 25M | ImageNet | 0.002 | 200 | 5 |
|OPT|1.3B| OpenWebText | 0.001 | 500 | 30 |
|Llama-1 |7B| OpenWebText | 0.0005 | 1000 | 50 |
|Mistral|7B| OpenWebText | 0.0002 | 2000 | 200 |

[^1]: http://proceedings.mlr.press/v70/koh17a/koh17a.pdf

[^2]: https://www.jmlr.org/papers/volume21/17-678/17-678.pdf

[^3]: http://www.cs.toronto.edu/~asamir/cifar/HFO_James.pdf

[^4]: For language models the batch size is counted in tokens, i.e. one sequence is sufficient.

# Instructions to start

A brief instruction to start up with using this code.

## Preparing models and data

- **Set up environment variable:** `STORAGE_PATH` denotes path to your storage for models and datasets to be save. Use `export STORAGE_PATH="path_to_your_storage"`, or set environment variable by other means. If not specified saves to ".", the models will be downloaded to `STORAGE_PATH/huggingface/models and the datasets` to `STORAGE_PATH/huggingface/datasets`.
- **Download models and datasets:** when running from a developer machine use ```python3 prepare/download_models_and_datasets.py```
- **Models:**  supporting GPT2, OPT, GPT-J, Llama-1, Mistral 7B. Add new huggingface models in ```networks.yaml```. For distributed, new models have to be added to ```tp_utils/hf_models.py```
- **Datasets:** Add new datasets to `datasets.yaml` and add a corresponding function to `dataset.py`. Use `python3 prapare/tokenize_webtext.py` to tokenizer the web-text dataset in advance. Refer to resulting dataset as 'web-text-tok-opt', 'web-text-tok-mistral', see `datasets.yaml` for details.

## Retrieval of top influencing functions

Here is an example how to run retrieval of top influencing training sequences for given prompt-completion pairs. The training sequences are extracted from WebText corpus. We extract top 10000 candidates based on similarity of BERT embeddings, which we need to precompute for the whole webtext.

- **Tokenizing dataset** typically takes a while to tokenize webtext, so we do it separately in `prepare/tokenize_web_text.py --network=...`
- **Additional preparation** run `prapare/embed_webtext.py` to precalcualte the BERT embeddings for candidate retrieval
- **Run retrieval** run `scripts/run_retrieval.py` specify path to a yaml config containing prompts and completions, example in `input_data/retrieval_test.yaml`. Other options for see in `config_lissa_steps.yaml`, including learning rate, batch size, etc. Tensor parallezation is done automatically when more than one GPU is detected. E.g. for Mistral-7B we recommend to use 4 A100/H800 with batch size 4.
