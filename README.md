# Switching from Adam to SGD

"*SWATS from [Keskar & Socher et al. (2017)](https://arxiv.org/pdf/1712.07628.pdf) a high-scoring paper by ICLR in 2018, a method proposed to automatically switch from Adam to SGD for better generalization performance.*"

"*The idea of the algorithm itself is very simple. It uses adam, which works well despite minimal tuning, but after learning until a certain stage, it is taken over by SGD.*"

## Usage

Installing the package is fairly straightforward with pip directly from this git repository with the following command.

```bash
pip install git+https://github.com/Mrpatekful/swats
```

After installation *SWATS* can be used as any other PyTorch `Optimizer`. The following code snippet serves as a simple example to use the algorithm.  For more examples, see this [gist]() with benchmarks and comparison of *SWATS* with other optimizers.

```python
optimizer = torch.optim.SWATS(model.parameters())
data_loader = torch.utils.data.DataLoader(...)
for epoch in range(10):
    for batch in data_loader:
        optimizer.zero_grad()
        train_batch(...)
        optimizer.step()
```