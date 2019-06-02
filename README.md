# Switching from Adam to SGD

*[Wilson et al. (2018)](https://arxiv.org/pdf/1705.08292.pdf)* shows that "*the solutions found by adaptive methods generalize worse (often significantly worse) than SGD, even when these solutions have better training performance. These results suggest that practitioners should reconsider the use of adaptive methods to train neural networks.*"

"*SWATS from [Keskar & Socher (2017)](https://arxiv.org/pdf/1712.07628.pdf) a high-scoring paper by ICLR in 2018, a method proposed to automatically switch from Adam to SGD for better generalization performance. The idea of the algorithm itself is very simple. It uses Adam, which works well despite minimal tuning, but after learning until a certain stage, it is taken over by SGD.*"

## Usage

Installing the package is straightforward with pip directly from this git repository or from pypi with either of the following commands.

```bash
pip install git+https://github.com/Mrpatekful/swats
```

```bash
pip install pytorch-swats
```

After installation *SWATS* can be used as any other `torch.optim.Optimizer`. The following code snippet serves as a simple overview of how to use the algorithm.

```python
import swats

optimizer = swats.SWATS(model.parameters())
data_loader = torch.utils.data.DataLoader(...)

for epoch in range(10):
    for inputs, targets in data_loader:
        # deleting the stored grad values
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # performing parameter update
        optimizer.step()
```
