# Graph Pyramid Pooling Transformer for Graph-level Representation Learning

![GPPTans-viz](https://github.com/Frank-qlu/GPPTrans/blob/main/fig/GPPTrans.png)

### Python environment setup with Conda

```bash
conda create -n GPPTans python=3.9
conda activate GPPTans

pip install -r requirements.txt

```


### Running GPPTans
```bash
conda activate GPPTans

# Running demo
python main.py --cfg configs/GPPTans/mnist.yaml  wandb.use False
```
# inference demo
python inference.py --cfg configs/GPPTans/mnist.yaml

