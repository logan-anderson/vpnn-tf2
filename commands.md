running a demo
```bash
python run_demo_mnist.py --layers 4 --hidden_activation chebyshev\
        --name chebyshev2 --tensorboard --cheby_M 2.0\
         --rotations 10 --optimizer adam --epochs 100 --batch_size 128\
         --permutation_arrangement 2 --permutation_max_range 5
```

```bash
 python run_demo_mnist.py --layers 4 --hidden_activation chebyshev\
        --name mnistSpatialV --tensorboard --cheby_M 2.0\
         --rotations 10 --optimizer adam --epochs 1 --batch_size 128\
         --permutation_arrangement 3  --permutation_max_range 5 --use_dropout True
```
```bash
python run_demo_hyper.py --layers 2 --rotations 2 --use_dropout True --total_runs 28 --epochs 300 --permutation_arrangement 4
```