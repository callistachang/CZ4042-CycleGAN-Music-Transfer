# Music Style Transfer With CycleGANs

Built with Tensorflow 2.3.1. 

Special thanks to [sumuzhao](https://github.com/sumuzhao) and the paper [Symbolic Music Genre Transfer with CycleGAN](https://arxiv.org/pdf/1809.07575.pdf) for providing the resources and inspiration required to complete this project for NTU's CZ4042 Neural Networks and Deep Learning module.

## Major Changes

* [sumuzhao's implementation](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization) did not run for 2 reasons:
    1. The usage of **lambda layers**, which was unsupported for the implementation of instance normalization layers and residual blocks.
        - New classes `InstanceNormalization` and `ResNetBlock` extending from `keras.layers.Layer` were created to replace them.
        - I raised a [GitHub issue](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization/issues/3) about this error on the original code repository. If the author approves, I will contribute a pull request with the above changes onto the repository. (Update: The author approved my pull request [here](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization/pull/4)! :smile:)
    2. Minor bugs in parsing command line arguments. 
        - Made a few alterations to the command line parsing logic. 

* `SGD` and `RMSprop` were added as optimizer choices.

* During CycleGAN training, discriminator, generator and cycle losses and accuracies over epochs are pickled for later examination.

* During classifier training, test losses and accuracies over epochs are pickled for later examination.

* During classifier testing, the test accuracies on the origin, cycle and transfer datasets are sorted and outputted to a CSV file for further examination.

## Additional Scripts

* `/notebooks/visualization.ipynb` was created to visualize pickled files containing the losses and accuracies over epochs during CycleGAN and classifier training.

* `/notebooks/tuning.ipynb` was created to tune the hyperparameters for the CycleGAN and classifier training. The tuned hyperparameters are as follows:
    1. Standard deviation of Gaussian noise (`sigma_d`)
    2. Number of filters in convolutional layers (`ndf` and `ngf`)
    3. Optimizer choice (`optimizer`)
    4. Optimizer momentum term (`beta1`)
    5. Optimizer learning rate (`lr`)

* `/scripts/classify.py` was created to test the classifier on a specified directory containing `.npy` music arrays.

* `/scripts/tomidi.py` was created to convert a `.npy` music array to a `.mid` file.

## Usage

```bash
# Train CycleGAN model
python main.py --dataset_A_dir=JC_J --dataset_B_dir=JC_C --phase=train --type=cyclegan --sigma_d=0

# Generate origin, cycle and transfer outputs with the trained CycleGAN model
python main.py --dataset_A_dir=JC_J --dataset_B_dir=JC_C --phase=test --type=cyclegan --sigma_d=0

# Train classifier model
python main.py --dataset_A_dir=JC_J --dataset_B_dir=JC_C --phase=train --type=classifier --sigma_c=0

# Test classifier model on origin, cycle and transfer outputs
python main.py --dataset_A_dir=JC_J --dataset_B_dir=JC_C --phase=test --type=classifier --sigma_c=0

# Test classifier model on a specified directory containing .npy arrays
python scripts/classify.py --classify_dir=JC_J/test

# Convert a .npy array to a MIDI file
python scripts/tomidi.py --npy_filepath=JC_J/test/jazz_piano_test_1.npy
```

## Datasets

The jazz, classical and pop datasets can be downloaded from the zip file [here](https://drive.google.com/file/d/1zyN4IEM8LbDHIMSwoiwB6wRSgFyz7MEH/view).
