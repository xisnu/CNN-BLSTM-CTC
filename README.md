# A Hybrid Deep Architecture for Robust Recognition of Text Lines of Degraded Printed Documents
---
A Tensorflow implementation of a CNN-BLSTM-CTC architecture used for *Degraded OCR Recognition*. This work is submitted under the title __"A Hybrid Deep Architecture for Robust Recognition of Text Lines of Degraded Printed Documents"__ in the *124th International Conference on Pattern Recognition*.

Requirements
---
1. Python 2.7
2. Tensorflow 0.12
3. H5Py

## Usage Instruction
- Install __*virtualenv*__ for *Python 2.7* (optional but highly recommended)
- Install __Tensorflow 0.12__ using *pip*. Instructions can be found [here](https://www.tensorflow.org/versions/r0.12/get_started/os_setup).
- Install __*h5py*__ using *pip*.
- Run *CNN-BLSTM-CTC.py* with argument *New*. This will create a network with randomly initialized weight variables.
## Network Description
The network consists of CNN layers followed by BLSTM layers. The loss is computed using CTC loss model. Follow the [diagram](https://github.com/xisnu/CNN-BLSTM-CTC/blob/master/Images/hybrid.jpg)
## Dataset Description
For demonstration of model a simple dataset __ICBOHR-04_Gist_feat__ is given in HDF5 format. This contains extracted features of only 540 samples of online handwritten Bangla words. No separate test file is given. Executing the same model for training with this dataset results a __91.5034%__ accuracy measured as per edit distance after 500 epochs. 

The implementaion is mostly inspired from the work of @igormq as given [here](https://github.com/igormq/ctc_tensorflow_example/blob/master/ctc_tensorflow_example.py).
