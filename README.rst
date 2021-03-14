BERT fine-tuning for Twitter sentiment analysis
===============================================
.. image:: assets/wordcloud.png

The aim of this repo is to fine-tune a BERT model for Twitter sentiment analysis,
i.e. classification into negative, neutral and positive tweets.

The code is mainly built on top of
the BERT implementation with pretrained weights from the `Hugging Face <https://huggingface.co/>`_ library and
wrapped in
`pytorch-lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ for handling standard training and testing procedures.

Requirements
------------
- Python >= 3.8
- (CUDA >= 10.1 for GPU training)

Installation
------------
1. Clone repo with :code:`git clone https://github.com/APirchner/sentbert.git`
2. Optional but recommended: Set up a virtual environment with Python >= 3.8
3. Install requirements with e.g. :code:`pip install -r requirements.txt`


Data
----
The data is taken from the Twitter sentiment analysis
challenge `SemEval2017 Task 4 A <https://www.aclweb.org/anthology/S17-2088/>`_.

For reproducting the exact train/val/test split,
run the `data split notebook <notebooks/consolidate_data.ipynb>`_.
Also check out the `data exploration notebook <notebooks/explore_data.ipynb>`_ to get a feel
for the training dataset and for the cleanup steps that are part of the data pipeline for training
and testing SentBERT.

SentBERT
--------

Training
........
When tuning SentBERT, it essential to fight overfitting: The dataset is way too small for
training BERT from scratch. Fine-tuning it for too long or with a learning rate that is too
large allows BERT to easily overfit on the training data, leading to horrible test set performance.
Therefore, the learning rate on BERT should be set an order of magnitude lower than for the
freshly initialized classification head (see my configuration below).
An additional weight-decay term does not hurt either.

The test set performance below has been achieved with this training configuration:

.. code-block::

   python train.py --gpus <number of gpus to use> \
                   --max_epochs 3 \
                   --path_train <path/to/training/data> \
                   --path_val <path/to/validation/data> \
                   --log_dir <path/to/log/dir> \
                   --batch_size 32 \
                   --lr_bert 1e-5 \
                   --lr_class 2e-4 \
                   --weight_decay 1e-2 \
                   --workers <number of dataloader workers>



Test set performance
....................
With the configuration above, I got the following results on the test set (n=10,039)

========  ========
Accuracy  0.7699
F1         0.7605
========  ========


TorchServe endpoint
...................
work in progress