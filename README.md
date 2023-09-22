# Sarcasm Eval

The project was done as a requirement of a Machine Learning Coursework, for the semester of Summer 2023, in BRAC University.

The project includes implementation of two tasks described in [SemEval 2022 - iSarcasmEval: Intended Sarcasm Detection In English and Arabic](https://sites.google.com/view/semeval2022-isarcasmeval/home).

The two tasks are as follows:
* Given a dataset containing tweets from the social media, Twitter, identify if the text is sarcastic or not.
* Given a dataset containing tweets from the aforementioned outlet, identify the level of ironic speech it represents - sarcasm, irony, satire, understatement, overstatement and rhetoric question

For both the task, a machine learning model is required to be trained, which - given a sentence or a text - would be able to classify it with one of the desired class label.

The two tasks involves cleaning up the data first. The code for cleaning up the datasets can be found in the `data_cleanup.py` file of the project.

The workflow undertaken to train and evaluate the model for Task 1 can be found in `train_model_A.py` file. The process involves using Bidirectional LSTM Neural Network algorithm, along with other word vectorization and word embedding layers that forms the representation of the texts.

The second task's workflow can be found in `train_model_B.py` file, with the model following the same architecture as the previous task. However, the hyperparameters were tweaked to better suit the amount of data, as well as the six class labels it need to classify.

Pre-requisite:
* Python 3
* Pip
* (Optional) Jupyter Notebook - for running the notebooks

The following libraries were used:
* Pandas
* Numpy
* Sci-kit Learn
* Tensorflow
* Matplotlib
* [Emoji](https://pypi.org/project/emoji/)

The datasets used for the project can be found [here](https://github.com/iabufarha/iSarcasmEval), as well as in the SemEval 2022 website linked above.