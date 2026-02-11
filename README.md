## Goal of EvalME


Thousands of ML models are produced every year. All have different architectures, training paradigms, data modalities, and configurations. 

However, the end result (after training) for every ML model practitioner is a set of weights and the task of deciding how to evaluate the produced model.

Oftentimes, this is not as simple as producing an accuracy score on a given dataset. Depending on the audience, different types of analyses may be required such as layer wise understanding, weight distributions, visualization plots, etc.

These types of analyses are often written on a project specific basis and we usually only write enough evaluation for our given use cases. This is fine for our personal projects, but when the scope of our problem changes or we start to look at models from different sources, it becomes a headache to have to redefine our evaluation process constantly to accomodate the new model setting.

Wouldn't it be better to specify a model and a test set and then immediately output all the visualizations, plots, and descriptive statistics that we desire?

Wouldn't it be better if we could just alter a configuration file and select the outputs and their associated settings?


## Maintenance Goals

This repo will undergo constant maintenance as it expands to include a wide variety of functionalities. This means the code will oftentimes be in a state of partial completion. 

## Current Goal


Complete PyTorch style pre-trained model section.
