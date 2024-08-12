This repository contains the code to develop a measure for the commercial potential of science, following Masclans, Hasan, and Cohen, 2024. 

The repository includes code for training models and generating predictions within containerized environments. This approach is particularly beneficial for handling large datasets or training multiple models simultaneously.

For users who find container management challenging, we've also provided simplified notebook versions that can be easily used, for instance, via Google Colab.

## Content
- `trainer-docker` and `predictions-docker`: Directories with code to train and get predictions within a containerized environment, using Docker.
- `revamped_abstracts_chatgpt_api.py`: Python code to make abstracts more commercially appealing using ChatGPT API. Used to assess the robustness of the mdoel to the use of topical, commercially-oriented language.
- `science_compot_montecarlo_dropout.ipynb`: Notebook to run the Monte Carlo Drop Out Simulations. Used to cast several predictions for a random sample of 100,000 abstracts and assess the model's uncertainty.
- `science_compot_train_and_preds.ipynb`: Notebook (simplified) to train and get predictions on the commercial potential of scientific research. 
