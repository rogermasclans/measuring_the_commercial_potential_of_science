This repository contains the code to develop a measure for the commercial potential of science, following Masclans, Hasan, and Cohen, 2024. 

The repository includes code for training models and generating predictions within containerized environments. This approach is particularly beneficial for handling large datasets or training multiple models simultaneously.

For users who find container management challenging, we've also provided simplified notebook versions that can be easily used, for instance, via Google Colab.

A dataset with the commercial potential measures is available at https://zenodo.org/records/10815144

If you use the code or data, please cite Masclans, R., Hasan, S., & Cohen, W. M. (2024). Measuring the Commercial Potential of Science (No. w32262). National Bureau of Economic Research.

## Content
- `trainer-docker` and `predictions-docker`: Directories with code to train and get predictions within a containerized environment, using Docker.
- `revamped_abstracts_chatgpt_api.py`: Python code to make abstracts more commercially appealing using ChatGPT, via API. Used to assess the robustness of the model to the use of topical, commercially-oriented language.
- `science_compot_montecarlo_dropout.ipynb`: Notebook to run the Monte Carlo Drop Out Simulations. Used to cast several predictions for a random sample of 100,000 abstracts and assess the model's uncertainty.
- `science_compot_train_and_preds.ipynb`: Notebook (simplified) to train and get predictions on the commercial potential of scientific research. 
