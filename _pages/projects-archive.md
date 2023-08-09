---
layout: collection
title: "Projects"
collection: projects
permalink: /projects/
author_profile: true
entries_layout: grid
classes: wide
header:
    overlay_image: /assets/images/joel-filipe-small-warmer.jpg
---
## Mini-Projects
[Go to MiniProjects](./miniprojects/){: .btn .btn--primary .btn--small}
## Projects
- ### Semi Supervised Learning - FixMatch 
  **Summary:** Implemented Fix Match SSL Algorithm on CIFAR-10 database and run ablation experiments on the essential components of Fix Match. Fix Match algorithm supplements limited labelled data by augmenting the images to produce additional labeled data. Was able to get an accuracy of 94.67% on using only 250 labeled examples on a WideResNet CNN Model.<br />
  **Tools Used:** Python, Pytorch, Numpy, Tensorboard.<br />
  **Hardware Used:** Google Colab (A100 GPU).<br />
  [Github](https://github.com/vimvenu-rgb/Fix_Mix_Match-Project){: .btn .btn--primary .btn--small}
- ### Reinforcement Learning - Google Football
  **Summary:** Configured a multi-agent Asynchronous Advantage Actor-Critic (A3C) algorithm, train it in a reduced Google football multi-agent environment and evaluate the trained agent against three baseline agents.  A3C algorithm was chosen to solve the multi-agent MDP problem as A3C inherently supports concurrent training of multi-agent policies and the Advantage function reduces high variance during the policy gradient update.<br />
  **Tools Used:** Docker, Ray RLib, Tensorboard, Pytorch.<br />
  **Hardware Used:** Google Cloud Platform (GCP).<br />
  [Github](https://github.com/vimvenu-rgb/A3C_RL_Project){: .btn .btn--primary .btn--small}
- ### Scikit Learn Projects
  **Summary:** Executed various algoirhtms to predict the target variable based on a set of input features. The following classification algoritms are used for comparision:
  - Decision Tree
  - Decision Tree + Adaboost
  - Artificial Neural Network (MLPClassifier)
  - SVM
  - KNN.<br />
  
  **Tools Used:** Jupyter Notebook, Scikit Learn, Numpy & Pandas.<br />
  **Hardware Used:** Google Colab.<br />
  [Github](https://github.com/vimvenu-rgb/ScikitLearn_Projects){: .btn .btn--primary .btn--small}