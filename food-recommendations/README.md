# Food Recommendations Project Scope

## Objective
The aim of the project is to build (and deploy) a useful recommender system for food recipes using what I've learnt from the TFRS tutorials. I'm also using this as an excuse to learn about ML software deployment, AWS, and hopefully a bit about front-end development.

Why food recipes?
1. I like food and cooking, as do many other people. If my project is around something I personally find useful and or am interest in, then I'm more likely to finish it.
2. Unlike most other open source recommandation datasets around popular media films, books etc (see [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html#bartering_data)) preferences for food recipes should be relatively stable across time. This means a recommender I build now based on past data should still produce relevant recommendations.


## User Stories
A user should be able to:
* Provide a few examples of recipes they like and allow the system to learn their preferences
* Be able to put constraints on search/recommendation results

More advanced features:
* Be able to search through a database of recipes by keyword
* Give feedback on recipes they get recommended, which inform future recommendations
* Be shown pictures of recipes, possibly AI generated
* Generate new recipes from scratch based on user preferences (see [here](https://aclanthology.org/D19-1613.pdf))

## Data
The main data source used will be the [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download) dataset.

To avoid having to do really extensive EDA, so I can focus on the implementation of the model and deployment, I'm going to get a lot of my understanding of this dataset from borrow [this](https://www.kaggle.com/code/etsc9287/food-com-eda-and-text-analysis) well rated (bronze) Kaggle notebook focusing on EDA: 

## Performance Expectations
A few notes on performance expectations:
* I'm expecting the model to return recommendations fast (sub 5 seconds).
* Ideally the end-system should be able to scale to a large number of users.
* Recommendation requests are expected to come in at up to a few per day.

## Project Phasing
I plan to break the project down into a few phases.

1. Build the simplest recommender system possible and benchmark performance against a validation set. This doesn't even have to be an ML model. Aim of this is to start getting a feel for the data available and what metrics are sensible.
2. Build a simple front-end to demo the model.
3. Build a slightly more complex ML recommender system and compare performance against the simple model.
4. Deploy the simple ML recommender backend with a minimal API
5. Build a nicer front-end to serve the model
6. Iterate and tackle the more advanced features
