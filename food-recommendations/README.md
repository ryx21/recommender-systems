# Food Recommendations Project

## Project Scope

### Objective
The aim of the project is to build (and deploy) a useful recommender system for food recipes using what I've learnt from the TFRS tutorials. I'm also using this as an excuse to learn about ML software deployment, AWS, and hopefully a bit about front-end development.

Why food recipes?
1. I like food and cooking, as do many other people. If my project is around something I personally find useful and or am interest in, then I'm more likely to finish it.
2. Unlike most other open source recommandation datasets around popular media films, books etc (see [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html#bartering_data)) preferences for food recipes should be relatively stable across time. This means a recommender I build now based on past data should still produce relevant recommendations.

### User Stories
A user should be able to:
* Provide a few examples of recipes they like and allow the system to learn their preferences
* Be able to put constraints on search/recommendation results

More advanced features:
* Be able to search through a database of recipes by keyword
* Give feedback on recipes they get recommended, which inform future recommendations
* Be shown pictures of recipes, possibly AI generated
* Generate new recipes from scratch based on user preferences (see [here](https://aclanthology.org/D19-1613.pdf))

### Data
The main data source used will be the [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download) dataset.

To avoid having to do really extensive EDA, so I can focus on the implementation of the model and deployment, I'm going to get a lot of my understanding of this dataset from borrow [this](https://www.kaggle.com/code/etsc9287/food-com-eda-and-text-analysis) well rated (bronze) Kaggle notebook focusing on EDA: 

### Performance Expectations
A few notes on performance expectations:
* I'm expecting the model to return recommendations fast (sub 5 seconds).
* Ideally the end-system should be able to scale to a large number of users.
* Recommendation requests are expected to come in at up to a few per day.

### Project Phasing
I plan to break the project down into a few phases.

1. Build the simplest recommender system possible and benchmark performance against a validation set. This doesn't even have to be an ML model. Aim of this is to start getting a feel for the data available and what metrics are sensible.
2. Build a slightly more complex ML recommender system and compare performance against the simple model.
3. Deploy one of the recommenders with a minimal API
4. Build a nicer front-end to serve the model(s)
5. Iterate and tackle the more advanced features

## Project Progress

### 1. Simple Recommender System

See [01_simple_recommender.ipynb](./01_simple_recommender.ipynb)

### 2. Neural Recommender System

See [02_neural_recommender.ipynb](./02_neural_recommender.ipynb)

Resources:
* https://www.tensorflow.org/guide/data
* https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
* https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
* https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed
* https://www.tensorflow.org/recommenders/examples/sequential_retrieval
* https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
* https://www.tensorflow.org/guide/keras/understanding_masking_and_padding

Todo:
* Remind myself how attention works


### 3. Front End

After reading [this](https://www.bighuman.com/blog/backend-frontend-web-development-where-do-you-start) article, I decide it's easier to build the front-end first before the backend.

First we'll build a simple end-to-end prototype using `streamlit`. This can be spun up using:

```
cd prototype
streamlit run app.py
```

Resources:
* https://react.dev/learn
* https://huggingface.co/inference-api
* https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
* https://huggingface.co/docs/api-inference/detailed_parameters


### 4. Simple Deployment

I need to learn how to build an API. These are my first pass at naive requirements.

API Design Notes:
1. An endpoint which returns a list of N recommended recipe ids when provided with a user or session id
    * Looks up in a database the list of recipes the user has previously liked
    * Prepares the model inputs based on this, and returns a list of N liked recipes
    * Optionally: exclude ids a user has already interacted with
    * Optionally: sample recipes from a distribution to achieve some randomness
2. An endpoint which fetches all the details of a recipe to be displayed to the user when provided a recipe id

Resources:
* https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design
* https://masteringbackend.com/posts/api-design-best-practices
* https://aws.amazon.com/getting-started/hands-on/build-web-app-s3-lambda-api-gateway-dynamodb/
* https://aws.amazon.com/getting-started/hands-on/?intClick=gsrc_navbar&getting-started-all.sort-by=item.additionalFields.content-latest-publish-date&getting-started-all.sort-order=desc&awsf.getting-started-category=*all&awsf.getting-started-content-type=*all
* https://aws.amazon.com/blogs/machine-learning/how-to-deploy-deep-learning-models-with-aws-lambda-and-tensorflow/
* https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/
* https://www.youtube.com/watch?v=0Sh9OySCyb4&ab_channel=BeABetterDev
* https://www.youtube.com/watch?v=bYkjYojgccY&ab_channel=BeABetterDev
* https://towardsdatascience.com/serverless-deployment-of-machine-learning-models-on-aws-lambda-5bd1ca9b5c42
    * https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/
