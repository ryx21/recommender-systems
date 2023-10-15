from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from typing import List

import streamlit as st
import numpy as np
import pandas as pd

import ast
import pickle
import random


INFO_COLS = [
    "name", "submitted", "tags", "n_steps",
    "calories", "total_fat", "sugar", "sodium",
    "protein", "saturated_fat", "carbohydrates",
    "description", "ingredients", "minutes", "steps"
]


def _transform_numerical_columns(df, columns):
    numerical_pipeline = make_pipeline(
        FunctionTransformer(lambda x: np.sign(x) * np.log(np.abs(x)+1)),
        StandardScaler()
    ).set_output(transform="pandas")
    df_numerical = numerical_pipeline.fit_transform(df[columns])
    df_numerical = df_numerical.rename(columns={col: f"F_{col}" for col in df.columns})
    return df_numerical


def _transform_text_columns(df, columns, n_features=20) -> pd.DataFrame:
    results = []
    for col in columns:
        vectorizer = HashingVectorizer(n_features=n_features)
        X = vectorizer.fit_transform(df[col].apply(lambda x: ' '.join(x)))
        column_headers = [f"F_{col}_hash_{i}" for i in range(n_features)]
        results.append(pd.DataFrame(X.todense(), columns=column_headers))
    return pd.concat(results, axis=1)


@st.cache_data
def recipe_preprocessing_pipeline(data: Path, cached_path: Path, read_only: bool=False):

    if read_only:
        df = pd.read_csv(cached_path, index_col="id")
        return df

    df = pd.read_csv(data)
    for col in ["tags", "nutrition", "steps", "ingredients"]:
        df[col] = df[col].apply(ast.literal_eval)

    # handle nans
    df["description"] = df["description"].fillna("")
    df = df.dropna(subset=["name"])

    # Convert nutrition info to individual columns
    NUTRITION_COLS = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    df[NUTRITION_COLS] = df["nutrition"].tolist()

    # Preprocess numerical features
    NUMERICAL_FEATURES = ["minutes", "n_steps", "n_ingredients"] + NUTRITION_COLS
    TEXT_FEATURES = ["ingredients"]

    df_numerical = _transform_numerical_columns(df, NUMERICAL_FEATURES)
    df_text = _transform_text_columns(df, columns=TEXT_FEATURES)
    df = pd.concat([df, df_numerical, df_text], axis=1)

    # TODO: the concat create a single nan row which casts all the int cols -> floats
    df = df.dropna()
    df["id"] = df["id"].apply(int)
    df = df.set_index("id")

    df.to_csv(cached_path)
    return df


class ContentBasedRecommender:

    def __init__(self,
        feature_df: pd.DataFrame,
        max_recommendations: int,
        model_save_path: Path,
    ):
        self.feature_df = feature_df
        self.max_recommendations = max_recommendations

        # feature cols must be prefixed with "F_"
        self.feature_cols = [col for col in feature_df if col.startswith("F_")]
        self.model_save_path = model_save_path

        # makes looking up features easier

    def fit(self):
        neigh = NearestNeighbors(n_neighbors=self.max_recommendations)
        neigh.fit(self.feature_df[self.feature_cols])
        pickle.dump(neigh, open(self.model_save_path, 'wb'))
        print(f"Saving classifier to {str(self.model_save_path)}")

    def _verify_ids(self, ids: List[str]):
        not_exist_ids = []
        for i in ids:
            if i not in self.feature_df.index:
                not_exist_ids.append(i)
        assert len(not_exist_ids) == 0, f"IDs {not_exist_ids} are not found in the item index"

    def get_random_recommendations(self, n_recommendations=10):
        return random.choices(self.feature_df.index, k=n_recommendations)

    def get_recommendations(self, input_ids: List[int], n_recommendations=10):

        assert n_recommendations <= self.max_recommendations, f"n_recommendations={n_recommendations} must be \
            less than max_recommendations={self.max_recommendations}"
        self._verify_ids(input_ids)
        
        with open(self.model_save_path, "rb") as f:
            model = pickle.load(f)

        embedding = self.feature_df.loc[input_ids][self.feature_cols].mean()

        # embedding needs to be reshaped to (1, n_features)
        embedding = embedding.to_numpy().reshape(1, -1)
        recommendations_idx = model.kneighbors(embedding, n_recommendations, return_distance=False)

        # recommendations will be a (1, n_recommendations) shape array
        recommended_items_idx = list(self.feature_df.iloc[recommendations_idx.flatten()].index)
        return recommended_items_idx


ITEM_QUEUED_KEY = "recipes_queued"
ITEM_INTERACTIONS_KEY = "recipes_seen"
STARTUP = "startup"

class SessionStateManager:

    def __init__(self):

        if ITEM_QUEUED_KEY not in st.session_state:
            st.session_state[ITEM_QUEUED_KEY] = []
        
        if ITEM_INTERACTIONS_KEY not in st.session_state:
            st.session_state[ITEM_INTERACTIONS_KEY] = {}

    def next_item(self, exclude_seen=True):
        if self.empty_queue():
            return None
        else:
            item = st.session_state[ITEM_QUEUED_KEY].pop()
            if exclude_seen and item in st.session_state[ITEM_INTERACTIONS_KEY]:
                return self.next_item(True)
            else:
                return item

    def is_startup(self):
        "Return True if first session, else false"
        if STARTUP not in st.session_state:
            st.session_state[STARTUP] = True
            return True
        else:
            return False

    def dislike_item(self, item_id: int):
        st.session_state[ITEM_INTERACTIONS_KEY][item_id] = False

    def like_item(self, item_id: int):
        st.session_state[ITEM_INTERACTIONS_KEY][item_id] = True

    def enqueue_items(self, item_id_list: List[int]):
        # assume items are given in descending order of recommendation
        # reverse list so most relevant items are popped off first
        # we could use a queue data structure to make this a bit more efficient
        st.session_state[ITEM_QUEUED_KEY] = item_id_list[::-1] + st.session_state[ITEM_QUEUED_KEY]

    def empty_queue(self):
        if not st.session_state[ITEM_QUEUED_KEY]:
            return True
        else:
            return False

    def positive_interactions(self):
        result = []
        for item_id, liked in st.session_state[ITEM_INTERACTIONS_KEY].items():
            if liked:
                result.append(item_id)
        return result
