import streamlit as st

from pathlib import Path

from utils import recipe_preprocessing_pipeline, ContentBasedRecommender, SessionStateManager, INFO_COLS


MIN_INTERACTIONS = 3

st.title("Recipe Recommendations :shallow_pan_of_food:")
st.markdown("A prototype app to show recipe recommendations based on recipes you like!")
st.markdown("TODO:\n* Add recipe sampling\n* Exclude seen items\n* Pick starting recipes carefully")

data_path = Path("../data/RAW_recipes.csv")
cached_path = Path("cached_features.csv")
if cached_path.exists():
    feature_df = recipe_preprocessing_pipeline(data_path, cached_path, True)
else:
    feature_df = recipe_preprocessing_pipeline(data_path, cached_path, False)

state_manager = SessionStateManager()

# create recommender and fit
recommender = ContentBasedRecommender(
    feature_df=feature_df,
    max_recommendations=100,
    model_save_path="knn_model.pkl"
)

# train knn model only on the first page load of the session
if state_manager.is_startup():
    st.spinner("Getting recipes ready...")
    recommender.fit()

# get next items
item = state_manager.next_item()

while not item:

    # a fresh user is given empty interactions, otherwise use the list of liked interactions
    positive_interactions_ids = state_manager.positive_interactions()
    st.info(positive_interactions_ids)
    if state_manager.empty_queue():
        if len(positive_interactions_ids) < MIN_INTERACTIONS:
            st.info("Getting random recommendations")
            items = recommender.get_random_recommendations(10)
        else:
            st.info("Getting informed recommendations")
            items = recommender.get_recommendations(positive_interactions_ids)
        state_manager.enqueue_items(items)

    item = state_manager.next_item()

# place buttons top of the page
col1, col2 = st.columns(2)

# display items
item_info = feature_df.loc[item][INFO_COLS].to_dict()
st.write(item_info)

with col1:
    if st.button(":thumbsup:", use_container_width=True):
        state_manager.like_item(item)
with col2:
    if st.button(":thumbsdown:", use_container_width=True):
        state_manager.dislike_item(item)
