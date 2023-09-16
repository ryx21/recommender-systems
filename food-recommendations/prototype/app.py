import streamlit as st

from pathlib import Path

from utils import recipe_preprocessing_pipeline, ContentBasedRecommender, SessionStateManager, INFO_COLS


MIN_INTERACTIONS = 3

st.title("Recipe Recommendations :shallow_pan_of_food:")
st.markdown("A prototype app to show recipe recommendations based on recipes you like!")
st.markdown("TODO:\n* Add recipe sampling\n* Exclude seen items\n* Pick starting recipes carefully")

feature_df = recipe_preprocessing_pipeline(Path("../data/RAW_recipes.csv"))

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

# a fresh user is given empty interactions, otherwise use the list of liked interactions
positive_interactions_ids = state_manager.positive_interactions()
if state_manager.empty_queue():
    if len(positive_interactions_ids) == 0:
        items = recommender.get_random_recommendations(10)
        print(items)
    else:
        items = recommender.get_recommendations(positive_interactions_ids)
        print(items)
    state_manager.enqueue_items(items)

# place buttons top of the page
col1, col2 = st.columns(2)

# display items
item = state_manager.next_item()
item_info = feature_df.loc[item][INFO_COLS].to_dict()
st.write(item_info)

with col1:
    if st.button(":thumbsup:", use_container_width=True):
        state_manager.like_item(item)
with col2:
    if st.button(":thumbsdown:", use_container_width=True):
        state_manager.dislike_item(item)
