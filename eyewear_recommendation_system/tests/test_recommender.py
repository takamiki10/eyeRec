from eyewear_system.recommender.recommender_node import RecommenderNode


def test_recommender_returns_exactly_three_recommendations():
    recommendations = RecommenderNode().recommend({})
    assert len(recommendations) == 3
    assert [item["rank"] for item in recommendations] == [1, 2, 3]
