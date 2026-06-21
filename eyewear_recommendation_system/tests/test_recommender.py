from eyewear_system.recommender.recommender_node import RecommenderNode

def test_recommender_returns_sentence_for_valid_input():
    features = {
        "face_shape": "round",
        "eye_shape": "almond",
        "eye_color": "brown",
        "pupil_distance": 0.46,
    }

    result = RecommenderNode().recommend(features)

    assert result["success"] is True
    assert result["warning"] is None

    assert result["face_shape"] == "round"
    assert result["eye_shape"] == "almond"
    assert result["eye_color"] == "brown"
    assert result["pupil_distance"] == 0.46
    assert result["eye_distance"] == "average"

    assert "rectangular" in result["recommended_shapes"]
    assert "wayfarer" in result["recommended_shapes"]
    assert "tortoiseshell" in result["recommended_colors"]
    assert "black" in result["recommended_colors"]

    assert "sentence" in result
    assert "For a round face with almond brown eyes" in result["sentence"]
    assert "average eyes" in result["sentence"]


def test_recommender_converts_close_set_pupil_distance():
    features = {
        "face_shape": "oval",
        "eye_shape": "almond",
        "eye_color": "blue",
        "pupil_distance": 0.40,
    }

    result = RecommenderNode().recommend(features)

    assert result["success"] is True
    assert result["eye_distance"] == "close_set"
    assert "narrow bridge" in result["eye_distance_modifier"]


def test_recommender_converts_wide_set_pupil_distance():
    features = {
        "face_shape": "square",
        "eye_shape": "round",
        "eye_color": "green",
        "pupil_distance": 0.53,
    }

    result = RecommenderNode().recommend(features)

    assert result["success"] is True
    assert result["eye_distance"] == "wide_set"
    assert "wider bridge" in result["eye_distance_modifier"]


def test_recommender_returns_warning_for_missing_input():
    features = {
        "face_shape": "round",
        "eye_shape": "almond",
        "eye_color": "brown",
        # Missing pupil_distance
    }

    result = RecommenderNode().recommend(features)

    assert result["success"] is False
    assert result["sentence"] is None
    assert result["warning"] == (
        "Recommendation could not be generated because one or more required "
        "input features are missing or invalid."
    )


def test_recommender_returns_warning_for_invalid_pupil_distance():
    features = {
        "face_shape": "round",
        "eye_shape": "almond",
        "eye_color": "brown",
        "pupil_distance": "not_a_number",
    }

    result = RecommenderNode().recommend(features)

    assert result["success"] is False
    assert result["sentence"] is None
    assert result["warning"] == (
        "Recommendation could not be generated because one or more required "
        "input features are missing or invalid."
    )


def test_recommend_sentence_returns_only_sentence():
    features = {
        "face_shape": "heart",
        "eye_shape": "hooded",
        "eye_color": "hazel",
        "pupil_distance": 0.46,
    }

    sentence = RecommenderNode().recommend_sentence(features)

    assert isinstance(sentence, str)
    assert "For a heart face with hooded hazel eyes" in sentence
    assert "standard bridge" in sentence