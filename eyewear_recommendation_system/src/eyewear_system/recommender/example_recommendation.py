from eyewear_system.recommender.recommender_node import RecommenderNode


def main():
    # TODO_REPLACE_LATER:
    # These values should later come from the trained neural networks.
    detected_features = {
        "face_shape": "oval",
        "eye_shape": "round",
        "eye_color": "blue",
        "pupil_distance": 0.46,
    }

    recommender = RecommenderNode()
    result = recommender.recommend(detected_features)

    if result["success"]:
        print(result["sentence"])
    else:
        print(result["warning"])


if __name__ == "__main__":
    main()