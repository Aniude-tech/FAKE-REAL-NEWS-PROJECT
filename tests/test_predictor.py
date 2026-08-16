from src.inference.predictor import predict_news


def test_prediction_returns_valid_result():
    text = """
    The government announced a new policy today.
    Officials said the policy will take effect next month.
    """

    result = predict_news(text)

    assert isinstance(result, dict)
    assert result["prediction"] in ["Real", "Fake"]
    assert result["label"] in [0, 1]
    assert isinstance(result["decision_score"], float)


def test_prediction_rejects_empty_text():
    try:
        predict_news("")
        assert False, "Expected ValueError"
    except ValueError:
        pass