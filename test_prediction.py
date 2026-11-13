"""Tests for the prediction path in stream.py and the pipeline in train.py."""

import math

import pandas as pd
import pytest

import stream
import train

SAMPLE = {
    "MSSubClass": [60],
    "MSZoning": ["RL"],
    "LotArea": [7000.0],
    "LotConfig": ["Inside"],
    "BldgType": ["1Fam"],
    "OverallCond": [5],
    "YearBuilt": [2000],
    "YearRemodAdd": [2010],
    "Exterior1st": ["VinylSd"],
    "BsmtFinSF2": [0.0],
    "TotalBsmtSF": [1000.0],
}


@pytest.fixture
def sample():
    return pd.DataFrame(SAMPLE)


def test_predict_returns_one_sensible_price(sample):
    (price,) = stream.predict_price(sample)
    assert math.isfinite(price)
    assert 0 < price < 10_000_000


def test_column_order_does_not_change_the_prediction(sample):
    shuffled = sample[list(reversed(sample.columns))]
    assert stream.predict_price(shuffled)[0] == pytest.approx(
        stream.predict_price(sample)[0]
    )


def test_unseen_category_does_not_crash(sample):
    # The encoder was fitted with handle_unknown="ignore", so a category it has
    # never seen must encode as all-zeros rather than raise.
    odd = sample.assign(Exterior1st=["NoSuchSiding"])
    assert math.isfinite(stream.predict_price(odd)[0])


def test_artifacts_are_cached():
    assert stream.load_artifacts() is stream.load_artifacts()


def test_saved_encoder_matches_what_train_builds():
    _, encoder = stream.load_artifacts()
    _, rebuilt = train.encode(train.load_data())
    assert list(rebuilt.get_feature_names_out()) == list(encoder.get_feature_names_out())


def test_model_and_encoder_agree_on_feature_count():
    model, encoder = stream.load_artifacts()
    numeric = len(SAMPLE) - len(encoder.feature_names_in_)
    assert len(model.feature_names_in_) == numeric + len(encoder.get_feature_names_out())
