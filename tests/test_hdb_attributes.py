import numpy as np
import pandas as pd

from src.research.hdb_attribute_features import (
    normal, prepare_resale, resale_features, supplied_current_macro,
)


def test_address_aliases_preserve_join_identity():
    assert normal("HOLLAND AVE") == normal("holland avenue")
    assert normal("BT BATOK ST 21") == normal("bukit batok street 21")


def test_sales_future_changes_do_not_enter_frozen_prediction():
    source = pd.DataFrame({
        "town": ["QUEENSTOWN"]*3, "block": ["8"]*3,
        "street_name": ["HOLLAND AVE"]*3, "flat_type": ["3 ROOM"]*3,
        "flat_model": ["Improved"]*3, "month": ["2024-10", "2024-12", "2025-01"],
        "floor_area_sqm": [59, 60, 70], "lease_commence_date": [1974]*3,
        "resale_price": [400000, 500000, 900000], "storey_range": ["01 TO 03"]*3,
    })
    query = pd.DataFrame({
        "TOWN": ["queenstown"]*2, "BLOCK": ["8"]*2,
        "STREET": ["holland avenue"]*2, "FLAT_TYPE": ["3-room"]*2,
        "FLOOR_AREA_SQM": [59, 59], "RENT_APPROVAL_DATE": ["2025-01", "2025-03"],
    })
    expected = resale_features(query, prepare_resale(source), origin="2025-01")
    source.loc[source.month.ge("2024-12"), "resale_price"] = 99999999
    changed = resale_features(query, prepare_resale(source), origin="2025-01")
    pd.testing.assert_frame_equal(expected, changed)
    assert expected.EXT_SALE_psm_median.eq(400000/59).all()
    assert expected.EXT_SALE_log_count.eq(np.log(2)).all()


def test_contemporaneous_and_origin_frozen_covariates_are_distinct():
    table = pd.DataFrame({"MACRO_X": [1, 2, 3, 4]},
                         index=pd.period_range("2024-12", periods=4, freq="M"))
    query = pd.DataFrame({"RENT_APPROVAL_DATE": ["2025-01", "2025-03"]})
    current = supplied_current_macro(table, query)
    frozen = supplied_current_macro(table, query, origin="2025-01")
    assert current.GIVEN_MACRO_X.tolist() == [2, 4]
    assert frozen.GIVEN_MACRO_X.tolist() == [1, 1]
