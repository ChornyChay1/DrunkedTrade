from unittest.mock import patch, AsyncMock

import pytest


# =====================================================
# CREATE
# =====================================================

def test_create_indicator(client, sample_indicator, mock_db_session):
    # мок recalc_indicator
    with patch(
        "api.indicator_routes.recalc_indicator",
        new_callable=AsyncMock
    ) as mock_recalc:
        mock_recalc.return_value = None

        resp = client.post("/indicator", json=sample_indicator)

    assert resp.status_code == 200
    data = resp.json()

    assert "id" in data
    assert isinstance(data["id"], int)

    mock_recalc.assert_called_once()