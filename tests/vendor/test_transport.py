"""Public presentation-data immutability contract."""

import pytest

from slidesmith.engine import PresentationData


class TestPresentationData:
    """Tests for PresentationData dataclass."""

    def test_is_frozen(self) -> None:
        """PresentationData is immutable."""
        data = PresentationData(
            presentation_id="test123",
            data={"slides": []},
        )

        with pytest.raises(AttributeError):
            data.presentation_id = "changed"  # type: ignore[misc]
