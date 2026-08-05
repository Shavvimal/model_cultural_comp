import pytest
from pydantic import ValidationError

from app.qn_classes import A008, A165, F118, Y002, Y003


class TestScaleEnums:
    def test_valid_values(self):
        assert A008(2) == A008.QUITE_HAPPY
        assert A165(1) == A165.TRUST
        assert F118(10) == F118.TEN

    @pytest.mark.parametrize("cls,bad", [(A008, 5), (A165, 3), (F118, 0), (F118, 11)])
    def test_out_of_range_rejected(self, cls, bad):
        with pytest.raises(ValueError):
            cls(bad)


class TestY002:
    def test_valid_pair(self):
        parsed = Y002(most_important=1, second_most_important=4)
        assert parsed.most_important == 1

    def test_duplicate_choices_rejected(self):
        with pytest.raises(ValidationError, match="different"):
            Y002(most_important=2, second_most_important=2)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Y002(most_important=5, second_most_important=1)


class TestY003:
    def test_valid_choices(self):
        parsed = Y003(choices=[1, 2, 6, 8, 9])
        assert len(parsed.choices) == 5

    def test_more_than_five_rejected(self):
        with pytest.raises(ValidationError, match="five"):
            Y003(choices=[1, 2, 3, 4, 5, 6])

    def test_duplicates_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            Y003(choices=[1, 1, 2])

    def test_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Y003(choices=[12])
