from enum import IntEnum
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List

#################################################
############# Question Classes ################
#################################################


class A008(IntEnum):
    """
    [ 1.  2.  3.  4. nan]
    """
    VERY_HAPPY = 1
    QUITE_HAPPY = 2
    NOT_VERY_HAPPY = 3
    NOT_AT_ALL_HAPPY = 4


class A165(IntEnum):
    """
    [ 2.  1. nan]
    1: Most people can be trusted
    2: Can´t be too careful
    """
    TRUST = 1
    BE_CAREFUL = 2


class E018(IntEnum):
    """
    [ 1.  2.  3. nan]
    1: Good thing
    2: Don´t mind
    3: Bad thing
    """
    GOOD = 1
    DONT_MIND = 2
    BAD = 3


class E025(IntEnum):
    """
    [ 2.  1.  3. nan]
    1: Have done
    2: Might do
    3: Would never do
    """
    SIGNED = 1
    MIGHT_DO = 2
    NEVER = 3


# Range from 1 to 10
class F063(IntEnum):
    """
    [ 7.  1.  8.  4.  3.  5. 10.  6.  2.  9. nan]
    """
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10


class F118(IntEnum):
    """
    [ 4.  9. 10.  6.  8.  7.  1.  5.  2.  3. nan]
    """
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10


class F120(IntEnum):
    """
    [ 2.  9.  5.  4.  1. 10.  6.  8.  7.  3. nan]
    """
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10


class G006(IntEnum):
    """
    [nan  1.  3.  2.  4.]
    1: Very proud
    2: Quite proud
    3: Not very proud
    4: Not at all proud
    """
    VERY_PROUD = 1
    QUITE_PROUD = 2
    NOT_VERY_PROUD = 3
    NOT_AT_ALL_PROUD = 4


class Y002Options(IntEnum):
    """
    Options for the most and second most important goals:
    1: Maintaining order in the nation
    2: Giving people more say in important government decisions
    3: Fighting rising prices
    4: Protecting freedom of speech
    """
    MAINTAINING_ORDER = 1
    GIVING_PEOPLE_SAY = 2
    FIGHTING_PRICES = 3
    PROTECTING_FREEDOM = 4


class Y002(BaseModel):
    most_important: Y002Options = Field(description="Most important goal")
    second_most_important: Y002Options = Field(description="Second most important goal")

    @field_validator("most_important", "second_most_important")
    def check_valid_values(cls, v):
        if v not in Y002Options:
            raise ValueError("Invalid value. Choose from 1, 2, 3, 4")
        return v

    @model_validator(mode='after')
    def check_combinations(cls, values):
        most_important = values.most_important
        second_most_important = values.second_most_important
        if most_important == second_most_important:
            raise ValueError("The two choices must be different")
        return values


class Y003Options(IntEnum):
    GOOD_MANNERS = 1
    INDEPENDENCE = 2
    HARD_WORK = 3
    FEELING_OF_RESPONSIBILITY = 4
    IMAGINATION = 5
    TOLERANCE_RESPECT = 6
    THRIFT = 7
    DETERMINATION = 8
    RELIGIOUS_FAITH = 9
    UNSELFISHNESS = 10
    OBEDIENCE = 11


class Y003(BaseModel):
    choices: List[Y003Options] = Field(description="List of chosen qualities, up to five")

    @field_validator("choices")
    def validate_choices(cls, v):
        if len(v) > 5:
            raise ValueError("You can only choose up to five qualities.")
        return v

    @model_validator(mode='after')
    def check_unique_choices(cls, values):
        choices = values.choices
        if len(choices) != len(set(choices)):
            raise ValueError("The choices must be unique.")
        return values
