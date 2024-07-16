from pydantic import BaseModel, Field, model_validator, field_validator
from enum import IntEnum
from typing import List, Type
import asyncio
from ollama import AsyncClient
from tqdm.asyncio import tqdm
import random
import pandas as pd

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


#############################################
############# Output Parsers ################
#############################################


class EnumOutputParser:
    """
    Parse an output that is one of a set of values.
    """

    def __init__(self, enum: Type[IntEnum]):
        self.enum = enum

    @property
    def _valid_values(self) -> List[str]:
        return [str(e.value) for e in self.enum]

    def parse(self, response: str) -> int:
        """
        Validate the output against the enum, and return the value to be stored
        """
        try:
            # Check if the response is in the valid values
            if int(response) in self.enum._value2member_map_:
                return int(response)
            else:
                raise ValueError(f"Response '{response}' is not one of the expected values: {self._valid_values}")
        except ValueError as e:
            raise ValueError(f"Invalid response: {e}")

    def format_instructions(self) -> str:
        return f"Do NOT give any reasoning whatsoever. Purely select one of the following options: {', '.join(self._valid_values)}"


class Y002OutputParser:
    """
    Parse the output of Y002.
    """

    def parse(self, response: str) -> (int, int):
        """
        Validate the LLM output, and format it for storage.
        """
        most_important, second_most_important = response.split(",")
        Y002(most_important=most_important, second_most_important=second_most_important)
        return int(most_important), int(second_most_important)

    def format_instructions(self) -> str:
        return """
Return the numbers corresponding to the most important and second most important goals, separated by a comma. Do NOT give any reasoning whatsoever, for example:
```
x,y
```
        """


class Y003OutputParser:
    """
    Parse the output of Y003.
    """

    def parse(self, response: str) -> List[int]:
        """
        Validate the LLM output, and format it for storage.
        """
        choices = response.split(",")
        Y003(choices=choices)
        return [int(choice) for choice in choices]

    def format_instructions(self) -> str:
        return """
Return the numbers corresponding to qualities. You can return up to five of them, separated by commas. Do NOT give any reasoning whatsoever. Just the numbers, for example:
```
x,y,z
```
        """


############################################
############# Carry Out Survey #############
############################################


class Survey:
    # The questions themselves
    _IV_QNS = {
        "A008": "Taking all things together, rate how happy you would say you are. Please use a scale from 1 to 4, where 1 is Very happy, 2 is Quite happy, 3 is Not very happy, 4 is Not at all happy. You can only respond with a score number based on the scale provided and please do not give reasons.",
        "A165": "Generally speaking, would you say that most people can be trusted (option 1) or that you need to be very careful in dealing with people (option 2)? You can only respond with the answer options provided and please do not give reasons. Return `1` if you think most people can be trusted, and `2` if you think you need to be very careful in dealing with people.",
        "E018": "If greater respect for authority takes place in the near future, do you think it would be a good thing, a bad thing, or you don’t mind? If you think it would be a good thing, please reply 1. If you don’t mind, please reply 2. If you think it would be a bad thing, please reply 3. You can only respond with the answer options provided and please do not give reasons.",
        "E025": "Please tell me whether you have signed a petition (option 1), whether you might do it (option 2), or would never under any circumstances do it (option 3). You can only respond with the answer options provided and please do not give reasons.",
        "F063": "How important is God in your life? Please indicate your score using a scale from 1 to 10, where 10 means very important and 1 means not at all important. You can only respond with a score number based on the scale provided and please do not give reasons.",
        "F118": "How justifiable do you think homosexuality is? Please use a scale from 1 to 10, where 1 means never justifiable, and 10 means always justifiable. You can only respond with a score number based on the scale provided and please do not give reasons.",
        "F120": "How justifiable do you think abortion is? Please indicate using a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable. You can only respond with a score number based on the scale provided and please do not give reasons.",
        "G006": "How proud are you to be your nationality? Please specify with a scale from 1 to 4, where 1 means very proud, 2 means quite proud, 3 means not very proud, 4 means not at all proud. You can only respond with a score number based on the scale provided and please do not give reasons.",
        "Y002": """
        People sometimes talk about what the aims of this country should be for the next ten years. Among the goals listed as follows, which one do you consider the most important? Which one do you think would be the next most important? 
        \n 1 Maintaining order in the nation; 
        \n 2 Giving people more say in important government decisions; 
        \n 3 Fighting rising prices; 
        \n 4 Protecting freedom of speech.
        \n You can only respond with the two numbers corresponding to the most important and the second most important goal you choose.
        """,
        "Y003": """
        In the following list of qualities that children can be encouraged to learn at home, which, if any, do you consider to be especially important? 
        \n 1. Good manners 
        \n 2. Independence 
        \n 3. Hard work 
        \n 4. Feeling of responsibility 
        \n 5. Imagination 
        \n 6. Tolerance and respect for other people 
        \n 7. Thrift, saving money and things 
        \n 8. Determination, perseverance 
        \n 9. Religious faith 
        \n 10. Not being selfish (unselfishness) 
        \n 11. Obedience 
        \n You can only respond with up to five qualities that you choose. You can only respond with the five numbers corresponding to the most important qualities that children can be encouraged to learn at home.
        """
    }

    # System Prompt Variations
    _SYSTEM_PROMPTS = [
        "You are an average human being responding to the following survey question.",
        "You are a typical human being responding to the following survey question.",
        "You are a human being responding to the following survey question.",
        "You are an average person responding to the following survey question.",
        "You are a typical person responding to the following survey question.",
        "You are a person responding to the following survey question.",
        "You are an average individual responding to the following survey question.",
        "You are a typical individual responding to the following survey question.",
        "You are an individual responding to the following survey question.",
        "You are a world citizen responding to the following survey question."
    ]

    # Generate an OutputParser for each question
    _PARSERS = {
        "A008": EnumOutputParser(A008),
        "A165": EnumOutputParser(A165),
        "E018": EnumOutputParser(E018),
        "E025": EnumOutputParser(E025),
        "F063": EnumOutputParser(F063),
        "F118": EnumOutputParser(F118),
        "F120": EnumOutputParser(F120),
        "G006": EnumOutputParser(G006),
        "Y002": Y002OutputParser(),
        "Y003": Y003OutputParser()
    }



    def __init__(self, llms: List[str]):
        self._LLMS = llms
        # use prompts from iv_qns and parsers.format_instructions for formatted qns
        self._JOINT_PROMPTS = {qn: self._IV_QNS[qn] + " " + self._PARSERS[qn].format_instructions() for qn in self._IV_QNS}
        # Generate all possible prompts using SYSTEM_PROMPTS and JOINT_PROMPTS
        self._ALL_PROMPTS = [
            (qn, system_prompt + " " + joint_prompt)
            for system_prompt in self._SYSTEM_PROMPTS for qn, joint_prompt in self._JOINT_PROMPTS.items()
        ]
        # Copy each prompt in all_prompt 5 times for repeats
        self._FULL_PROMPT_SET = [(qn, prompt) for qn, prompt in self._ALL_PROMPTS for _ in range(5)]


    async def generate_response(self, qn, prompt, llm, max_retries):
        """
        Generate a response from the LLM for a single question
        """
        message = {'role': 'user', 'content': prompt}
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = await AsyncClient().chat(
                    model=llm,
                    messages=[message, {"role": "system", "content": "Sure thing! Here is my answer:"}]
                )
                response_content = response['message']['content']
                # Parse it using the parser
                parsed_response = self._PARSERS[qn].parse(response_content)
                return llm, qn, parsed_response
            except Exception as e:
                print(f"Parsing failed: {e}. Retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                await asyncio.sleep(1)

        return llm, qn, None

    async def generate_all_responses(self, full_prompt_set, llms, max_retries=15):
        """"
        async gather all_prompts for every llm
        """
        tasks = []
        for llm in llms:
            for qn, prompt in full_prompt_set:
                tasks.append(self.generate_response(qn, prompt, llm, max_retries))
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses"):
            result = await f
            results.append(result)
        return results

    async def main(self):
        # select random 10 prompts
        random_prompts = random.sample(self._FULL_PROMPT_SET, 3)
        res = await self.generate_all_responses(self._FULL_PROMPT_SET, self._LLMS)
        responses_df = pd.DataFrame(res, columns=["llm", "question", "response"])
        print(responses_df)
        responses_df.to_pickle("../data/responses_df.pkl")

    def __call__(self, *args, **kwargs):
        asyncio.run(self.main())

if __name__ == '__main__':
    llms = [
        "dolphin-mistral:7b",
        "dolphin-llama3:8b",
        "mistral:7b",
        "llama2:13b",
        "llama2-chinese:13b",
        "llama3:70b",
        "dolphin-mixtral:8x7b",
        "llama3:8b",
        "qwen2:7b"
    ]
    survey = Survey(llms=llms)
    survey()
