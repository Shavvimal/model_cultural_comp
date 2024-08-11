from pydantic import BaseModel, Field, model_validator, field_validator
from enum import IntEnum
from typing import List, Type
import asyncio
from ollama import AsyncClient
from tqdm.asyncio import tqdm
import random
import pandas as pd
from qn_classes import  A008, A165, E018, E025, F063, F118, F120, G006, Y002, Y003

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
            response = response.replace(".", "")
            # Check if the response is in the valid values
            if int(response) in self.enum._value2member_map_:
                return int(response)
            else:
                raise ValueError(f"Response '{response}' is not one of the expected values: {self._valid_values}")
        except ValueError as e:
            # Try by removing full stop
            raise ValueError(f"Invalid response: {e}")

    def format_instructions(self) -> str:
        return f"Do NOT explain any reasoning whatsoever. Purely select one of the following options: {', '.join(self._valid_values)}"


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
                    messages=[message, {"role": "system", "content": "Sure thing! Here is my numerical answer:"}]
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

    async def generate_all_responses(self, full_prompt_set, llm, max_retries=15, num_workers=10):
        """"
        async gather all_prompts for every llm
        """
        results = []

        async def worker(task_queue, pbar):
            while not task_queue.empty():
                task = await task_queue.get()
                try:
                    returned = await task
                    results.append(returned)
                    pbar.update(1)  # Update the progress bar
                except Exception as e:
                    print(f"Error executing task: {e}")

        try:
            task_queue = asyncio.Queue()

            for qn, prompt in full_prompt_set:
                task_queue.put_nowait(asyncio.create_task(self.generate_response(qn, prompt, llm, max_retries)))

            total_tasks = task_queue.qsize()
            print(f"Starting {total_tasks} tasks with {num_workers} workers for {llm}...")
            with tqdm(total=total_tasks) as pbar:
                await asyncio.gather(*[worker(task_queue, pbar) for _ in range(num_workers)])

        except Exception as e:
            print(f"\nUnable to get data: {e}\n")
        return results

    async def main(self):
        # select random 10 prompts
        random_prompts = random.sample(self._FULL_PROMPT_SET, 10)

        for llm in self._LLMS:
            res = await self.generate_all_responses(self._FULL_PROMPT_SET, llm)
            responses_df = pd.DataFrame(res, columns=["llm", "question", "response"])
            # Make llm filename safe
            llm_name = llm.replace(":", "-")
            llm_name = llm_name.replace("/", "-")
            responses_df.to_pickle(f"../data/collection/{llm_name}_responses_df.pkl")
            print(f"Responses for {llm_name} saved.")

    def __call__(self, *args, **kwargs):
        asyncio.run(self.main())

if __name__ == '__main__':
    llms = [
        # "dolphin-mistral:7b",
        # "dolphin-llama3:8b",
        # "mistral:7b",
        # "llama3:70b",
        # "dolphin-mixtral:8x7b",
        # "llama3:8b",
        # "llama2:13b", # Was too error prone
        # "gemma2:27b", # Was really good at following instructions
        # Chinese but can be prompted in Eng
        # "llama2-chinese:13b", # Was too errir prone
        # "qwen2:7b" # was really good actually
        "wangrongsheng/llama3-70b-chinese-chat"
    ]
    survey = Survey(llms=llms)
    survey()
