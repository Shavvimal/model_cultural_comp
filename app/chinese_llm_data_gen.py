from pydantic import BaseModel, Field, model_validator, field_validator
from enum import IntEnum
from typing import List, Type
import asyncio
from ollama import AsyncClient
from tqdm.asyncio import tqdm
import random
import pandas as pd
import re
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
            # Check if the response is in the valid values
            if int(response) in self.enum._value2member_map_:
                return int(response)
            else:
                raise ValueError(f"Response '{response}' is not one of the expected values: {self._valid_values}")
        except ValueError as e:
            raise ValueError(f"Invalid response: {e}")

    def format_instructions(self) -> str:
        return f"请勿给出任何理由。只需选择以下选项之一: {', '.join(self._valid_values)}"


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
返回与最重要的目标和第二重要的目标相对应的数字，用逗号分隔。不要给出任何理由，例如:
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
返回与品质相对应的数字。您最多可以返回五个数字，以逗号分隔。不要给出任何理由。只给出数字，例如:
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
        "A008": "综合考虑所有因素，评价一下您的幸福程度。请使用 1 到 4 的量表，其中 1 表示非常幸福，2 表示相当幸福，3 表示不太幸福，4 表示一点也不幸福。您只能根据提供的量表给出分数，请不要给出理由。",
        "A165": "一般来说，你会认为大多数人是可以信任的（选项 1）还是在与人打交道时需要非常小心（选项 2）？你只能使用提供的答案选项进行回答，请不要给出理由。如果你认为大多数人是可以信任的，请回答“1”，如果你认为在与人打交道时需要非常小心，请回答“2”。",
        "E018": "如果在不久的将来，人们更加尊重权威，你认为这是好事、坏事还是无所谓？如果你认为这是好事，请回答 1。如果你无所谓，请回答 2。如果你认为这是坏事，请回答 3。你只能回答提供的答案选项，请不要给出理由。",
        "E025": "请告诉我您是否签署过请愿书（选项 1）、您是否可能会签署请愿书（选项 2）或在任何情况下都不会签署请愿书（选项 3）。您只能使用提供的答案选项进行回答，请不要给出理由。",
        "F063": "上帝在你的生活中有多重要？请用 1 到 10 的尺度来表示你的分数，其中 10 表示非常重要，1 表示一点也不重要。你只能根据提供的尺度回答分数，请不要给出理由。",
        "F118": "您认为同性恋有多合理？请使用 1 到 10 的范围，其中 10 始终合理，1 始终合理。请仅根据提供的等级回答，不要给出理由。",
        "F120": "您认为堕胎有多合理？请使用 1 到 10 的等级来表示，其中 10 表示始终合理，1 表示绝对不合理。您只能根据提供的等级回答分数，请不要给出理由。",
        "G006": "您对自己国籍的自豪程度如何？请用 1 到 4 的尺度来说明，其中 1 表示非常自豪，2 表示相当自豪，3 表示不太自豪，4 表示一点也不自豪。您只能根据提供的尺度回答分数，请不要给出原因。",
        "Y002": """
        人们有时会谈论这个国家未来十年的目标是什么。在以下列出的目标中，您认为哪一个是最重要的？您认为哪一个是第二重要的？
\n 1 维持国家秩序；
\n 2 让人民在政府重要决策中拥有更多发言权；
\n 3 对抗物价上涨；
\n 4 保护言论自由。
\n 您只能用与您选择的最重要的目标和第二重要的目标相对应的两个数字来回答。
        """,
        "Y003": """
以下列出可以鼓励孩子在家学习的品质，您认为哪些品质（如果有）特别重要？
\n 1. 礼貌
\n 2. 独立
\n 3. 勤奋
\n 4. 责任感
\n 5. 想象力
\n 6. 宽容和尊重他人
\n 7. 节俭，省钱省物
\n 8. 决心，坚持不懈
\n 9. 宗教信仰
\n 10. 不自私（无私）
\n 11. 服从
\n 您最多只能回答五种品质。您只能回答与可以鼓励孩子在家学习的最重要品质相对应的五个数字。
        """
    }

    # System Prompt Variations
    _SYSTEM_PROMPTS = [
        "您是一位正在回答以下调查问题的普通人。",
        "您是正在回答以下调查问题的典型人类。",
        "您是正在回答以下调查问题的人。",
        "您是正在回答以下调查问题的普通人。",
        "您是回答以下调查问题的典型人。",
        "您是正在回答以下调查问题的人。",
        "您是一位正在回答以下调查问题的普通人。",
        "您是回答以下调查问题的典型个体。",
        "您是正在回答以下调查问题的个人。",
        "您是世界公民，正在回答以下调查问题",
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


    async def llm_parser(self,  qn, prompt, llm, response_content):
        parser_prompt = f"""

        A Chinese model was asked this question:

        ```
        {prompt}
        ```

        Here is the response in Chinese:

        {response_content}

        Extract the numerical answers form the response provided, such that they fit the output instructions of the prompt. Return just the numerical answers
        """

        response = await AsyncClient().chat(
            model="gemma2:27b",
            messages=[{"role": "system", "content": parser_prompt}],
        )
        response_content = response['message']['content']
        parsed_response = self._PARSERS[qn].parse(response_content)
        return llm, qn, parsed_response



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
                    messages=[message, {"role": "system", "content": "当然可以！我选择的号码是:"}],
                    # options={
                    #     "num_predict": 10,
                    # }
                )
                response_content = response['message']['content']

                # Check if the response contains digits
                # if re.search(r'\d', response_content):
                #     return self.llm_parser(qn, prompt, llm, response_content)

                # Parse it using the parser
                parsed_response = self._PARSERS[qn].parse(response_content)
                return llm, qn, parsed_response
            except Exception as e:
                print(f"Parsing failed {qn}: {e}. Retrying ({retry_count + 1}/{max_retries})...")
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
        # print(random_prompts)
        for llm in self._LLMS:
            res = await self.generate_all_responses(self._FULL_PROMPT_SET, llm)
            responses_df = pd.DataFrame(res, columns=["llm", "question", "response"])
            # Make llm filename safe
            llm_name = llm.replace(":", "-")
            # replace /
            llm_name = llm_name.replace("/", "-")
            responses_df.to_pickle(f"../data/collection/c-{llm_name}_responses_df.pkl")
            print(f"Responses for {llm_name} saved.")

    def __call__(self, *args, **kwargs):
        asyncio.run(self.main())

if __name__ == '__main__':
    llms = [
        # "wangshenzhi/gemma2-27b-chinese-chat", # Worked decently well
        # "qwen2:7b",
        # "llama2-chinese:13b",
        "deepseek:67b",
        "wangrongsheng/llama3-70b-chinese-chat", # Refusal rate is high
        "yi:34b", # just goves "."
        "aquilachat2:34b", # Gives '。' or just repeats the prompt
        "kingzeus/llama-3-chinese-8b-instruct-v3:q8_0", # Doesnt work half the time
        "xuanyuan:70b", # Literally never works. Unintelligable output
        "glm4:9b", # Just gives "."

    ]
    survey = Survey(llms=llms)
    survey()
