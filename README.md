# Model Comparison Github Repo

## IVS Data

We followed the same procedure detailed on the website of the WVS Association for [creating the World Cultural Map](https://www.worldvaluessurvey.org/wvs.jsp). EVS and WVS time-series data-sets are released independently by EVS/GESIS and the WVSA/JDS. The Integrated Values Surveys (IVS) dataset 1981-2022 is constructed by merging the [EVS Trend File 1981-2017](https://search.gesis.org/research_data/ZA7503) [^26] and the [WVS trend 1981-2022 data-set](https://www.worldvaluessurvey.org/WVSEVStrend.jsp) [^27].

I have decided to go ahead with the SPSS format, although it makes no difference. The files you will need are the following:

- WVS Trend File 1981-2022 (4.0.0): `Trends_VS_1981_2022_sav_v4_0.sav`
- EVS Trend File 1981-2017 (3.0.0): `ZA7503_v3-0-0.sav`
- IVS_Merge_Syntax (SPSS): `EVS_WVS_Merge Syntax_Spss_June2024.sps`

Use these to create `Trends_VS_1981_2022_sav_v4_0.sav`

## Running Models

GGUF is a file format used for storing large language models. I use this file to run LLMs in Ollama. For the comparison, we are looking at these models in particular:


1. **Chinese LLMs:**

   - **[AquilaChat2-34B-16K](https://huggingface.co/BAAI/AquilaChat2-34B-16K):** Using the Q4 quantization `aquilachat2-34b-16k.Q4_K_M.gguf` from [`TheBloke/AquilaChat2-34B-16K-GGUF`](https://huggingface.co/TheBloke/AquilaChat2-34B-16K-GGUF). This model has an architecture similar to LLama2, making it ideal for comparison.
   - **[Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat):** Using the Q4 quantization `yi-34b-chat.Q4_K_M.gguf` from [`TheBloke/Yi-34B-Chat-GGUF`](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF).
   - **[Qwen2:7B](https://huggingface.co/Qwen/Qwen2-7B):** Using the [Q4 model](https://ollama.com/library/qwen2).
   - **[GLM-4v-9B](https://huggingface.co/THUDM/glm-4-9b-chat):** Using the Q4 quantization `glm-4-9b-chat.Q4_K.gguf` from [legraphista/glm-4-9b-chat-GGUF](https://huggingface.co/legraphista/glm-4-9b-chat-GGUF).
   - **[Llama2-Chinese](https://ollama.com/library/llama2-chinese):** A Llama 2 based model fine-tuned to improve Chinese dialogue ability.

2. **Western LLMs:**

   - **[LLaMA2:7B](https://ollama.com/library/llama2):** Using the Q4 quantization.
   - **[LLaMA3:8B](https://ollama.com/library/llama3:8b):** Using the Q4 quantization.
   - **[Mistral](https://ollama.com/library/mistral):** The 7B model released by Mistral AI.

3. **Dolphin Models:** These models are uncensored and have been trained on data filtered to remove alignment and bias, making them more compliant.
   - **[Dolphin-Mixtral:8x7B](https://ollama.com/library/dolphin-mixtral):** An uncensored, fine-tuned model based on the Mixtral mixture of experts models, available in 8x7B and 8x22B configurations.
   - **[dolphin-llama3:8b](https://ollama.com/library/dolphin-llama3)**: Dolphin 2.9 is a new model with 8B and 70B sizes by Eric Hartford based on Llama 3.


Ollama models are stored at `C:\Users\%username%\.ollama\models` so I will save the models there aswell. I `mkdir` a folder for `huggingface-hub` then save the models there. Then I use the `huggingface-cli` command to download the models

````bash
poetry run huggingface-cli download {repo_name} {file_name} --local-dir  "C:\Users\shavh\.ollama\models\huggingface-hub\" 
````

For example:

````bash
poetry run huggingface-cli download TheBloke/AquilaChat2-34B-16K-GGUF aquilachat2-34b-16k.Q4_K_M.gguf --local-dir  "C:\Users\shavh\.ollama\models\huggingface-hub\" 
````

| Repo                              | File                            |
| --------------------------------- | ------------------------------- |
| TheBloke/AquilaChat2-34B-16K-GGUF | aquilachat2-34b-16k.Q4_K_M.gguf |
| TheBloke/Yi-34B-Chat-GGUF         | yi-34b-chat.Q4_K_M.gguf         |
| legraphista/glm-4-9b-chat-GGUF    | glm-4-9b-chat.Q4_K.gguf         |
| RichardErkhov/Duxiaoman-DI_-_XuanYuan-70B-gguf | XuanYuan-70B.Q4_0.gguf |
| TheBloke/deepseek-llm-67b-chat-GGUF | deepseek-llm-67b-chat.Q4_K_M.gguf |

We then have to create [ModelFiles](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) for each of these models. The model file is the blueprint to create and share models with Ollama. For this one, simply it is:

```ModelFile
FROM C:\Users\shavh\.ollama\models\huggingface-hub\aquilachat2-34b-16k.Q4_K_M.gguf
```

Then we use an ollama command to create the model from the ModelFiles:

```bash
ollama create aquilachat2:34b -f aquilachat2
ollama create yi:34b -f yi
ollama create glm4:9b -f glm
```


Also pull from ollama, for example:

```
ollama pull llama2:13b
ollama pull mistral:7b
ollama pull dolphin-llama3:8b
ollama pull dolphin-mistral:7b
```

## References

[^26]: EVS (2022): EVS Trend File 1981-2017. GESIS Data Archive, Cologne. ZA7503 Data file Version 3.0.0, doi:10.4232/1.14021
[^27]: Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen et al. (eds.). 2022. World Values Survey Trend File (1981-2022) Cross-National Data-Set. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. Data File Version 4.0.0, doi:10.14281/18241.27.



