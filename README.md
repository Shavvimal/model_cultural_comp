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


## Poetry 

````bash
poetry add pandas numpy matplotlib jupyter pyreadstat scikit-learn factor-analyzer umap-learn 
````

## References

[^26]: EVS (2022): EVS Trend File 1981-2017. GESIS Data Archive, Cologne. ZA7503 Data file Version 3.0.0, doi:10.4232/1.14021
[^27]: Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen et al. (eds.). 2022. World Values Survey Trend File (1981-2022) Cross-National Data-Set. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. Data File Version 4.0.0, doi:10.14281/18241.27.



