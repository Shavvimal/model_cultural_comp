# GraphRAG

- [ ] Manage updating the Graph with new data without re-indexing 
- [ ] Check Backlinks from Updates
- [ ] Write a visualisation script
  - See the streamlit app in the demo. You can see the communities, view the summaries on another tab. You can also click in to see the sources of the relationships, nodes and also the data used to make the response.
  - Critical for using this data
  - Think about using a second agent to provide a verification evaluation based on the provided context to see if there are any hallucinations in the returned response. Helps with after the fact analysis on if the response was correctly grounded or not
- [ ] Hook into exisitng workflow to continuously update the Graph 
  - CRON 
  - Index all my AI news data and set up a CRON job to update it
- [ ] Link it to my own twitter 
- [ ] Do this for Podcasts and create a marketing thing
  - throw it at the transcripts for a deep podcast list - you could really pick out connections between concepts that you didn't see initially

# Getting Started 

```bash
python -m graphrag.index --root ./bin
```

```bash
python -m graphrag.query --root ./bin --method global "What are the top 5 companies in the AI space?"
```

```bash
python -m graphrag.query --root ./bin --method local "What is OpenAI, and what are the main relationships?"
```

# Visualise

Can use the Notebook I've added. I need to see if I can get the visualise python script they have left in the package to work. 

## Resources

- [GraphRAG Github](https://github.com/microsoft/graphrag)
- [GraphRAG Blog Post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [GraphRAG Docs](https://microsoft.github.io/graphrag/)
- [GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [Project GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/overview/)
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher)
- [Plan-and-Solve](https://arxiv.org/abs/2305.04091)
- [RAG](https://arxiv.org/abs/2005.11401)
- [Knowledge Graphs are key to unlocking the power of AI](https://www.youtube.com/watch?v=lhRYnZS7yu4&list=WL&index=29&t=373s)
- [How to Build Knowledge Graphs With LLMs (python tutorial)](https://www.youtube.com/watch?v=tcHIDCGu6Yw)
- [Exploring Large Language Models for Knowledge Graph Completion](https://arxiv.org/abs/2308.13916)
- [4 Ways Unstructured Data Management Will Change in 2024](https://builtin.com/articles/unstructured-data-management-change)
- [Expert Reveals Key Data Management Trends for 2024 to Know](https://solutionsreview.com/data-management/data-management-trends-for-2024/)
- [Harnessing LLMs With Neo4j](https://medium.com/neo4j/harnessing-large-language-models-with-neo4j-306ccbdd2867)
- [Fine-Tuning vs Retrieval-Augmented Generation](https://medium.com/neo4j/knowledge-graphs-llms-fine-tuning-vs-retrieval-augmented-generation-30e875d63a35)
- [Knowledge Graphs & LLMs: Multi-Hop Question Answering](https://medium.com/neo4j/knowledge-graphs-llms-multi-hop-question-answering-322113f53f51)
- [Knowledge Graphs & LLMs: Real-Time Graph Analytics](https://medium.com/neo4j/knowledge-graphs-llms-real-time-graph-analytics-89b392eaaa95)
- [Construct Knowledge Graphs From Unstructured Text](https://medium.com/neo4j/construct-knowledge-graphs-from-unstructured-text-877be33300a2)
- [Project NaLLM](https://github.com/neo4j/NaLLM?tab=readme-ov-file)
- [Constructing knowledge graphs from text using OpenAI functions](https://bratanic-tomaz.medium.com/constructing-knowledge-graphs-from-text-using-openai-functions-096a6d010c17)
- [LangChain Cypher Search: Tips & Tricks](https://medium.com/neo4j/langchain-cypher-search-tips-tricks-f7c9e9abca4d)
- [Extract knowledge from text: End-to-end information extraction pipeline with spaCy and Neo4j](https://towardsdatascience.com/extract-knowledge-from-text-end-to-end-information-extraction-pipeline-with-spacy-and-neo4j-502b2b1e0754)
- [Text to Knowledge Graph Made Easy with Graph Maker](https://towardsdatascience.com/text-to-knowledge-graph-made-easy-with-graph-maker-f3f890c0dbe8)
- [How to Convert Any Text Into a Graph of Concepts](https://towardsdatascience.com/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a)
- [Accelerating Scientific Discovery with Generative Knowledge Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning](https://arxiv.org/abs/2403.11996)
- [GraphRAG: LLM-Derived Knowledge Graphs for RAG](https://www.youtube.com/watch?v=r09tJfON6kE)
- [GraphRAG Ollama: 100% Local Setup, Keeping your Data Private](https://www.youtube.com/watch?v=BLyGDTNdad0)
- [Easy GraphRAG with Neo4j Visualisation Locally](https://www.youtube.com/watch?v=Dw2g2NEdsw0)
- [Sciphi Triplex](https://www.youtube.com/watch?v=GR0jyxTKyYY)
- [R2R Knowledge Graphs](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph)
- [GraphRAG: LLM-Derived Knowledge Graphs for RAG](https://www.youtube.com/watch?v=r09tJfON6kE&t=833s)
