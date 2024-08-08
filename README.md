# GraphRAG

- [ ] Add a visualisation script
  - See the streamlit app in the demo. You can see the communities, view the summaries on another tab. You can also click in to see the sources of the relationships, nodes and also the data used to make the response.
  - Critical for using this data
  - Use a second agent to provide a verification evaluation based on the provided context to see if there are any hallucinations in the returned response. Helps with after the fact analysis on if the response was correctly grounded or not
- [ ] See a way to Continuously update the Graph 
- [ ] See if i can invoke it progrmatically
- [ ] Index all my AI news data and set up a CRON job to update it
- Link it to my own twitter 
- Do this for Podcasts and sell a marketing thing
  - throw it at the transcripts for a deep podcast list - you could really pick out connections between concepts that you didnt see.

# Getting Started 

```bash
python -m graphrag.index --root .
```

```bash
python -m graphrag.query --root . --method global "What are the top 5 companies in the AI space?"
```

```bash
python -m graphrag.query --root . --method local "What is OpenAI, and what are the main relationships?"
```

# Visualise

Can use the Notebook. Lets get the visualise python script to work. 

