# Project Log
## Nov 30, 2017
#### Redesign Categorization Workflow

- Update the Categorization.csv file by adding tier, themes, and keywords
    - I removed the *execution behavior* as I think the Verb Phrase detection can 
    do better than defining verbs manually -- remain to be examined
    - We can always add *execution behavior* back if we want a pre-defined 
    behavior for reporting-purposes, and at that time, we will need to cluster 
    similar verbs together
        - For example: how many feedbacks are about "adding a feedback"
- Create a class for categorization dictionary and a function dedicated to 
read and store the dictionary
- Noun Phrases can  be a supplement to Verb Phrases in keyword detection. When 
no keyword is identified in Verb Phrases, use keywords from Noun Phrases
    - If no keywords are identified in NP and VP, search for keywords in the
    entire text
    - Idea: we may improve this by assigning different weights to keywords from
    None Phrases and from Verb Phrases
- Extract the key phrases for actions: Use the identified keyword as a clue to 
extract the original phrases in the sentence
    - If the keyword is in Verb Phrase, identify that Verb Phrase and use that
    as the action phrase
    - Otherwise, identify the location of the keyword in the original feedback
    text, tract back by up to 2 words, and use that phrase for the action phrase.
    
<br>

## Jan 7
Identify problem related to the dataframe length in data processing
- When iterating through the dataframe in categorization part, the index will go beyond the length of the dataset.
- Solution: save the dataframe into a csv file into a local path and reload it, the problem will get solved without 
the need of any other changes
- Try to cut the dataframe length by `df = df[:len(df)]`, cannot solve the problem

Add a summarize function to summarize the issue
- Now just an easy trick: use the first action identified or if there is no action identified, simply give the sentiment
as "General Positive Feedback", etc.

<br>

## Jan 18
Reformat the outputs to remove the quotation marks around the outputs in components, actions, etc.

<br>

## Jan 22
[x] Revise the output format of Categorization
[x] Clustering
[x] Issue summarization for same cluster of feedbacks

<br>

## Feb 12
[x] Extract keywords for each clusters
- Select keywords from VP, NP, and Actions
- Skip stop words and keywords
- Recover from stemmed words

<br>

## Feb 18
[x] Generate synonyms and keywords for Components: 
- Goal: the user only need to give in a list of Components, and the system can automatically generate the relevant keywords
- Approach: 
    - Use WordNet to find synonyms: accurate but does not help in finding non-synonyms but relevant words
        - Example: find `spotlight` for `highlight`, `TV` for `video`, `universal resource locator` for `URL`
        - Filter out the synonyms whose similiarity score is below 0.9
    - Find all the high frequent words from the original translated feedbacks, and match them with the component based on semantic similarity
        - Does not help, thus abandoned 
    - Cut terminology: cut `History/cookies/cache` into `history`, `cookies`, and `cache` and find synonyms individually
    
## Feb 19
[x] Improve clustering
- Problem: currently the clustering method can effectively cluster the sentences with the same set of words. However, 
the algorithm cannot deal with synonyms (words that are semantically closed but in different forms). 
- Approach:
    - Word selection: Find all the high-frequent words among all the feedback texts. 
    (Words from the same origin but presented in the different tenses/plural forms are considered as a same word)
        - For example: `difficult` and `hard`
    - Identify and cluster synonyms: find out the words that are semantically similar to each other
    - For each cluster, find out the word with the highest frequency, and add it to the sentences that the other words appear in
        - I did insertion instead of replacement, in order retain the original information
    - Process the sentences before passing to the clustering model: stem words, lower case, remove stop words 

[x] Modify the decision making on components
- Use both the Verb Phrases and Noun Phrases for deciding components, as long as one of them is available
    - Motivation: this can help when there is no indicative verb in the sentence, but some key terms are mentioned
    - Example: "Runs smoothly and is a lot faster than the safari. Keep it up" 
        - Clearly this is a feedback that should be categorize for "General Browser", but if we only rely on `Verb Phrase`, 
        we only get "run smoonthly" and "is a lot". These are not indicative. In this case, the addition of "Noun Phrase" will 
        help. 

[x] Limit the number of tags:
- Set an upper limit on tags for a cluster to be 5

[x] Create a new dataframe for tag
- Modify Pandas DataFrame
- Modify the SQL code



    