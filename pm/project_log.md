# Project Log
## Nov 30, 2017
### Redesign Categorization Workflow

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
    
    
## Questions
- Shall we set up categories for: 
    - Menu
    
## To-Do
- Build up a feedback database
- Develop a function to extract high frequent verb and noun associate with a given
component
    - Input: a keyword (component)
    - Output: 
        - A list of top frequent verb with corresponding percentage
        - A list of top frequent noun with corresponding percentage
- Pull out the top 100 frequent keywords in the feedback database
- Build up a parameterized classifier
- Cannot extract the Verb Phrase `sort my favorites`
    - `my` is `PRP$`
    - `sort the favorites` is extractable
    - For now, we can still extract the keyword `favorites` and use that as a 
    clue


    