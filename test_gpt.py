import os
import openai
openai.organization = None
openai.api_key = None

# seems like the api key is not correct

input = """
        Here are some examples of extracting relationships from text:

        Text: I love music
        Relationship: [(I, love): music]
        Text: Today was a great day, I had a blast
        Relationship: [(Today, great day): was, (I, blast): had]
        Text: My dog had to see the vet because he got sick
        Relationship: [(my, dog): , (dog, vet): to see, (dog, sick): got ]
        Text: I really liked the AlphaFold paper that Deepmind released
        Relationship: [(I, AlphaFold paper): really liked, (AlphFold paper, Deepmind): released]
        Text:
        """

completion = openai.Completion.create(engine="deployment-name", prompt="Hello world")

# print the completion
print(completion.choices[0].text)