from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import spacy

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))

from graph import GraphVisualization


tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-large')
model = TFGPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-large')


visualization = GraphVisualization()

visualization.generate_graph_dict("I am a vegetarian")
visualization.generate_graph_dict("I have an Android")
visualization.generate_graph_dict("I love learning")
visualization.generate_graph_dict("I loved the weather")
visualization.generate_graph_dict("I am happy today")
visualization.generate_graph_dict("I love my family")
visualization.generate_graph_dict("I love my friends")

# visualization.draw_graph()

step = 0
while step < 15:
    # encode the new user input, add the eos_token and return a tensor in Tensorflow
    input_text = input(">> User: ")

    tokenized = sent_tokenize(input_text)
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        wordsList = [w for w in wordsList if not w in stop_words]

        outputs = nltk.pos_tag(wordsList)

    nouns = []
    for word, label in outputs:
        if "NN" in label or "PRP" in label:
            nouns += [word]
    
    context_arr = [visualization.pull_data(noun) for noun in nouns]
    context_string = ""

    for i in context_arr:
        for j in i:
            context_string += j + " "

    new_user_input_ids = tokenizer.encode(context_string + input_text + tokenizer.eos_token, return_tensors='tf')

    visualization.generate_graph_dict(input_text)

    # append the new user input tokens to the chat history
    bot_input_ids = tf.concat([chat_history_ids, new_user_input_ids], axis=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

    step += 1

visualization.draw_graph()