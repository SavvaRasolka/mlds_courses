import json
import matplotlib.pyplot as plt


def count_tags(filename):
    categories = []
    values = []
    with open(filename, 'r', encoding='utf-8') as f:
                dataset_gpt = json.load(f)
    for intent in dataset_gpt["intents"]:
         values.append(len(intent["patterns"]))
         categories.append(intent["tag"])
    plt.bar(categories, values)
    plt.title(filename)
    plt.xlabel('Tag')
    plt.ylabel('Amount')  
    plt.show()