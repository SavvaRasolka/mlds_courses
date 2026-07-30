import json

def concatenate_intents():
    with open('old_val_intents.json', 'r', encoding='utf-8') as f:
        dataset_deepseek = json.load(f)
    with open('new_intents.json', 'r', encoding='utf-8') as f:
            dataset_gpt = json.load(f)
    dataset = {"intents": [
    {
      "tag": "greeting",
      "patterns": [
       
      ],
      "responses": []
    },
    {
      "tag": "goodbye",
      "patterns": [],
      "responses": []
    },
    {
      "tag": "thanks",
      "patterns": [],
      "responses": []
    },
    {
      "tag": "care",
      "patterns": [
        
      ],
      "responses": []
    },
    {
      "tag": "feed",
      "patterns": [
        
      ],
      "responses": [
              ]
    },
    {"tag": "health",
      "patterns": [
        
      ],
      "responses": [
              ]
},
    {"tag": "behavior",
      "patterns": [
        
      ],
      "responses": [
              ]
},
    {"tag": "training",
      "patterns": [
        
      ],
      "responses": [
              ]
},
    {"tag": "toys",
      "patterns": [
        
      ],
      "responses": [
              ]
},
    {"tag": "space",
      "patterns": [
        
      ],
      "responses": [
              ]
},
    {
      "tag": "laws",
      "patterns": [
        
      ],
      "responses": [
             ]
    }
  ]
}
    for intent, deepseek, gpt in zip(dataset["intents"], dataset_deepseek["intents"],dataset_gpt["intents"]):
        intent["patterns"].extend(deepseek["patterns"])
        intent["patterns"].extend(gpt["patterns"])
        print(len(intent["patterns"]), intent["tag"])            
    
    with open('train_intents.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
       