import json
import csv

file = open("yelp_training_set/yelp_training_set_review.json")
with open('yelp.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['text', 'stars', 'sentiment'])
    for line in file:
        data = json.loads(line)
        if data['stars'] >= 4:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        csvwriter.writerow([data['text'], data['stars'], sentiment])
