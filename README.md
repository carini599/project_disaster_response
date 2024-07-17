### Project Disaster Response Pipeline

## Installation

## Project Motivation

The project "disaster response pipeline" is part of my nanodegree course in Data Science at Udacity. The program imports csv-files provided by Appen (www.appen.com) with messages during catastrophic events and categorizes them using a ML learning pipeline.
After importing the csv-files and storing them to a sqlite-database, the data is used to train a machine learning pipeline. The pipeline reads in the tokenized messages and vectorizes them and transforms them using a tfidf-transformer.
The TFIDF-transformed messages are then used to train a Multiple Output Classifier, which is based on a Random Forrest Classifier.
To improve results, GridSearch is used to find the best results for certain parameters. 

The model is finally displayed in a flask web app, where the user can enter potential messages which then are classified according to the ML model. 

## File Descriptions

The project is subdivided into three folders. 
1. data
    * disaster_categories.csv
    * disaster_messages.csv
    * DisasterResponse.db
    * process_data.py
2. models
    * train_classifier.py
    * classifier.pkl
3. app
    * run.py
    * templates
        ** master.html
        ** go.html

## Licensing, Authors, Acknowledgements

I have to thank Appen for providing the data for this exciting project and Udacity for pushing myself to the limits again and again and thereby expanding my horizon enormously.