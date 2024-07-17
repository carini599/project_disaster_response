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
        The file contains the message id and categories.
    * disaster_messages.csv
        The file contains the message id, message text, the original message and the genre.
    * DisasterResponse.db
        Data of the csv-files are combined in the table cat_messages.
    * process_data.py
        The program loads the CSV-files and transforms them and stores the combined data in the table cat_messages of the DisasterResponse.db
2. models
    * train_classifier.py
        The program train_classifier.py imports data from the cat_messages table of the sqlite database DisasterResponse.db and creates a machine learning model from it. 
        Therefore the messages are tokenized, vectorized and tfidf-transformed, before they are used to train a Random Forrest Classifier for multiple outputs.
        Finally the program exports the model to the classifier.pkl Pickle file.
    * classifier.pkl
        The pickle-file stores the machine learning model, to classify messages.
3. app
    * run.py
        The program starts the flask web server, which displays the model and contains the plotly code for the charts on the web app.
    * templates
        * master.html
            Project Landing Page with several charts and input for classification of a custom sentence.
        * go.html
            Classification Results of the entered message.

## Licensing, Authors, Acknowledgements

I have to thank Appen for providing the data for this exciting project and Udacity for pushing myself to the limits again and again and thereby expanding my horizon enormously.