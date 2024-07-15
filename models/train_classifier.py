# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine, text
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    
    #sqllite engine file path
    path = f'sqlite:///{database_filepath}'
    
    #Create connetion to sqllite database and get table cat_messages
    engine = create_engine(path)
    df = pd.read_sql(sql=text('SELECT * FROM cat_messages'), con=engine.connect())

    #Create List with category names from columns
    columns_Y=list(df.columns)[4:]

    X = df['message']
    Y = df[columns_Y]

    return X,Y


def tokenize(text):
    '''Function to tokenize text, convert it to lowercase and remove spaces.'''

    tok_words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #List of tokenized words
    clean_toks= []
    
    #Loop over each word in text, convert it to lowercase and append it to list clean_toks.
    for tok in tok_words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_toks.append(clean_tok)
    
    return clean_toks


def build_model():
    ''' Create pipeline, that vectorizes tokenzized words, 
    convert it to tfidf and calculates a random forrest model for mulitple outputs '''
    
    model = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf' , TfidfTransformer()),
    ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    # Predict values for test data
    Y_pred = pd.DataFrame(model.predict(X_test))
    Y_pred.columns = category_names

    # For each category (column) evaluate model with classification report
    for cat in category_names:
        print(f'Category: {cat}/n')
        print(classification_report(Y_test[cat], Y_pred[cat]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()