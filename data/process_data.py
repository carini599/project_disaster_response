# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

def load_data(messages_filepath, categories_filepath):
    '''The function loads data from the messages and the categories csv into the dataframes, merges them and return a combined dataframe.'''
    
    # Import CSVs
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge messages and categories
    df = messages.merge(categories, on='id', how='inner')

    return df

def clean_data(df):
    '''Divide column categories into columns for each category'''
    
    # Create a list of the 36 individual category columns
    row = list(df[df['id']==2]['categories'].str.split(';', expand=True).values[0])

    category_colnames = []

    for r in row:
        category_colnames.append(r.split('-')[0])

    # Split category column into separate category columns
    df[category_colnames] = df['categories'].str.split(';', expand=True)

    # Loop over all category columns and keep only values behind the '-'
    for cat in category_colnames:
        df[cat]=df[cat].str.split('-',n=2,expand=True)[1]
        df[cat]=df[cat].astype('int')
    
    # Delete original categories column, which is no longer needed
    df.drop(columns='categories', inplace=True)

    df.drop_duplicates(inplace=True)
    
    return df
    

def save_data(df, database_filename):
    '''Save dataframe to database'''
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('cat_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()