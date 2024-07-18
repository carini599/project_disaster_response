# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

def load_data(messages_filepath, categories_filepath):
    '''The function loads data from the messages and the categories csv into the dataframes, merges them and return a combined dataframe.'''
    
    # Import CSVs from messages_filepath
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge messages and categories
    df = messages.merge(categories, on='id', how='inner')

    return df

def clean_data(df):
    '''Divide column categories into separate columns for each category'''
    
    # Create a list of the 36 individual category columns
    # Take first row of categories column and split it by the seperator ; and convert to list
    row=df['categories'].iloc[0].split(';')
    
    category_colnames = []

    # For each element in the list get only the first part before the "-" and append it to category_colnames list.
    for r in row:
        category_colnames.append(r.split('-')[0])

    # Split category column values into separate category columns
    df[category_colnames] = df['categories'].str.split(';', expand=True)

    # Loop over all category columns and keep only values behind the '-'
    for cat in category_colnames:
        df[cat]=df[cat].str.split('-',n=2,expand=True)[1]
        df[cat]=df[cat].astype('int').astype('bool')
    
    # Delete original categories column, which is no longer needed
    df.drop(columns='categories', inplace=True)

    # Delete duplicates from the dataset
    df.drop_duplicates(inplace=True)
    
    return df
    

def save_data(df, database_filename):
    '''Save dataframe df to specified database database_filename'''
    
    # Create database connection
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save dataset df to table cat_messages in database
    df.to_sql('cat_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        #Loading Data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        #Cleaning data
        print('Cleaning data...')
        df = clean_data(df)
        
        #Save data to specified database
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