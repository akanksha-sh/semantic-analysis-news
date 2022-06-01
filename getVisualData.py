import pandas as pd
import glob
import os

def getCombinedData():
    # For artcile info
    files = os.path.join("./out/", "article_info*.csv")
    files = glob.glob(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv('./data/combined_articles.csv', index=False)

    # For topic info
    files = os.path.join("./out/", "topics*.csv")
    files = glob.glob(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv('./data/combined_topics.csv', index=False)