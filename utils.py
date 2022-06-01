def save_artcile_info(category, year, file_suffix, article_df):
    article_df.reset_index(inplace=True)
    article_df.rename({'index':'id'}, inplace=True, axis = 'columns')
    article_df.insert(0, 'Category', category)
    article_df.insert(0, 'Year', year)
    article_df.to_csv('./out/article_info-{0}.csv'.format(file_suffix), index=False)

def save_topics(category, year, file_suffix, topics_df):
    topics_df.reset_index(inplace=True)
    col = topics_df.pop('TopicId')
    topics_df.insert(1, col.name, col)
    topics_df.insert(0, 'Category', category)
    topics_df.insert(0, 'Year', year)
    topics_df.to_csv('./out/topics-{0}.csv'.format(file_suffix), index=False)