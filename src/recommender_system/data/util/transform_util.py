from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

def item_lookup(train_df:pd.DataFrame):
    unique_item_ids=train_df.article_id.unique() # list of unique article_ids found in training dataset
    vocab=build_vocab_from_iterator([yield_tokens(unique_item_ids)], specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    article_vocab_size=len(unique_item_ids)+1
    return [vocab,article_vocab_size]

def user_lookup(train_df:pd.DataFrame):
    unique_user_ids=train_df.customer_id.unique() # list of unique customer_ids found in training dataset
    vocab=build_vocab_from_iterator([yield_tokens(unique_user_ids)], specials=["<unk>"]) # vocab is a torchtext.vocab.Vocab object
    user_vocab_size=len(unique_user_ids)+1
    return [vocab,user_vocab_size]

def yield_tokens(unique_ids):
    for id in unique_ids:
        yield str(id)
