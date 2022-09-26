# Sentence Selection

This folder handles selection and filtering of sentences from the various corpora. 

Corpora:
- COCA (Spoken)
- Cornell Movie Subtitles (Spoken)
- Brown (Print)
- Wall Street Journal (Print)
- Universal Dependencies (Print)
- Toronto (Print)
- C4 (Web) 

## sent_selection.py
Arguments:
- `data_dir`: directory where data files are found
- `filter_and_save`: set this flag to perform initial filtering
- `downsample`: set this flag to perform downsampling

### Filter
Sentences from the corpora are extracted and saved. The following filters are applied:
- Remove sentences containing more than 4 puncutation characters in a row
- Remove sentences in which more than 50% of characters are capitalized
- Remove sentences containing words with more than 20 characters
- Remove sentences where more than 50% of words contain numerals

### Downsample
The previous step results in a very large number of sentences, and some corpora are represented much more than others. We also want to assign sentences to high-, mid-, and low-memorability categories using our ground truth word memorability dataset. 
- The `get_memorable_words()` function uses results from the word memorability study to identify ground truth high-, mid-, and low-memorability words within the sentence datasets. It returns a dataframe with additional columns for these ground truth words. 
- Next we identify and exclude outlier high-frequency ground truth words. These are defined as ground truth words that appear in more than 10% of sentences. For example, the word "she" was identified as high-memorability but appeared in >10% of sentences. 
- Finally, we sample a fixed number (currently set at 25k) of sentences from each category: high-, mid-, and low-memorability, as well as a fourth category of "no ground truth words". 

### Mean Memorability
Given the filtered and downsampled sentences, we want to assign a pseudo-memorability score to each sentence based on our trained linear model that maps from word embeddings to memorability scores. 
- We calculate non-contextual word embeddings for each word in the dataset (using GloVe). 
- For each word, we apply the linear model to get a pseudo-memorability score. 
- The pseudo-memorability score of an entire sentence is defined simply as the mean score of its words. 