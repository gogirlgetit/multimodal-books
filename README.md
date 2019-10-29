# multimodal-books
Given a textbook, make it multi-modal by embedding Video links in the textbook. Research paper: https://www.ijntr.org/download_data/IJNTR05100013.pdf

Reads an ePub. Identifies all important topics in the book. For each topic, search for videos - identify top few videos. For each video, compute a relevance score based on the ML Model we used. Choose the most relevant video and link the video to the topic from the book. Write this modified content to a new ePub.

Steps:
1. Create your own training data. In this repo, hand_curated_ratings.tsv is the training dataset.
2. Run training_data_features.py to pull features for the hand curated training data set.
3. Train an ML model based on this training data with features using model.py.
4. Use the co-efficients from the Model in book.py and run book.py on an ePub.
5. Test to make sure that out.epub has YouTube video links embedded and generally enhance the quality of the original text.
