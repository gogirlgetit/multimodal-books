# 
# Iterate through hand curated training data and complement it with
# features from YT.
#
# Command line: python <training_data_features.py>
#

from apiclient.discovery import build
from apiclient.errors import HttpError
from datetime import datetime
from datetime import timedelta
from nltk.corpus import stopwords
from nltk.metrics import jaccard_distance
from nltk.metrics import edit_distance
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import csv
import io
import isodate
import math
import nltk
import os
import re
import sys
import urllib

# Set API_KEY to the "API key" value from the "Access" tab of the
# Google APIs Console http://code.google.com/apis/console#access
#API_KEY = "AIzaSyCuejn2NYwsJt-VPCUM22-ai0kgGGC8yfM" #
API_KEY = "AIzaSyCt6Bk4nBb7-GviQyP1vr98K4GeaJWigvM" # Temporary12345
#API_KEY = "AIzaSyAsdP8B0_yF5TV-2cTXngAFsDi7XUvESqA" #Education
#API_KEY = "AIzaSyCIbdpAbIb_KcOdnjX30HyjnPyWX2RxdVI" #JustKutir
#API_KEY = "AIzaSyA8vLXEpBXLumG_cYLSmUppCfyCWJKBUZc" #YetAnotherOne
#API_KEY = "AIzaSyAyQqgwMkxDyM0sXe_tDWXvy-3cyu0sI-s"
#API_KEY = "AIzaSyCmIyBZhu367qs7bpajVy2ZBtEo1KBwRr0"
#API_KEY = "AIzaSyAZesb1gmYziPrK4RY9tXFUX57kJZqeYRU"
#API_KEY = "AIzaSyCHSOXAlVFd_WkLKtKo3vDiL7_d9SuFLMI"
#API_KEY = "AIzaSyDU12FW5qY6yjc8fBK26ngE96Nc4ZsOGi8"
#API_KEY = "AIzaSyCQYLyWWga30OYI3iKwdeYZ3_CXpAjVAFI"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

stop_words = set(stopwords.words('english')) 
ps = PorterStemmer()

# Given a text string, remove all non-alphanumeric
# characters (using Unicode definition of alphanumeric).
def stripNonAlphaNum(text):
    import re
    return re.compile(r'\W+', re.UNICODE).split(text)

# Get youtube video to extract features for training
def get_video(vid):
  youtube = build(
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    developerKey=API_KEY
  )

  # Need snippet, content details, localizations, stats and
  # topic details for training
  results = youtube.videos().list(
    part="snippet,contentDetails,localizations,statistics,topicDetails",
    id=vid
  ).execute()

  for item in results.get("items", []):
    return item
  return None

# Output file for training data with features.
ftraining = io.open("training_data_features.out", "w+", encoding="utf8")

# Iterate through hand curated training data (hand_curated_ratings.tsv) and
# complement it with features from YT.
with open('hand_curated_ratings.tsv','rb') as tsvin:
  tsvin = csv.reader(tsvin, delimiter='\t')
  for row in tsvin:
    print(row)

    # Parse the row into Query, Video Id, Relevance score and Search Rank
    # Remove punctuation marks and other non Alpha Num characters
    query = ' '.join(stripNonAlphaNum(unicode(row[0], 'utf-8')))
    vid = row[1]
    relevance = float(row[2])
    rank = float(row[3])

    # Remove stop words and normalize the query
    query_tokens = word_tokenize(query)
    #filtered_query = [ps.stem(w) for w in query_tokens if not w in stop_words]
    filtered_query = [w.lower() for w in query_tokens if not w in stop_words]
    nquery = ' '.join(filtered_query)

    video = get_video(vid)
    if (video is not None):
      # extract audio language
      audioLang = "en"
      lang_match = 1
      if ('defaultAudioLanguage' in video.get("snippet")):
        audioLang = video.get("snippet")["defaultAudioLanguage"]
      if (audioLang.startswith("en") == False): lang_match = 0

      # Extract title and remove non Alpha Num characters
      title = ' '.join(stripNonAlphaNum(video.get("snippet")["title"]))
      # Remove stop words and stem the title words.
      title_tokens = word_tokenize(title)
      #filtered_title = [ps.stem(w) for w in title_tokens if not w in stop_words]
      filtered_title = [w.lower() for w in title_tokens if not w in stop_words]
      ntitle = ' '.join(filtered_title)

      # Extract age of the video
      publishedAt = video.get("snippet")["publishedAt"]
      publishedAtDT = datetime.strptime(publishedAt, '%Y-%m-%dT%H:%M:%S.%fZ')
      age = datetime.today() - publishedAtDT

      # Extract video duration
      dur = isodate.parse_duration(video["contentDetails"]["duration"])

      # Compute edit distance between the query and the title
      #edistance = jaccard_distance(set(ntitle), set(nquery))
      edistance = edit_distance(ntitle, nquery)

      # Extract Views, Likes and Dislikes
      view_count = like_count = dislike_count = "1"
      if ('viewCount' in video.get("statistics")):
        view_count = video.get("statistics")["viewCount"]
      if ('likeCount' in video.get("statistics")):
        like_count = video.get("statistics")["likeCount"]
      if ('dislikeCount' in video.get("statistics")):
        dislike_count = video.get("statistics")["dislikeCount"]

      # Write training data with features to a file
      ftraining.write(
          u"%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%f\t%f\n" %
          (query.encode('ascii', 'ignore'),
           title.encode('ascii', 'ignore'),
           age.total_seconds(),
           lang_match,
           view_count,
           like_count,
           dislike_count,
           dur.total_seconds(),
           edistance,
           rank,
           relevance))
      print('==================================')

