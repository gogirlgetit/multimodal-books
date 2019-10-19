#
# Reads an ePub. Identifies all important topics in the book. For each topic,
# search for videos - identify top few videos. For each video, compute a
# relevance score based on the ML Model we used. Choose the most relevant
# video and link the video to the topic from the book. Write this modified
# content to a new ePub.
#
# Command: python <book.py>
#
# Uses ebooklib, nltk and beautifulsoup libraries.
#

from apiclient.discovery import build
from apiclient.errors import HttpError
from bs4 import BeautifulSoup
from datetime import datetime
from ebooklib import epub
from nltk.corpus import stopwords
from nltk.metrics import jaccard_distance
from nltk.metrics import edit_distance
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import ebooklib
import isodate
import math
import os
import re
import sys
import urllib


# Given a text string, remove all non-alphanumeric
# characters (using Unicode definition of alphanumeric).
def stripNonAlphaNum(text):
    import re
    return re.compile(r'\W+', re.UNICODE).split(text)

# Stop words and blacklist words
stop_words = set(stopwords.words('english'))
blacklist = {u"chapter", u"fig", u"figure", u"fig:", u"fig :", u"(i)"}

# Read a given epub.
book = epub.read_epub('in.epub')

# Set API_KEY to the "API key" value from the "Access" tab of the
# Google APIs Console http://code.google.com/apis/console#access
API_KEY = "AIzaSyAsdP8B0_yF5TV-2cTXngAFsDi7XUvESqA"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_WATCH_BASE_URL = "https://www.youtube.com/watch?v="

# Returns the best video for the topic
def search_by_keyword(query):
  youtube = build(
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    developerKey=API_KEY
  )

  # Use YT search to come up with an initial list of 25 videos.
  search_response = youtube.search().list(
    q=query,
    part="id,snippet",
    type="video",
    maxResults=25
  ).execute()

  # Remove stop words and normalize the query words.
  query_tokens = word_tokenize(' '.join(stripNonAlphaNum(query)))
  filtered_query = [w.lower() for w in query_tokens if not w in stop_words]
  nquery = ' '.join(filtered_query)

  # Build a dictionary of videoid and search rank
  ids = ""
  rank = 1
  search_rank = {}
  for search_result in search_response.get("items", []):
      search_rank[search_result["id"]["videoId"]] = rank
      ids = ids + "," + search_result["id"]["videoId"]
      rank = rank + 1

  # Get metadata for all these videos.
  video_list = youtube.videos().list(
    part="id,snippet,contentDetails,localizations,statistics,topicDetails",
    id=ids
  ).execute()

  # Extract features from this metadata
  sorted_videos = {}
  for video in video_list.get("items", []):
      # Extract audio language
      audioLang = "en"
      lang_match = 1
      if ('defaultAudioLanguage' in video.get("snippet")):
        audioLang = video.get("snippet")["defaultAudioLanguage"]
      if (audioLang.startswith("en") == False): lang_match = 0

      # Remove non Alpha Num characters from the title
      title = ' '.join(stripNonAlphaNum(video.get("snippet")["title"]))
      # Remove stop words and stem the title words.
      title_tokens = word_tokenize(title)
      filtered_title = [w.lower() for w in title_tokens if not w in stop_words]
      ntitle = ' '.join(filtered_title)

      # Compute Age of the Video
      publishedAt = video.get("snippet")["publishedAt"]
      publishedAtDT = datetime.strptime(publishedAt, '%Y-%m-%dT%H:%M:%S.%fZ')
      age = datetime.today() - publishedAtDT

      # Compute Video duration
      dur = isodate.parse_duration(video["contentDetails"]["duration"])

      # Compute edit distance between the query and title
      edistance = edit_distance(ntitle, nquery)

      # Extract views, likes and dislikes
      view_count = like_count = dislike_count = 0
      if ('viewCount' in video.get("statistics")):
        view_count = int(video.get("statistics")["viewCount"])
      if ('likeCount' in video.get("statistics")):
        like_count = int(video.get("statistics")["likeCount"])
      if ('dislikeCount' in video.get("statistics")):
        dislike_count = int(video.get("statistics")["dislikeCount"])
      if like_count == "0": like_count = "1"

      # Compute score based on ML model
      score = .0000332 * like_count + .00000192 * view_count + 1.08 * lang_match
      score += -.00344 * dislike_count - .881 * edistance - .000147 * dur.total_seconds()
      score += .00000000413 * age.total_seconds() - .0283 * search_rank[video.get("id")]
      sorted_videos[score] = video

  # return top video from the sorted list of Videos.
  return sorted(sorted_videos.iteritems(), key = lambda x : x[0], reverse=True)[0][1]


book2 = epub.EpubBook()

# Copy Identifiers meta data
identifiers = book.get_metadata("DC", "identifier")
for iden in identifiers:
  book2.add_metadata("DC", "identifier", iden[0])

# Copy Creators meta data
creators = book.get_metadata('DC', 'creator')
for creator in creators:
  book2.add_metadata('DC', 'creator', creator[0])

# Copy Contributors meta data
contributors = book.get_metadata('DC', 'contributor')
for contributor in contributors:
  book2.add_metadata('DC', 'contributor', contributor[0])

# Copy Titles and meta data
title_num = 1
titles = book.get_metadata('DC', 'title')
for title in titles:
  book2.add_metadata('DC', 'title', title[0])
  if (title_num == 1): title_type = "main"
  else: title_type = "subtitle"
  refines = "#t" + str(title_num)
  # Epub3.x uses this set of title meta-data
  book2.add_metadata(None, 'meta', title_type, {'refines': refines, 'property': 'title-type'})
  book2.add_metadata(None, 'meta', str(title_num), {'refines': refines, 'property': 'display-seq'})
  title_num += 1

# Copy Description meta data
descriptions = book.get_metadata('DC', 'description')
for description in descriptions:
  book2.add_metadata('DC', 'description', description[0])

# Copy Dates meta data
dates = book.get_metadata('DC', 'date')
for date in dates:
  book2.add_metadata('DC', 'date', date[0])

# Copy Coverages meta data
coverages = book.get_metadata('DC', 'coverage')
for coverage in coverages:
  book2.add_metadata('DC', 'coverage', coverage[0])

# Copy Publishers meta data
publishers = book.get_metadata('DC', 'publisher')
for publisher in publishers:
  book2.add_metadata('DC', 'publisher', publisher[0])

# Copy Rights meta data
rights = book.get_metadata('DC', 'rights')
for right in rights:
  book2.add_metadata('DC', 'rights', right[0])

for item in book.get_items():
  if item.get_type() == ebooklib.ITEM_DOCUMENT:
    # Append to the Spine
    book2.spine.append(item.get_id())

    soup = BeautifulSoup(item.get_content(), 'html.parser')
    topics = set()

    print('==================================')
    print('ID : ', item.get_id())
    print('----------------------------------')

    # Add YT video link to all H1 tags
    for h1 in soup.find_all('h1'):
        output = re.sub('[0-9\.]+', '', ' '.join(h1.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(h1.strings)
        h1.replace_with(youtube_link)

    # Add YT video link to all H2 tags
    for h2 in soup.find_all('h2'):
        output = re.sub('[0-9\.]+', '', ' '.join(h2.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(h2.strings)
        h2.replace_with(youtube_link)

    # Add YT video link to all H3 tags
    for h3 in soup.find_all('h3'):
        output = re.sub('[0-9\.]+', '', ' '.join(h3.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(h3.strings)
        h3.replace_with(youtube_link)

    # Add YT video link to all H4 tags
    for h4 in soup.find_all('h4'):
        output = re.sub('[0-9\.]+', '', ' '.join(h4.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(h4.strings)
        h4.replace_with(youtube_link)

    # Add YT video link to all H5 tags
    for h5 in soup.find_all('h5'):
        output = re.sub('[0-9\.]+', '', ' '.join(h5.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(h5.strings)
        h5.replace_with(youtube_link)

    # Add YT video link to all H6 tags
    for h6 in soup.find_all('h6'):
        output = re.sub('[0-9\.]+', '', ' '.join(h6.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(h6.strings)
        h6.replace_with(youtube_link)

    # Add YT video link to all P ConceptHeading tags
    for concept in soup.find_all("p", class_="ConceptHeading"):
        output = re.sub('[0-9\.]+', '', ' '.join(concept.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(concept.strings)
        concept.replace_with(youtube_link)

    # Add YT video link to all P SubHeading tags
    for subheading in soup.find_all("p", class_="SubHeading"):
        output = re.sub('[0-9\.]+', '', ' '.join(subheading.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(subheading.strings)
        subheading.replace_with(youtube_link)

    # Add YT video link to all P SubHeading2 tags
    for subhead2 in soup.find_all("p", attrs={"class": "SubHeading2"}):
        output = re.sub('[0-9\.]+', '', ' '.join(subhead2.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(subhead2.strings)
        subhead2.replace_with(youtube_link)

    # Add YT video link to all B tags
    for b in soup.find_all('b'):
        output = re.sub('[0-9\.]+', '', ' '.join(b.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(b.strings)
        b.replace_with(youtube_link)

    # Add YT video link to all Strong tags
    for strong in soup.find_all('strong'):
        output = re.sub('[0-9\.]+', '', ' '.join(strong.strings))
        video = search_by_keyword(output.strip().lower())
        youtube_link = soup.new_tag("a", href=YOUTUBE_WATCH_BASE_URL + video["id"])
        youtube_link.string = ' '.join(strong.strings)
        strong.replace_with(youtube_link)

    for bitem in blacklist:
        topics.discard(bitem)

    print('==================================')
    item.set_content(str(soup))
  book2.add_item(item)

# write to the file
epub.write_epub('out.epub', book, {})
epub.write_epub('out2.epub', book2, {})

