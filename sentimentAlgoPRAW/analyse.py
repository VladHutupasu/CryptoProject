import praw
from textblob import TextBlob
import math


reddit = praw.Reddit(client_id='M8X9AvTd2vFqlg',
                     client_secret='0Li9mKaR1q-D3W7Q0k3PhsxzNzs',
                     user_agent='subSentiment')

# opens file with subreddit names
with open('sb.txt') as f:

    for line in f:
        subreddit = reddit.subreddit(line.strip())
        # write web agent to get converter for datetime to epoch on a daily basis for updates
        # day_start = 1523750401
        # day_end = 1523836799

        day_start = 1510635601
        day_end = 1510721999

        sub_submissions = subreddit.submissions(day_start, day_end)
        print(sub_submissions)

        sub_sentiment = 0
        num_comments = 0

        for submission in sub_submissions:
            if not submission.stickied:
                # submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                        blob = TextBlob(comment.body)

                        #adds comment sentiment to overall sentiment for subreddit
                        comment_sentiment = blob.sentiment.polarity
                        sub_sentiment += comment_sentiment
                        num_comments += 1

                        #prints comment and polarity
                        #print(str(comment.body.encode('utf-8')) + ': ' + str(blob.sentiment.polarity))

        print('/r/' + str(subreddit.display_name))
        try:
            print('Ratio: ' + str(math.floor(sub_sentiment/num_comments*100)) + '\n')
        except:
            print('No comment sentiment.' + '\n')
            ZeroDivisionError

# add key:value subredditname:sentiment to dict
# order dictionary by sentiment value
# output dictionary key + ' sentiment: ' + sentiment value