import praw
import json
import os
from datetime import datetime

# Reddit API credentials
reddit = praw.Reddit(
    client_id="4Mqq1qhf9ugFLxACcpcK3w",
    client_secret="zueS3m_6nvrZJfaI04W31_IKXcVEiA",
    user_agent="fractal_geometry",
)

# Get a specific post by ID
submission = reddit.submission(id="106frj6")  # Replace with an actual post ID. Normally found in the post URL

# Remove "Load More Comments"
submission.comments.replace_more(limit=None)  # Fetch all comments

# Get post creation time
post_timestamp = datetime.fromtimestamp(submission.created_utc)

# Function to recursively extract comments with timestamps
def extract_comments(comment, depth=0):
    comment_timestamp = datetime.fromtimestamp(comment.created_utc)
    
    return {
        "author": comment.author.name if comment.author else "Deleted",
        "body": comment.body,
        "score": comment.score,
        "depth": depth,
        "timestamp": comment_timestamp.strftime("%Y-%m-%d %H:%M:%S"), #yyyy/mm/dd
        "replies": [extract_comments(reply, depth + 1) for reply in comment.replies]
    }

# Extract structured comments
comment_data = [extract_comments(comment) for comment in submission.comments]

# Find the last comment timestamp
all_timestamps = []
def collect_timestamps(comment):
    all_timestamps.append(datetime.strptime(comment["timestamp"], "%Y-%m-%d %H:%M:%S"))
    for reply in comment["replies"]:
        collect_timestamps(reply)

for comment in comment_data:
    collect_timestamps(comment)

if all_timestamps:
    last_comment_time = max(all_timestamps)
    time_difference = last_comment_time - post_timestamp
else:
    last_comment_time = None
    time_difference = None

# Print results
output_data = {
    "post_title": submission.title,
    "post_timestamp": post_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    "last_comment_timestamp": last_comment_time.strftime("%Y-%m-%d %H:%M:%S") if last_comment_time else "No comments",
    "time_difference": str(time_difference) if time_difference else "No comments",
    "comments": comment_data
}

# Create jason_data folder if it doesn't exist
os.makedirs("json_data", exist_ok=True)

# Save to file
with open("json_data/post55lb_reddit_comments_with_time.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Reddit post and comments saved to json_data/post55lb_reddit_comments_with_time.json")