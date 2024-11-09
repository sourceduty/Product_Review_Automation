import requests
from textblob import TextBlob
from transformers import pipeline

# Fetch reviews (Example: Simulated review data)
def fetch_reviews():
    # Simulated review dataset
    reviews = [
        "The product is fantastic! It works perfectly.",
        "Terrible quality, broke within a week. Not recommended.",
        "Average experience. Not great, but not bad either.",
        "Amazing! Exceeded my expectations.",
        "The product is okay, but the customer service was awful."
    ]
    return reviews

# Analyze sentiment of each review
def analyze_sentiment(reviews):
    sentiment_results = []
    for review in reviews:
        blob = TextBlob(review)
        polarity = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        sentiment_results.append({"Review": review, "Sentiment": sentiment, "Polarity": polarity})
    return sentiment_results

# Summarize reviews
def summarize_reviews(reviews):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    combined_reviews = " ".join(reviews)  # Combine all reviews for summary
    summary = summarizer(combined_reviews, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Main program
if __name__ == "__main__":
    print("Fetching reviews...")
    reviews = fetch_reviews()
    
    print("\nAnalyzing sentiment...")
    sentiment_results = analyze_sentiment(reviews)
    for result in sentiment_results:
        print(f"Review: {result['Review']}\nSentiment: {result['Sentiment']}\nPolarity: {result['Polarity']}\n")
    
    print("\nSummarizing reviews...")
    summary = summarize_reviews(reviews)
    print(f"Summary of Reviews: {summary}")
