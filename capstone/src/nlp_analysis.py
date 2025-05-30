import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec

# Visualization settings
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
sns.set(style="whitegrid", font_scale=1.2)

# Download necessary NLTK resources
def download_nltk_resources():
    """Download required NLTK resources if not already available."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        print("Download complete.")

# 1. Data Preparation and Text Cleaning
# ------------------------------------

def load_and_preprocess_reviews(file_path=None, df=None, text_column='review_text', 
                               rating_column='rating', product_id_column='product_id'):
    """
    Load and preprocess product reviews data.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the reviews CSV file
    df : pandas DataFrame, optional
        DataFrame containing reviews if already loaded
    text_column : str
        Column name containing review text
    rating_column : str
        Column name containing ratings
    product_id_column : str
        Column name containing product IDs
    
    Returns:
    --------
    pandas DataFrame
        Preprocessed reviews dataframe
    """
    print("Loading and preprocessing review data...")
    
    # Load data if dataframe not provided
    if df is None:
        if file_path is None:
            raise ValueError("Either file_path or df must be provided")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} reviews from {file_path}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    else:
        print(f"Using provided dataframe with {len(df)} reviews")
    
    # Check if required columns exist
    required_columns = [text_column]
    if rating_column not in df.columns:
        print(f"Warning: Rating column '{rating_column}' not found. Sentiment analysis will be based only on text.")
    else:
        required_columns.append(rating_column)
    
    if product_id_column not in df.columns:
        print(f"Warning: Product ID column '{product_id_column}' not found. Product-specific analysis will be limited.")
    else:
        required_columns.append(product_id_column)
    
    # Check if text column exists
    if text_column not in df.columns:
        print(f"Error: Required text column '{text_column}' not found in data")
        return None
    
    # Create a working copy of the dataframe with relevant columns
    reviews_df = df[required_columns].copy()
    
    # Handle missing values
    if text_column in reviews_df.columns:
        # Drop rows with missing review text
        initial_count = len(reviews_df)
        reviews_df = reviews_df.dropna(subset=[text_column])
        dropped_count = initial_count - len(reviews_df)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} rows with missing review text")
    
    # Convert review text to string type
    reviews_df[text_column] = reviews_df[text_column].astype(str)
    
    # Convert ratings to numeric if available
    if rating_column in reviews_df.columns:
        reviews_df[rating_column] = pd.to_numeric(reviews_df[rating_column], errors='coerce')
    
    # Add review length as a feature
    reviews_df['review_length'] = reviews_df[text_column].apply(len)
    reviews_df['word_count'] = reviews_df[text_column].apply(lambda x: len(str(x).split()))
    
    print(f"Preprocessed dataframe contains {len(reviews_df)} reviews")
    
    return reviews_df

def clean_text(text, remove_stopwords=True, lemmatize=True):
    """
    Clean and preprocess text data.
    
    Parameters:
    -----------
    text : str
        Text to clean
    remove_stopwords : bool
        Whether to remove stopwords
    lemmatize : bool
        Whether to lemmatize words
    
    Returns:
    --------
    str
        Cleaned text
    """
    # Check if input is a string
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def add_cleaned_text(reviews_df, text_column='review_text', remove_stopwords=True, lemmatize=True):
    """
    Add cleaned text column to reviews dataframe.
    
    Parameters:
    -----------
    reviews_df : pandas DataFrame
        Reviews dataframe
    text_column : str
        Column name containing review text
    remove_stopwords : bool
        Whether to remove stopwords
    lemmatize : bool
        Whether to lemmatize words
    
    Returns:
    --------
    pandas DataFrame
        Dataframe with added cleaned text column
    """
    print("Cleaning review text...")
    
    # Add cleaned text column
    reviews_df['cleaned_text'] = reviews_df[text_column].apply(
        lambda x: clean_text(x, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    )
    
    # Filter out empty cleaned texts
    initial_count = len(reviews_df)
    reviews_df = reviews_df[reviews_df['cleaned_text'] != '']
    dropped_count = initial_count - len(reviews_df)
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with empty cleaned text")
    
    print(f"Cleaned text added to {len(reviews_df)} reviews")
    
    return reviews_df

# 2. Sentiment Analysis
# -------------------

def analyze_sentiment(reviews_df, text_column='cleaned_text', rating_column=None):
    """
    Perform sentiment analysis on review text.
    
    Parameters:
    -----------
    reviews_df : pandas DataFrame
        Reviews dataframe
    text_column : str
        Column name containing (cleaned) review text
    rating_column : str, optional
        Column name containing ratings
    
    Returns:
    --------
    pandas DataFrame
        Dataframe with added sentiment columns
    """
    print("Performing sentiment analysis...")
    
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to each review
    sentiment_scores = reviews_df[text_column].apply(sid.polarity_scores)
    
    # Extract sentiment scores
    reviews_df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    reviews_df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    reviews_df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    reviews_df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
    
    # Categorize sentiment
    reviews_df['sentiment_category'] = reviews_df['sentiment_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    # Compare with ratings if available
    if rating_column in reviews_df.columns:
        # Normalize ratings to 0-1 scale
        max_rating = reviews_df[rating_column].max()
        min_rating = reviews_df[rating_column].min()
        
        if max_rating != min_rating:
            reviews_df['normalized_rating'] = (reviews_df[rating_column] - min_rating) / (max_rating - min_rating)
        else:
            reviews_df['normalized_rating'] = 0.5  # Default if all ratings are the same
        
        # Create rating-based sentiment
        reviews_df['rating_sentiment'] = reviews_df['normalized_rating'].apply(
            lambda x: 'positive' if x >= 0.7 else ('negative' if x <= 0.3 else 'neutral')
        )
        
        # Calculate agreement between text sentiment and rating sentiment
        reviews_df['sentiment_rating_agreement'] = (
            reviews_df['sentiment_category'] == reviews_df['rating_sentiment']
        )
        
        agreement_rate = reviews_df['sentiment_rating_agreement'].mean() * 100
        print(f"Sentiment-rating agreement rate: {agreement_rate:.2f}%")
    
    sentiment_distribution = reviews_df['sentiment_category'].value_counts(normalize=True) * 100
    print("Sentiment distribution:")
    for category, percentage in sentiment_distribution.items():
        print(f"  - {category}: {percentage:.2f}%")
    
    return reviews_df

def visualize_sentiment(reviews_df, rating_column=None, product_id_column=None):
    """
    Visualize sentiment analysis results.
    
    Parameters:
    -----------
    reviews_df : pandas DataFrame
        Reviews dataframe with sentiment analysis results
    rating_column : str, optional
        Column name containing ratings
    product_id_column : str, optional
        Column name containing product IDs
    
    Returns:
    --------
    None
    """
    print("Visualizing sentiment analysis results...")
    
    # Create figure with multiple subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Sentiment distribution
    plt.subplot(2, 3, 1)
    sentiment_counts = reviews_df['sentiment_category'].value_counts()
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    # Add count labels
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count + 5, f'{count}', ha='center')
    
    # 2. Sentiment compound score distribution
    plt.subplot(2, 3, 2)
    sns.histplot(reviews_df['sentiment_compound'], bins=20, kde=True)
    plt.title('Distribution of Sentiment Compound Scores')
    plt.xlabel('Compound Score')
    plt.ylabel('Count')
    
    # 3. Sentiment vs. Review Length
    plt.subplot(2, 3, 3)
    sns.boxplot(x='sentiment_category', y='word_count', data=reviews_df)
    plt.title('Review Length by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Word Count')
    
    # 4. Sentiment components
    plt.subplot(2, 3, 4)
    sentiment_components = reviews_df[['sentiment_pos', 'sentiment_neu', 'sentiment_neg']].mean()
    sns.barplot(x=sentiment_components.index, y=sentiment_components.values)
    plt.title('Average Sentiment Components')
    plt.xlabel('Sentiment Component')
    plt.ylabel('Average Score')
    
    # 5. If rating column is available, plot rating vs. sentiment
    if rating_column in reviews_df.columns:
        plt.subplot(2, 3, 5)
        sns.boxplot(x='sentiment_category', y=rating_column, data=reviews_df)
        plt.title('Rating by Sentiment Category')
        plt.xlabel('Sentiment')
        plt.ylabel('Rating')
        
        # 6. Rating distribution by sentiment
        plt.subplot(2, 3, 6)
        sentiment_rating = reviews_df.groupby('sentiment_category')[rating_column].value_counts(normalize=True).unstack() * 100
        sentiment_rating.plot(kind='bar', ax=plt.gca())
        plt.title('Rating Distribution by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.legend(title='Rating')
    
    # If product_id is available, create additional visualizations
    if product_id_column in reviews_df.columns:
        plt.figure(figsize=(15, 10))
        
        # Get top products by review count
        top_products = reviews_df[product_id_column].value_counts().head(10).index
        
        # Filter for top products
        top_products_df = reviews_df[reviews_df[product_id_column].isin(top_products)]
        
        # Average sentiment by product
        plt.subplot(2, 1, 1)
        product_sentiment = top_products_df.groupby(product_id_column)['sentiment_compound'].mean().sort_values(ascending=False)
        sns.barplot(x=product_sentiment.index, y=product_sentiment.values)
        plt.title('Average Sentiment by Product (Top 10 by Review Count)')
        plt.xlabel('Product ID')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=45)
        
        # Sentiment distribution by product
        plt.subplot(2, 1, 2)
        sentiment_by_product = pd.crosstab(
            top_products_df[product_id_column], 
            top_products_df['sentiment_category'], 
            normalize='index'
        ) * 100
        sentiment_by_product.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Sentiment Distribution by Product (Top 10)')
        plt.xlabel('Product ID')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')
    
    plt.tight_layout()
    plt.show()

# 3. Word Frequency Analysis
# ------------------------

def analyze_word_frequency(reviews_df, text_column='cleaned_text', by_sentiment=True, 
                         top_n=20, min_word_length=3):
    """
    Analyze word frequency in reviews.
    
    Parameters:
    -----------
    reviews_df : pandas DataFrame
        Reviews dataframe
    text_column : str
        Column name containing (cleaned) review text
    by_sentiment : bool
        Whether to analyze word frequency by sentiment category
    top_n : int
        Number of top words to display
    min_word_length : int
        Minimum word length to include
    
    Returns:
    --------
    tuple
        (word_counts, word_counts_by_sentiment)
    """
    print("Analyzing word frequency...")
    
    # Combine all review text
    all_text = ' '.join(reviews_df[text_column])
    
    # Count word frequencies
    words = all_text.split()
    word_counts = Counter([word for word in words if len(word) >= min_word_length])
    
    # Get top words
    top_words = word_counts.most_common(top_n)
    print(f"Top {top_n} words overall:")
    for word, count in top_words:
        print(f"  - {word}: {count}")
    
    # Analyze by sentiment if requested
    word_counts_by_sentiment = None
    if by_sentiment and 'sentiment_category' in reviews_df.columns:
        word_counts_by_sentiment = {}
        
        for sentiment in reviews_df['sentiment_category'].unique():
            # Combine text for this sentiment
            sentiment_text = ' '.join(
                reviews_df[reviews_df['sentiment_category'] == sentiment][text_column]
            )
            
            # Count word frequencies
            sentiment_words = sentiment_text.split()
            sentiment_word_counts = Counter([word for word in sentiment_words if len(word) >= min_word_length])
            
            # Get top words
            top_sentiment_words = sentiment_word_counts.most_common(top_n)
            word_counts_by_sentiment[sentiment] = dict(top_sentiment_words)
            
            print(f"\nTop {top_n} words in {sentiment} reviews:")
            for word, count in top_sentiment_words:
                print(f"  - {word}: {count}")
    
    return word_counts, word_counts_by_sentiment

def visualize_word_frequency(word_counts, word_counts_by_sentiment=None, top_n=20):
    """
    Visualize word frequency analysis results.
    
    Parameters:
    -----------
    word_counts : Counter
        Word frequency counts
    word_counts_by_sentiment : dict, optional
        Word frequency counts by sentiment
    top_n : int
        Number of top words to display
    
    Returns:
    --------
    None
    """
    print("Visualizing word frequency analysis...")
    
    # Plot overall word frequency
    plt.figure(figsize=(12, 8))
    top_words = dict(word_counts.most_common(top_n))
    
    # Sort by frequency for better visualization
    words = list(top_words.keys())
    freqs = list(top_words.values())
    
    # Sort words by frequency (ascending)
    sorted_indices = np.argsort(freqs)
    sorted_words = [words[i] for i in sorted_indices]
    sorted_freqs = [freqs[i] for i in sorted_indices]
    
    plt.barh(sorted_words, sorted_freqs)
    plt.title(f'Top {top_n} Words Overall')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.show()
    
    # Generate word cloud
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate_from_frequencies(word_counts)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - All Reviews')
    plt.tight_layout()
    plt.show()
    
    # Plot word frequency by sentiment
    if word_counts_by_sentiment:
        # Define sentiment colors
        sentiment_colors = {
            'positive': 'green',
            'neutral': 'gray',
            'negative': 'red'
        }
        
        # Create a figure with subplots for each sentiment
        n_sentiments = len(word_counts_by_sentiment)
        fig, axes = plt.subplots(n_sentiments, 1, figsize=(12, 5 * n_sentiments))
        
        # If there's only one sentiment, wrap axes in a list
        if n_sentiments == 1:
            axes = [axes]
        
        for i, (sentiment, counts) in enumerate(word_counts_by_sentiment.items()):
            # Sort words by frequency
            words = list(counts.keys())
            freqs = list(counts.values())
            
            # Sort words by frequency (ascending)
            sorted_indices = np.argsort(freqs)[-top_n:]  # Get indices of top N
            sorted_words = [words[i] for i in sorted_indices]
            sorted_freqs = [freqs[i] for i in sorted_indices]
            
            # Plot
            color = sentiment_colors.get(sentiment, 'blue')
            axes[i].barh(sorted_words, sorted_freqs, color=color, alpha=0.7)
            axes[i].set_title(f'Top {top_n} Words in {sentiment.capitalize()} Reviews')
            axes[i].set_xlabel('Frequency')
            axes[i].set_ylabel('Word')
        
        plt.tight_layout()
        plt.show()
        
        # Generate word clouds by sentiment
        plt.figure(figsize=(15, 5 * (n_sentiments // 3 + 1)))
        
        for i, (sentiment, counts) in enumerate(word_counts_by_sentiment.items()):
            plt.subplot(n_sentiments // 3 + 1, 3, i + 1)
            
            color = sentiment_colors.get(sentiment, 'blue')
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                max_words=100, contour_width=3, contour_color=color)
            wordcloud.generate_from_frequencies(counts)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud - {sentiment.capitalize()} Reviews')
        
        plt.tight_layout()
        plt.show()

# 4. Topic Modeling
# ---------------

def perform_topic_modeling(reviews_df, text_column='cleaned_text', num_topics=5, num_words=10):
    """
    Perform topic modeling on review text using LDA.
    
    Parameters:
    -----------
    reviews_df : pandas DataFrame
        Reviews dataframe
    text_column : str
        Column name containing (cleaned) review text
    num_topics : int
        Number of topics to extract
    num_words : int
        Number of top words to display per topic
    
    Returns:
    --------
    tuple
        (vectorizer, lda_model, feature_names, topic_words, document_topics)
    """
    print(f"Performing topic modeling with {num_topics} topics...")
    
    # Vectorize text
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(reviews_df[text_column])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Create and fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42,
        max_iter=10,
        learning_method='online'
    )
    
    # Fit model and transform documents
    document_topics = lda_model.fit_transform(dtm)
    
    # Get top words for each topic
    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-num_words-1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_words.append(top_features)
        
        print(f"Topic {topic_idx+1}: {', '.join(top_features)}")
    
    return vectorizer, lda_model, feature_names, topic_words, document_topics

def visualize_topics(lda_model, feature_names, topic_words, document_topics, reviews_df, num_words=10):
    """
    Visualize topic modeling results.
    
    Parameters:
    -----------
    lda_model : LatentDirichletAllocation
        Fitted LDA model
    feature_names : array
        Feature names from vectorizer
    topic_words : list
        List of top words for each topic
    document_topics : array
        Document-topic matrix
    reviews_df : pandas DataFrame
        Reviews dataframe
    num_words : int
        Number of top words to display per topic
    
    Returns:
    --------
    None
    """
    print("Visualizing topic modeling results...")
    
    # Number of topics
    num_topics = len(topic_words)
    
    # Create figure
    fig, axes = plt.subplots(num_topics, 1, figsize=(12, num_topics * 3))
    
    # If there's only one topic, wrap axes in a list
    if num_topics == 1:
        axes = [axes]
    
    # Plot top words for each topic
    for i, (topic_words_list, ax) in enumerate(zip(topic_words, axes)):
        # Get word weights for this topic
        topic = lda_model.components_[i]
        word_indices = topic.argsort()[:-num_words-1:-1]
        word_weights = topic[word_indices]
        
        # Normalize weights for better visualization
        word_weights = word_weights / word_weights.sum()
        
        # Get words
        words = [feature_names[j] for j in word_indices]
        
        # Plot
        ax.barh(words, word_weights, align='center')
        ax.invert_yaxis()
        ax.set_title(f'Topic {i+1}')
        ax.set_xlabel('Normalized Weight')
    
    plt.tight_layout()
    plt.show()
    
    # Plot topic distribution
    plt.figure(figsize=(10, 6))
    
    # Compute the average topic distribution across all documents
    topic_dist = document_topics.mean(axis=0)
    
    # Create labels for x-axis
    topic_labels = [f'Topic {i+1}' for i in range(num_topics)]
    
    # Plot
    plt.bar(topic_labels, topic_dist)
    plt.title('Topic Distribution Across All Reviews')
    plt.xlabel('Topic')
    plt.ylabel('Average Weight')
    plt.ylim(0, topic_dist.max() * 1.2)  # Add some headroom
    
    # Add labels on top of bars
    for i, v in enumerate(topic_dist):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Assign dominant topic to each document
    dominant_topics = document_topics.argmax(axis=1)
    reviews_df['dominant_topic'] = dominant_topics + 1  # 1-based indexing
    
    # Plot number of reviews per dominant topic
    plt.figure(figsize=(10, 6))
    topic_counts = reviews_df['dominant_topic'].value_counts().sort_index()
    
    bars = plt.bar(topic_labels, topic_counts)
    plt.title('Number of Reviews per Dominant Topic')
    plt.xlabel('Topic')
    plt.ylabel('Number of Reviews')
    
    # Add count labels
    for i, v in enumerate(topic_counts):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # If sentiment_category is available, analyze topic distribution by sentiment
    if 'sentiment_category' in reviews_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a crosstab of dominant topic vs. sentiment
        topic_sentiment = pd.crosstab(
            reviews_df['dominant_topic'], 
            reviews_df['sentiment_category'], 
            normalize='index'
        ) * 100
        
        # Plot
        topic_sentiment.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Sentiment Distribution by Topic')
        plt.xlabel('Topic')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.legend(title='Sentiment')
        
        plt.tight_layout()
        plt.show()

# 5. Recommendations From Product Reviews
# ------------------------------------

def extract_product_insights(reviews_df, text_column='cleaned_text', 
                           product_id_column='product_id', rating_column=None,
                           min_reviews=10):
    """
    Extract product insights from reviews.
    
    Parameters:
    -----------
    reviews_df : pandas DataFrame
        Reviews dataframe
    text_column : str
        Column name containing review text
    product_id_column : str
        Column name containing product IDs
    rating_column : str, optional
        Column name containing ratings
    min_reviews : int
        Minimum number of reviews for a product to be included
    
    Returns:
    --------
    pandas DataFrame
        Dataframe with product insights
    """
    if product_id_column not in reviews_df.columns:
        print("Error: Product ID column not found")
        return None
    
    print("Extracting product insights from reviews...")
    
    # Group by product_id
    product_groups = reviews_df.groupby(product_id_column)
    
    # Filter products with enough reviews
    product_review_counts = product_groups.size()
    products_with_min_reviews = product_review_counts[product_review_counts >= min_reviews].index
    
    if len(products_with_min_reviews) == 0:
        print(f"No products with at least {min_reviews} reviews")
        return None
    
    print(f"Analyzing {len(products_with_min_reviews)} products with at least {min_reviews} reviews")
    
    # Initialize results list
    product_insights = []
    
    # Process each product
    for product_id in products_with_min_reviews:
        product_reviews = reviews_df[reviews_df[product_id_column] == product_id]
        
        # Basic stats
        review_count = len(product_reviews)
        avg_review_length = product_reviews['review_length'].mean()
        avg_word_count = product_reviews['word_count'].mean()
        
        # Sentiment analysis
        sentiment_counts = product_reviews['sentiment_category'].value_counts()
        positive_pct = sentiment_counts.get('positive', 0) / review_count * 100
        neutral_pct = sentiment_counts.get('neutral', 0) / review_count * 100
        negative_pct = sentiment_counts.get('negative', 0) / review_count * 100
        avg_sentiment = product_reviews['sentiment_compound'].mean()
        
        # Rating analysis (if available)
        avg_rating = None
        rating_variance = None
        if rating_column in product_reviews.columns:
            avg_rating = product_reviews[rating_column].mean()
            rating_variance = product_reviews[rating_column].var()
        
        # Common words
        all_text = ' '.join(product_reviews[text_column])
        words = all_text.split()
        word_counts = Counter([word for word in words if len(word) >= 3])
        top_words = dict(word_counts.most_common(10))
        
        # Store results
        product_insights.append({
            'product_id': product_id,
            'review_count': review_count,
            'avg_review_length': avg_review_length,
            'avg_word_count': avg_word_count,
            'positive_pct': positive_pct,
            'neutral_pct': neutral_pct,
            'negative_pct': negative_pct,
            'avg_sentiment': avg_sentiment,
            'avg_rating': avg_rating,
            'rating_variance': rating_variance,
            'top_words': ', '.join(top_words.keys())
        })
    
    # Create results dataframe
    insights_df = pd.DataFrame(product_insights)
    
    print(f"Generated insights for {len(insights_df)} products")
    
    return insights_df

def analyze_product_reviews(reviews_df, text_column='review_text', rating_column='rating', 
                          product_id_column='product_id', generate_wordclouds=True):
    """
    Complete workflow for analyzing product reviews with NLP.
    
    Parameters:
    -----------
    reviews_df : pandas DataFrame
        Reviews dataframe
    text_column : str
        Column name containing review text
    rating_column : str
        Column name containing ratings
    product_id_column : str
        Column name containing product IDs
    generate_wordclouds : bool
        Whether to generate word clouds
    
    Returns:
    --------
    dict
        Dictionary containing all analysis results
    """
    print("Starting comprehensive product review analysis...")
    
    # Download required NLTK resources
    download_nltk_resources()
    
    # Step 1: Preprocess reviews
    if reviews_df is None:
        print("Error: No reviews dataframe provided")
        return None
    
    # Add cleaned text
    reviews_df = add_cleaned_text(reviews_df, text_column=text_column)
    
    # Step 2: Sentiment analysis
    reviews_df = analyze_sentiment(reviews_df, text_column='cleaned_text', rating_column=rating_column)
    
    # Step 3: Visualize sentiment
    visualize_sentiment(reviews_df, rating_column=rating_column, product_id_column=product_id_column)
    
    # Step 4: Word frequency analysis
    word_counts, word_counts_by_sentiment = analyze_word_frequency(
        reviews_df, text_column='cleaned_text', by_sentiment=True
    )
    
    # Step 5: Visualize word frequency
    visualize_word_frequency(word_counts, word_counts_by_sentiment)
    
    # Step 6: Topic modeling
    vectorizer, lda_model, feature_names, topic_words, document_topics = perform_topic_modeling(
        reviews_df, text_column='cleaned_text', num_topics=5
    )
    
    # Step 7: Visualize topics
    visualize_topics(lda_model, feature_names, topic_words, document_topics, reviews_df)
    
    # Step 8: Extract product insights
    product_insights = extract_product_insights(
        reviews_df, 
        text_column='cleaned_text',
        product_id_column=product_id_column,
        rating_column=rating_column
    )
    
    # Step 9: Create summary report
    analysis_summary = {
        'total_reviews': len(reviews_df),
        'sentiment_distribution': reviews_df['sentiment_category'].value_counts().to_dict(),
        'avg_sentiment_score': reviews_df['sentiment_compound'].mean(),
        'most_common_words': dict(word_counts.most_common(20)),
        'topic_words': topic_words,
        'products_analyzed': len(product_insights) if product_insights is not None else 0
    }
    
    # Return all results
    results = {
        'reviews_df': reviews_df,
        'sentiment_analysis': {
            'distribution': reviews_df['sentiment_category'].value_counts(),
            'avg_compound_score': reviews_df['sentiment_compound'].mean()
        },
        'word_analysis': {
            'overall_counts': word_counts,
            'by_sentiment': word_counts_by_sentiment
        },
        'topic_modeling': {
            'model': lda_model,
            'topics': topic_words,
            'document_topics': document_topics
        },
        'product_insights': product_insights,
        'summary': analysis_summary
    }
    
    print("✅ Complete NLP analysis finished!")
    print(f"📊 Analyzed {len(reviews_df)} reviews")
    print(f"🎯 Identified {len(topic_words)} main topics")
    print(f"📈 Generated insights for {len(product_insights) if product_insights is not None else 0} products")
    
    return results

# Example usage and testing functions
def create_sample_reviews_data(n_reviews=1000):
    """
    Create sample reviews data for testing purposes.
    
    Parameters:
    -----------
    n_reviews : int
        Number of sample reviews to generate
    
    Returns:
    --------
    pandas DataFrame
        Sample reviews dataframe
    """
    print(f"Creating {n_reviews} sample reviews for testing...")
    
    # Sample product IDs
    product_ids = [f'product_{i}' for i in range(1, 21)]  # 20 products
    
    # Sample review templates
    positive_reviews = [
        "Great product! Really love it. Excellent quality and fast shipping.",
        "Amazing quality! Highly recommend this item. Worth every penny.",
        "Perfect! Exactly what I was looking for. Great customer service too.",
        "Outstanding product. Exceeded my expectations. Will buy again.",
        "Fantastic! Great value for money. Very satisfied with purchase."
    ]
    
    neutral_reviews = [
        "Okay product. Does what it's supposed to do. Average quality.",
        "It's fine. Nothing special but works as expected.",
        "Average product. Could be better but not terrible.",
        "Decent quality. Met my basic needs but nothing exceptional.",
        "Fair product. Good for the price but could be improved."
    ]
    
    negative_reviews = [
        "Terrible quality. Broke after one use. Don't waste your money.",
        "Very disappointing. Not as described. Poor customer service.",
        "Awful product. Cheap materials and poor construction.",
        "Waste of money. Doesn't work properly. Very frustrated.",
        "Poor quality. Not worth the price. Looking for refund."
    ]
    
    # Generate sample data
    reviews_data = []
    
    for i in range(n_reviews):
        # Random sentiment
        sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=[0.5, 0.3, 0.2])
        
        # Select review text based on sentiment
        if sentiment == 'positive':
            review_text = np.random.choice(positive_reviews)
            rating = np.random.choice([4, 5], p=[0.3, 0.7])
        elif sentiment == 'neutral':
            review_text = np.random.choice(neutral_reviews)
            rating = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])
        else:
            review_text = np.random.choice(negative_reviews)
            rating = np.random.choice([1, 2], p=[0.7, 0.3])
        
        reviews_data.append({
            'user_id': f'user_{i}',
            'product_id': np.random.choice(product_ids),
            'review_text': review_text,
            'rating': rating,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        })
    
    return pd.DataFrame(reviews_data)

def run_sample_analysis():
    """
    Run a complete sample analysis for demonstration.
    """
    print("🚀 Running sample NLP analysis...")
    
    # Create sample data
    sample_reviews = create_sample_reviews_data(500)
    
    # Run analysis
    results = analyze_product_reviews(
        sample_reviews,
        text_column='review_text',
        rating_column='rating',
        product_id_column='product_id'
    )
    
    return results

# Main execution
if __name__ == "__main__":
    # Run sample analysis
    sample_results = run_sample_analysis()
    print("Sample analysis complete!")
# NLP Analysis of Product Reviews
# ===============================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec

# Visualization settings
import matplotlib.pyplot as plt
plt.style.use('seaborn-whiteg