import pandas as pd
import re
import spacy
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TextCleaner:
    @staticmethod
    def clean_text(text):
        if isinstance(text, str):
            text = text.replace('\u2019', "'")
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ''


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_words = set(self.analyzer.lexicon.keys())  # Get sentiment words from VADER lexicon
        self.sentiment_scores = {word: score for word, score in self.analyzer.lexicon.items()}

    def classify_sentiment(self, score):
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        else:
            return 'neutral'

    def invert_sentiment(self, sentiment):
        """Invert the sentiment: positive -> negative, negative -> positive, neutral remains the same."""
        if sentiment == 'positive':
            return 'negative'
        elif sentiment == 'negative':
            return 'positive'
        return sentiment


class AspectExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = SentimentAnalyzer()

    def check_for_compound_nouns(self, token_head):
        global aspect
        aspect_words = [token_head.text]

        for child in token_head.children:
            if child.dep_ == 'compound' and child.pos_ == 'NOUN':
                aspect_words.insert(0, child.text)

        aspect = ' '.join(aspect_words)
        return aspect
    
    def check_for_negation(self, doc, token):
        negation_words = {'not', "n't", 'never', 'no'}
        negated = False

        for i in range(1, 4):  # Check the previous 1, 2, and 3 tokens
            if token.i - i >= 0:
                prev_token = doc[token.i - i]
                if prev_token.text in ['.', '!', '?']:
                    break
                # Check if the previous token's text is in negation words
                if prev_token.text in negation_words:
                    negated = not negated
                    break
                # Alternatively, check for substrings
                elif any(negation in prev_token.text for negation in negation_words):
                    negated = not negated
                    break

        return negated

    def extract_sentiment_aspects(self, review):
        doc = self.nlp(review)
        results = []
        aspect_scores = []

        for sent in doc.sents:
            for token in sent:
                negated = False

                # Detect sentiment words (ADJ or VERB)
                if token.text.lower() in self.sentiment_analyzer.sentiment_words:
                    sentiment_phrase = token.text

                    # check for negation before the sentiment word
                    negated = self.check_for_negation(doc, token)

                    # Preceding ADV modifies the sentiment word (e.g., "very bad")
                    if token.i > 0 and doc[token.i - 1].pos_ == 'ADV':
                        sentiment_phrase = doc[token.i - 1].text + ' ' + token.text

                    aspects = []
                    if token.pos_ == 'ADJ':
                        # Check if the adjective is a modifier of a noun e.g. 'beautiful'
                        if token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
                            aspects.append(self.check_for_compound_nouns(token.head))
                        
                        # Check if the adjective is a complement of the verb e.g. 'The phone is great'
                        elif token.dep_ == 'acomp':
                            # Find the verb head of the adjective
                            head_verb = token.head
                            # Check if the verb has a subject or object that can be the aspect
                            for child in head_verb.children:
                                if child.dep_ == ('nsubj' or child.dep_ == 'dobj') and child.pos_ == 'NOUN':
                                    aspects.append(self.check_for_compound_nouns(child))

                        elif token.dep_ == 'ROOT': 
                            for child in token.children:
                                if child.dep_ == 'prep':
                                    for ix, grandchild in enumerate(child.children):
                                        if grandchild.dep_ == 'pobj' and list(child.children)[ix -1].dep_ != 'poss':
                                            aspects.append(self.check_for_compound_nouns(grandchild))

                        sentiment_score = self.sentiment_analyzer.sentiment_scores.get(token.lower_, 0)
                        aspect_sentiment = self.sentiment_analyzer.classify_sentiment(sentiment_score)

                        # Apply negation if necessary
                        if negated:
                            aspect_sentiment = self.sentiment_analyzer.invert_sentiment(aspect_sentiment)
                            sentiment_phrase = sentiment_phrase + ' ' + '(neg)'
                            sentiment_score = -sentiment_score

                        if aspects:
                            for aspect in aspects:
                                aspect_scores.append(sentiment_score)
                                results.append({
                                    'sentiment_word': sentiment_phrase,
                                    'aspect_word': aspect,
                                    'context': sent.text.strip(),
                                    'aspect_sentiment': aspect_sentiment
                                })

                    # Handle verbs for aspect detection (VERB -> NOUN as dobj or nsubj)
                    elif token.pos_ == 'VERB':
                        # Detect aspects related to the verb
                        aspects = [self.check_for_compound_nouns(child) for child in token.children if child.dep_ in ['dobj', 'nsubj'] and child.pos_ == 'NOUN']

                        if not aspects:
                            # Check prepositional phrases (e.g., "with great features")
                            aspects = [self.check_for_compound_nouns(child) for child in token.children if child.dep_ == 'prep' and child.pos_ == 'NOUN']

                        sentiment_score = self.sentiment_analyzer.sentiment_scores.get(token.lower_, 0)
                        aspect_sentiment = self.sentiment_analyzer.classify_sentiment(sentiment_score)

                        # check for negation before the sentiment word
                        negated = self.check_for_negation(doc, token)
                        if negated:
                            aspect_sentiment = self.sentiment_analyzer.invert_sentiment(aspect_sentiment)
                            sentiment_phrase = sentiment_phrase + ' ' + '(neg)'
                            sentiment_score = -sentiment_score

                        # Log the results for verb-related aspects
                        if aspects:
                            for aspect in aspects:
                                aspect_scores.append(sentiment_score)
                                results.append({
                                    'sentiment_word': sentiment_phrase,
                                    'aspect_word': aspect,
                                    'context': sent.text.strip(),
                                    'aspect_sentiment': aspect_sentiment
                                })

        return results, aspect_scores



class ReviewProcessor:
    def __init__(self, file_path):
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            raise

        self.text_cleaner = TextCleaner()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aspect_extractor = AspectExtractor()

    def process_reviews(self):
        structured_results = {}

        # Clean text
        self.data['reviewTitle'] = self.data['reviewTitle'].apply(self.text_cleaner.clean_text)
        self.data['reviewBody'] = self.data['reviewBody'].apply(self.text_cleaner.clean_text)

        # Process each review
        for index, row in self.data.iterrows():
            review_id = str(index + 1)  # Use 1-based index as the key
            title_results, title_scores = self.aspect_extractor.extract_sentiment_aspects(row['reviewTitle'])
            body_results, body_scores = self.aspect_extractor.extract_sentiment_aspects(row['reviewBody'])

            # Calculate overall scores from aspect scores
            overall_title_score = self.calculate_overall_score(title_scores, 'reviewTitle', row)
            overall_body_score = self.calculate_overall_score(body_scores, 'reviewBody', row)

            structured_results[review_id] = {
                "reviewTitle": row['reviewTitle'],
                "reviewBody": row['reviewBody'],
                "sentimentTitle": {
                    "overall_score": overall_title_score,
                    "sentiments": title_results
                },
                "sentimentBody": {
                    "overall_score": overall_body_score,
                    "sentiments": body_results
                }
            }

        return structured_results

    def calculate_overall_score(self, scores, column, row):
        """Calculate the overall sentiment score based on aspect scores."""
        if not scores or len(scores) == 1:
            return self.sentiment_analyzer.classify_sentiment(self.sentiment_analyzer.analyzer.polarity_scores(row[column])['compound'])

        average_score = sum(scores) / len(scores)
        return self.sentiment_analyzer.classify_sentiment(average_score)

    def save_results(self, results, output_file):
        try:
            with open(output_file, 'w') as json_file:
                json.dump(results, json_file, indent=4)
        except Exception as e:
            raise


if __name__ == "__main__":
    review_processor = ReviewProcessor('test.csv')
    # review_processor = ReviewProcessor('SentimentAssignmentReviewCorpus.csv')
    results = review_processor.process_reviews()
    review_processor.save_results(results, 'sentiment_analysis_results.json')