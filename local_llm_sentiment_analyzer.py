import pandas as pd
from langchain_ollama import OllamaLLM
import json
import re


class SentimentAnalyzer:
    def __init__(self, corpus_file, output_file="sentiment_analysis_results_ollama.json"):
        self.corpus_file = corpus_file
        self.output_file = output_file
        self.corpus_df = None
        self.corpus_sentiment_analysis = {}
        self.ollama = None

    def load_corpus(self):
        """Loads the review corpus from a CSV file."""
        self.corpus_df = pd.read_csv(self.corpus_file)
        print("Corpus loaded successfully.")
        print(self.corpus_df.head())

    def initialize_model(self):
        """Initialize the Ollama model if not already done."""
        if self.ollama is None:
            self.ollama = OllamaLLM(model='llama3.1:8b')
            print("Ollama model initialized.")

    def query_ollama(self, prompt):
        """Query the local Ollama model."""
        self.initialize_model()

        response = self.ollama.invoke(prompt)

        return response

    def aspect_based_sentiment_analysis(self, review_title, review_body):
        """Performs aspect-based sentiment analysis on a review title and body."""
        prompt = f"""
        Review Title: {review_title}
        Review Body: {review_body}

        Perform aspect-based sentiment analysis for both the review title and body. Identify sentiment words (must be 
        verbs or adjectives and link them to aspect words (must be nouns). Do not invent aspect words, and omit any 
        sentiment without a noun as its aspect. If there are no sentiments in the review, return 'neutral' for the 
        overall score and leave the block empty.

        The response must be valid JSON. Only use 'positive', 'negative', or 'neutral' for sentiment values. No 
        explanations, notes or comments. The JSON structure should be as follows:
        
        {{
            "reviewTitle": "{review_title}",
            "reviewBody": "{review_body}",
            "sentimentTitle": {{
                "overall_score": "[positive/neutral/negative]",
                "sentiments": [
                    {{
                        "sentiment_word": "[Sentiment Word (Verb or Adjective)]",
                        "aspect_word": "[Aspect Word (Noun) or Compound Aspect Word (Nouns)]",
                        "aspect_sentiment": "[positive/neutral/negative]"
                    }}
                ]
            }},
            "sentimentBody": {{
                "overall_score": "[positive/neutral/negative]",
                "sentiments": [
                    {{
                        "sentiment_word": "[Sentiment Word (Verb or Adjective)]",
                        "aspect_word": "[Aspect Word (Noun) or Compound Aspect Word (Nouns)]",
                        "aspect_sentiment": "[positive/neutral/negative]"
                    }}
                ]
            }}
        }}
        """

        response = self.query_ollama(prompt)

        return response

    def structure_review_analysis(self, review_title, review_body, analysis_result, index):
        """Extracts JSON from the model response and structures the sentiment analysis results."""
        json_match = re.search(r'\{[\S\s]*\}', analysis_result, re.DOTALL)

        if json_match:
            json_string = json_match.group(0)
            print("Response String:", json_string)
            try :
                json_data = json.loads(json_string)
            except json.JSONDecodeError:
                print("Error decoding JSON.")
                return None

            structured_review = {
                "index": index,
                "reviewTitle": json_data.get("reviewTitle", review_title),
                "reviewBody": json_data.get("reviewBody", review_body),
                "sentimentTitle": {
                    "overall_score": json_data.get("sentimentTitle", {}).get("overall_score", "neutral"),
                    "sentiments": json_data.get("sentimentTitle", {}).get("sentiments", [])
                },
                "sentimentBody": {
                    "overall_score": json_data.get("sentimentBody", {}).get("overall_score", "neutral"),
                    "sentiments": json_data.get("sentimentBody", {}).get("sentiments", [])
                }
            }

            return structured_review

    def analyze_corpus(self):
        """Analyzes each review in the corpus."""
        for index, row in self.corpus_df.iterrows():
            review_id = str(index)
            review_title = row["reviewTitle"]
            review_body = row["reviewBody"]

            # Perform aspect-based sentiment analysis
            analysis_result = self.aspect_based_sentiment_analysis(review_title, review_body)

            # Structure the sentiment analysis result
            structured_review = self.structure_review_analysis(review_title, review_body, analysis_result, index)

            # Store the structured review analysis
            if structured_review:
                self.corpus_sentiment_analysis[review_id] = structured_review

    def save_results(self):
        """Saves the sentiment analysis results to a JSON file."""
        with open(self.output_file, "w") as outfile:
            json.dump(self.corpus_sentiment_analysis, outfile, indent=4)
        print(f"Sentiment analysis results saved to {self.output_file}")

    def run(self):
        """Main method to load the corpus, analyze it, and save results."""
        self.load_corpus()
        self.analyze_corpus()
        self.save_results()


def main():
    sentiment_analyzer = SentimentAnalyzer("test.csv")
    # sentiment_analyzer = SentimentAnalyzer("SentimentAssignmentReviewCorpus.csv")
    sentiment_analyzer.run()


if __name__ == "__main__":
    main()
