from transformers import pipeline

class CryptoSentimentEngine:
    """
    Usage FinBERT model, classify text into positive, negetive or nautral 
    
    """

    def __init__(self):
        # load the prosAI/Finbert model using HF pipeline
        # pytorch is used as the underline engine 
        
        # Model is around 400MB
        self.analyzer = pipeline("sentiment-analysis", model ="ProsusAI/finbert")

        # define mapping to convert text labels into numaric scores 
        self.sentiment_map={
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }


    def analyse_text(self, headline:list[str]) -> float:
        # catch empty 
        if not headline:
            return 0.0

        # piprline processes the entire list of Strings in one batch 
        results = self.analyzer(headline)

        scores =[]

        # loop through theoutput of each headline 
        for result in results:
            lable = result['label']
            confidence = result['score']

            # multipy thr direction by confidential presentage 

            polarity = self.sentiment_map.get(lable, 0.0)
            weighted_score = polarity * confidence
            scores.append(weighted_score)

        # Retur the average sentiment across all parsed news items 
        return sum(scores) / len(scores)

    