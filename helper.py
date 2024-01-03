from scipy.special import softmax   

class helper():
    def polarity_scores(self, example):
        encoded_text = self.tokenizer(example, return_tensors='pt')
        output = self.model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
        'Negative': scores[0],
        'Neutral': scores[1],
        'Positive': scores[2]
        }
        return scores_dict