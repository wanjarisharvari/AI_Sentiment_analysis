from django.db import models

class SentimentAnalysis(models.Model):
    text = models.TextField(null=False, blank=False)
    cleaned_text = models.TextField(null=True)
    predicted_sentiment = models.CharField(max_length=20, null=True)
    comment = models.TextField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"{self.text[:20]} - {self.predicted_sentiment}"
