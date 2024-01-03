# IMPORTS
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import StringVar
from helper import helper
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch


class LiveSentiment():
    def __init__(self):
        # Set up GUI
        self.root = tk.Tk()
        self.root.title('Live Sentiment')
        self.root.geometry('500x500')

        self.scores = None
        self.sentiment = None
        self.MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)

        # Set each sentiment to a variable that can be changed.
        self.neg = StringVar()
        self.neu = StringVar()
        self.pos = StringVar()
        self.sent = StringVar()

        # Initialize tkinter labels & pack
        self.label_neg = tk.Label(self.root, 
                                  textvariable = self.neg, 
                                  font=('Arial', 20)).pack()
        self.label_neu = tk.Label(self.root, 
                                  textvariable = self.neu, 
                                  font=('Arial', 20)).pack()
        self.label_pos = tk.Label(self.root, 
                                  textvariable = self.pos, 
                                  font=('Arial', 20)).pack()
        self.label_sent = tk.Label(self.root, 
                                   textvariable = self.sent, 
                                   font=('Arial', 20)).pack()

        # Initialize text box for user input
        self.text = ScrolledText(self.root, 
                                 font=('Arial', 14))
        
        # Bind release of key (space) to check function
        self.text.bind('<KeyRelease>', 
                       self.check)
        self.text.pack()

        self.old_spaces = 0
        self.root.mainloop()

    # Check function to be executed after a space bar release
    def check(self, _):
        content = self.text.get('1.0', 
                                tk.END)
        space_count = content.count(' ')

        if space_count != self.old_spaces:
            self.old_spaces = space_count
            
            # Calculate polarity of content in the text box and update sentiments
            self.scores = helper.polarity_scores(self, content)
            self.neg.set(f'Negative: {np.round(float(self.scores["Negative"]), 2)*100}%')
            self.neu.set(f'Neutral: {np.round(float(self.scores["Neutral"]), 2)*100}%')
            self.pos.set(f'Positive: {np.round(float(self.scores["Positive"]), 2)*100}%')

            # Get overall sentiment
            self.sent.set(f'Sentiment: {max(self.scores, key=self.scores.get)}')

LiveSentiment()