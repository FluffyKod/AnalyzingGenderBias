"""
The program which trains a neural network on the data found in aclimdb folder.

IMPORTANT: The model has already been trained and is found under neural_sentiment_model. Howevever, if you would like to train a model for yourself, you must extract the aclimdb.zip folder to get access to the training data, or use training data of your own!
"""

##################################
# LIBRARIES
##################################

import os
import random
import spacy
from spacy.util import minibatch, compounding
import pandas as pd

##################################
# TRAIN MODEL
##################################

def train_model(model_name, training_data, test_data, epochs=20):
    """
    Takes in a corpus of training data and trains a neural network.
    """

    # load spacy model
    nlp = spacy.load("en_core_web_sm")

    # Build pipeline

    # check if the text cateogirser pipe is not added
    if "textcat" not in nlp.pipe_names:
        # if not, create it
        textcat = nlp.create_pipe(
            # change to other network
            "textcat", config={"architecture": "ensemble"} # or simple_cnn
        )

        # addd pipe to pipeline
        nlp.add_pipe(textcat, last=True)
    else:
        # if already created, just get the pipe
        textcat = nlp.get_pipe("textcat")

    # add labels to look for
    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]

    # train textcat
    with nlp.disable_pipes(training_excluded_pipes):

        # create an optimizer wrapper
        optimizer = nlp.begin_training()

        # Training loop
        print("Training begins...")
        print("Loss\tPrecision\tRecall\tF-score")

        # create different size batches to train with
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )

        # go through training data multiple times
        for i in range(epochs):
            # keep track of current epoch
            print(f"Epoch: {i}")

            # keep track of loss
            loss = {}

            # shuffle data to avoid overfitting certain patterns
            random.shuffle(training_data)

            # get a batch of training samples
            batches = minibatch(training_data, size=batch_sizes)

            # train model on data
            for batch in batches:
                # get the text and the correct classification
                text, labels = zip(*batch)

                # update the network based on the text
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)

            # evaluate model
            with textcat.model.use_params(optimizer.averages):
                # get evaluation
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data,
                )

                # print the
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model for later use
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(model_name)

##################################
# EVALUATE MODEL
##################################

def evaluate_model(tokenizer, textcat, test_data):
    """
    Evaluates a neural network categorizer with test data.
    """

    # extract data from testing data
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)

    # keep track of results
    true_positives = 0
    false_positives = 1e-8
    true_negatives = 0
    false_negatives = 1e-8

    # go through every review
    for i, review in enumerate(textcat.pipe(reviews)):
        # get the actual classification
        true_label = labels[i]["cats"]

        # go through prediction
        for predicted_label, score in review.cats.items():
            # only consider psotive labels
            if predicted_label == "neg":
                continue

            # correctly positive classified
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1

            # falsly positive classified
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1

            # correctly negative classified
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1

            # falsly negative classified
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1

    # calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # calculate f_score
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    # return results
    return {"precision": precision, "recall": recall, "f-score": f_score}

##################################
# TEST MODEL
##################################

def test_model(input_data):
    """
    Test model on input.
    """

    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")

    # Generate prediction
    parsed_text = loaded_model(input_data)

    # Determine prediction
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]

    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )

##################################
# LOAD TRAINING DATA
##################################

def load_training_data(data_directory="aclImdb/train", split=0.8, limit=0):
    """
    Parses training data from directory.
    """

    # Keep track of all training data entries
    reviews = []

    # get all positive and negative reviews respectively
    for label in ["pos", "neg"]:
        # only get one kind of reviews at a time
        labeled_directory = f"{data_directory}/{label}"

        # go through every review in specified directory
        for review in os.listdir(labeled_directory):

            # check if text file
            if review.endswith(".txt"):

                # open file
                with open(f"{labeled_directory}/{review}") as f:
                    # read and clean data
                    text = f.read()
                    text = text.replace("<br />", "\n\n")

                    # if review contains text
                    if text.strip():
                        # set correct classification
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label,
                            }
                        }

                        # add classified review to array
                        reviews.append((text, spacy_label))

    # ranomly scramble all reviews
    random.shuffle(reviews)

    # if an upper limit specified, only use a certain number of reviews
    if limit:
        reviews = reviews[:limit]

    # split data into training data and testing data
    split = int(len(reviews) * split)

    # return training data and testing data
    return reviews[:split], reviews[split:]

##################################
# MAIN
##################################

def main():
    """
    Load training data, runs the training algorithm and outputs the results.
    """

    # get training and testing data
    train, test = load_training_data(limit=3000)

    # set model name
    MODEL_NAME = 'neural_sentiment_model'

    # train model on testing data and evaluate over time
    print("Training model...")
    train_model(MODEL_NAME, train, test)

    # optional: test model with one example
    # test_model()


if __name__ == "__main__":
    main()
