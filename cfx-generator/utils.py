from prompts import prompt_templates
import json
import logging
import math
import os
import random
from pathlib import Path
import time
import openai
from openai import OpenAI
from datasets import concatenate_datasets, Dataset, DatasetDict, load_dataset
from collections import Counter

def label_distribution(dataset):
    """Prints the label distribution and total number of samples for every split in the dataset."""
    for split in dataset.keys():
        print(f"Split: {split}")
        if "label" in dataset[split].column_names:
            labels = dataset[split]["label"]
            label_counts = Counter(labels)
            total_samples = sum(label_counts.values())
            print(f"  Total samples: {total_samples}")
            for label, count in label_counts.items():
                print(f"  Label {label}: {count/total_samples:.2%} ({count} samples)")
        else:
            print("  'label' column not found in this split.")
        print()  # Blank line for readability


class Generate_Counterfactuals:

    def __init__(self, args):
        self.return_dataset = args.return_dataset
        self.method = args.method
        self.dataset_name = args.dataset_name
        self.model = args.model
        self.subset_size = args.subset_size
        self.output_dir = args.output_dir
        self.save = args.save
        self.seed = args.seed

        if self.model == "gpt-4o" or self.model == "gpt-4o-mini":
            self.framework = "openai"
            self.client = OpenAI(api_key="YOUR_API")



    def query(self, user_query, sys_query, temperature=0.0):
        if self.framework == "openai":
                attempt, retries = 0, 6
                resp_on_error = ""
                while attempt < retries:
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": sys_query},
                                {"role": "user", "content": user_query},
                            ],
                            temperature=temperature,
                        )
                        return completion.choices[0].message.content, False
                    # except openai.AuthenticationError as e:
                    #     print("Error: ", e)
                    #     print("Refreshing Token")
                    #     attempt += 1
                    #     self.refresh()
                    #     pass
                    except openai.RateLimitError as e:
                        print("Error: ", e)
                        
                        attempt += 1
                        wait_time = 5 * (2 ** attempt)
                        print(f"Retrying in {wait_time} seconds")
                        time.sleep(wait_time)
                        pass
                    except Exception as e:
                        print("Error: ", e)
                        return resp_on_error, True
                    
    
    def generate_counterfactuals(self):

        new_data = []  # List to store new counterfactual samples

        raw_datasets = self.get_dataset()

        for i in range(len(raw_datasets["train"])):
            # Show progress bar
            print(f"Processing {i+1}/{len(raw_datasets['train'])}", end='\r')
            sentence = raw_datasets["train"][i]['sentence']
            true_sentiment = "positive" if raw_datasets["train"][i]['label'] == 1 else "negative"

            if self.dataset_name == "cola":
                true_sentiment = "Acceptable" if raw_datasets["train"][i]['label'] == 1 else "Unacceptable"

            counterfactual_sentiment = 0 if raw_datasets["train"][i]['label'] == 1 else 1

            prompt = prompt_templates[self.dataset_name]["user"].format(sentence = sentence, sentiment=  true_sentiment)

            response, error = self.query(prompt, "")
            if error:
                print("Error in generating counterfactual for sentence: ", i)
                continue  # Skip this iteration if there was an error

            new_idx = len(raw_datasets["train"]) + len(new_data)-1  

            # Store new counterfactual data
            new_data.append({'sentence': response, 'label': counterfactual_sentiment, 'idx': new_idx})


        if new_data:
            # Convert new_data to a Dataset
            new_dataset = Dataset.from_dict({key: [d[key] for d in new_data] for key in new_data[0]})

            # Get full feature schema from raw_datasets['train']
            features = raw_datasets['train'].features

            # Ensure new_dataset has the same schema
            new_dataset = new_dataset.cast(features)

            raw_datasets["train"] = concatenate_datasets([raw_datasets['train'], new_dataset])

        print("Finished generating counterfactuals")
        print(raw_datasets)

        if self.save:
            self.save_dataset(raw_datasets, self.output_dir)

        return raw_datasets
    
    def generate_counterfactuals(self, cf_field=None):
        """
        If cf_field is provided (e.g. "counterfactual"), update each training example by adding a new column
        with that key. Otherwise, generate counterfactual samples (new rows) and concatenate them with the original dataset.
        """
        raw_datasets = self.get_dataset()

        if cf_field == "pair":
            # Update each training sample with a new counterfactual column.
            def add_counterfactual(example):
                true_sentiment = "positive" if example['label'] == 1 else "negative"
                prompt = prompt_templates[self.dataset_name]["user"].format(sentence=example["sentence"], sentiment=true_sentiment)
                response, error = self.query(prompt, "")
                if error:
                    print("Error generating counterfactual for sentence idx:", example.get("idx", "N/A"))
                    example["counterfactual"] = ""
                    raise Exception("Error generating counterfactual")
                else:
                    example["counterfactual"] = response
                return example

            updated_train = raw_datasets["train"].map(add_counterfactual)
            raw_datasets["train"] = updated_train
            print("Finished generating counterfactuals column:", cf_field)
        else:
            # Default behavior: generate new counterfactual samples and then concatenate them with original dataset
            new_data = []  # List to store new counterfactual samples
            for i in range(len(raw_datasets["train"])):
                print(f"Processing {i+1}/{len(raw_datasets['train'])}", end='\r')
                sentence = raw_datasets["train"][i]['sentence']
                
                true_sentiment = "positive" if raw_datasets["train"][i]['label'] == 1 else "negative"
                counterfactual_sentiment = 0 if raw_datasets["train"][i]['label'] == 1 else 1

                prompt = prompt_templates[self.dataset_name]["user"].format(sentence=sentence, sentiment=true_sentiment)
                response, error = self.query(prompt, "")
                if error:
                    print("Error in generating counterfactual for sentence:", i)
                    continue

                new_idx = len(raw_datasets["train"]) + len(new_data) - 1  
                new_data.append({'sentence': response, 'label': counterfactual_sentiment, 'idx': new_idx})

            if new_data:
                new_dataset = Dataset.from_dict({key: [d[key] for d in new_data] for key in new_data[0]})
                features = raw_datasets['train'].features
                new_dataset = new_dataset.cast(features)
                raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], new_dataset])
            print("Finished generating counterfactuals (concatenated new samples)")

        print(raw_datasets)

        if self.save:
            self.save_dataset(raw_datasets, self.output_dir)

        return raw_datasets
    
    def generate_data(self):

        raw_datasets = self.get_dataset()

        if self.save:
            self.save_dataset(raw_datasets, self.output_dir)

        return raw_datasets


    
    def save_dataset(self, raw_datasets, dataset_path):
        # Create the directory if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)
        
        # Create a subdirectory with the dataset name and subset size
        subset_dir = os.path.join(dataset_path, f"{self.dataset_name}_{self.subset_size}_seed{self.seed}")
        os.makedirs(subset_dir, exist_ok=True)
        
        # Save the dataset to the subdirectory
        raw_datasets.save_to_disk(subset_dir)


    def get_dataset(self):
        # New branch for Yelp dataset
        if self.dataset_name == "yelp":
            raw_dataset = load_dataset("Yelp/yelp_review_full")
            for split in raw_dataset.keys():
                raw_dataset[split] = raw_dataset[split].filter(lambda x: len(x['text']) <= 250)
            def remap_label(example):
                if example["label"] in [1, 2]:
                    example["label"] = 0
                elif example["label"] in [4, 5]:
                    example["label"] = 1
                return example
            for split in raw_dataset.keys():
                raw_dataset[split] = raw_dataset[split].filter(lambda x: x["label"] != 3)
                raw_dataset[split] = raw_dataset[split].map(remap_label)
            for split in raw_dataset.keys():
                raw_dataset[split] = raw_dataset[split].rename_column("text", "sentence")


            test_dataset = raw_dataset["test"].shuffle(seed=self.seed)
            raw_dataset["validation"] = test_dataset.select(range(1000))
            raw_dataset["test"] = test_dataset.select(range(1000, len(test_dataset)))
            self.dataset = raw_dataset

        elif self.dataset_name == "imdb":
            self.dataset = load_dataset("imdb")
            # Filter out samples with text length greater than 300 across all splits
            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].filter(lambda x: len(x['text']) <= 300)
            
            # Remap "text" column to "sentence" for each split
            for split in list(self.dataset.keys()):
                self.dataset[split] = self.dataset[split].rename_column("text", "sentence")
            
            # Rename splits: rename 'test' to 'validation' and 'unsupervised' to 'test'
            if "test" in self.dataset:
                self.dataset["validation"] = self.dataset.pop("test")
            if "unsupervised" in self.dataset:
                self.dataset["test"] = self.dataset.pop("unsupervised")

        elif self.dataset_name == "sst2" or self.dataset_name == "cola":
            self.dataset = load_dataset("glue", self.dataset_name)

        elif self.dataset_name == "sentiment140":

            self.dataset = load_dataset(self.dataset_name)
            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].rename_column("text", "sentence")
                self.dataset[split] = self.dataset[split].rename_column("sentiment", "label")
                # Filter out neutral examples (label 2)
                self.dataset[split] = self.dataset[split].filter(lambda x: x["label"] != 2)
                # Map positive label from 4 to 1 (negative remains 0)
                self.dataset[split] = self.dataset[split].map(lambda x: {"label": 1} if x["label"] == 4 else x)
                # Remove unwanted columns
                self.dataset[split] = self.dataset[split].remove_columns(
                    [col for col in self.dataset[split].column_names if col not in ["sentence", "label"]]
                )
            # Split the training dataset into train and validation sets (e.g., 99% train, 1% validation)
            train_val_split = self.dataset["train"].train_test_split(test_size=0.001, seed=self.seed)
            train_dataset = train_val_split["train"]
            validation_dataset = train_val_split["test"]
            # Rebuild the dataset dictionary
            self.dataset = DatasetDict({
                "train": train_dataset,
                "validation": validation_dataset,
                "test": self.dataset["test"]
            })

        elif self.dataset_name == "sst5":
            raw_dataset = load_dataset("sst", "default")
            def filter_and_remap(example):
                if example["label"] in [0, 2, 4]:
                    return True
                return False

            def remap_label(example):

                label_map = {0: 0, 2: 1, 4: 2}
                example["label"] = label_map[example["label"]]
                return example

            for split in raw_dataset.keys():
                raw_dataset[split] = raw_dataset[split].filter(filter_and_remap)
                raw_dataset[split] = raw_dataset[split].map(remap_label)
                raw_dataset[split] = raw_dataset[split].rename_column("sentence", "sentence")  # Explicit renaming

            self.dataset = raw_dataset
        elif self.dataset_name == "amazon":

            raw_dataset = load_dataset("amazon_polarity")
            # Filter out samples with text length greater than 60 and remove "title" if present
            for split in raw_dataset.keys():
                if "title" in raw_dataset[split].column_names:
                    raw_dataset[split] = raw_dataset[split].remove_columns("title")
                raw_dataset[split] = raw_dataset[split].filter(lambda x: len(x['content']) <= 60)
            # Rename "content" column to "sentence"
            for split in list(raw_dataset.keys()):
                raw_dataset[split] = raw_dataset[split].rename_column("content", "sentence")
            # Rename splits: rename 'test' to 'validation' and 'unsupervised' to 'test'
            if "test" in raw_dataset:
                raw_dataset["validation"] = raw_dataset.pop("test")

            self.dataset = raw_dataset        
    
        # Add idx column if not present in the training set.
        for split in self.dataset.keys():
            if "idx" not in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].map(
                    lambda example, idx: {**example, "idx": idx},
                    with_indices=True
                )
        
        # Always balance the dataset regardless of mode ("clean" or "cfx")
        if self.subset_size:
            # Get indices for each class
            class_0_indices = [i for i, label in enumerate(self.dataset["train"]["label"]) if label == 0]
            class_1_indices = [i for i, label in enumerate(self.dataset["train"]["label"]) if label == 1]
            
            # Shuffle indices with a fixed seed for reproducibility
            random.seed(self.seed)
            random.shuffle(class_0_indices)
            random.shuffle(class_1_indices)
            
            # Calculate the number of samples per class
            if self.return_dataset == "clean":
                num_samples_per_class = self.subset_size // 2
            elif self.return_dataset == "cfx":
                num_samples_per_class = self.subset_size // 4
            elif self.return_dataset == "pair_cfx":
                num_samples_per_class = self.subset_size // 4

            # Select the indices
            selected_indices = class_0_indices[:num_samples_per_class] + class_1_indices[:num_samples_per_class]
            random.shuffle(selected_indices)
            
            # Select the subset of the dataset
            self.dataset["train"] = self.dataset["train"].select(selected_indices)

            # Print class distribution
            label_distribution(self.dataset)

        return self.dataset