import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from datasets import load_dataset
import time
import openai
from openai import OpenAI
from utils import Generate_Counterfactuals

client = OpenAI(api_key="YOUR_API_KEY_HERE")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Counterfactual Explanations for a Dataset")

    parser.add_argument("--return_dataset",
                        type=str,
                        required=True,
                        choices=["clean", "cfx", "pair_cfx"],
                        help="Specify whether to return the dataset as 'clean' (original) or 'cfx' (with counterfactuals)")


    parser.add_argument("--method",
                        type=str,
                        default="llm-prompt",
                        help="The method to use for distillation")
    

    parser.add_argument("--dataset_name", 
                        type=str, 
                        required=True, 
                        choices=["sst2", 
                                 "cola", 
                                 "imdb", 
                                 "sentiment140",
                                 "amazon",
                                 "yelp"], 
                        help="The name of the dataset to use")
    
    
    parser.add_argument("--model", 
                        default="gpt-4o-mini",
                        choices=["gpt-4o", "gpt-4o-mini"],
                        type=str, 
                        help="The name of the model to use")
    
    parser.add_argument("--subset_size",
                            type=int,
                            default=None,
                            help="The size of the subset to use. If not provided, the entire dataset will be used")
    
    parser.add_argument("--save",
                        type=bool,
                        default=True,
                        help="Whether to save the generated data")
    
    parser.add_argument("--output_dir", 
                        type=str, 
                        required=True,  
                        help="The directory to save the output")
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():
    args = parse_args()


    if args.return_dataset == "cfx":

        cf_generator = Generate_Counterfactuals(args)
        cfx_dataset = cf_generator.generate_counterfactuals()
        
        print("Counterfactual dataset generated successfully.")


    elif args.return_dataset == "clean":

        cf_generator = Generate_Counterfactuals(args)
        clean_dataset = cf_generator.generate_data()
        print("Clean dataset generated successfully.")
    

    elif args.return_dataset == "pair_cfx":

        cf_generator = Generate_Counterfactuals(args)
        pair_cfx_dataset = cf_generator.generate_counterfactuals('pair')
        print("Pair counterfactual dataset generated successfully.")
    else:
        raise ValueError("Invalid value for --return_dataset.")

if __name__ == "__main__":
    main()







