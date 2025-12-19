from langchain_core.prompts import PromptTemplate


prompt_templates = {
        "sst2": {
            "sys": "",
            "user": PromptTemplate.from_template(
                """You are an AI assistant tasked with generating counterfactuals for sentiment analysis.\n
                    Given a sentence and its true sentiment label, your goal is to make the minimal necessary change to flip the sentiment while preserving the structure and meaning as much as possible.\n\n
                    For example, if the input is:\n
                    Sentence: "I love this movie."\n
                    True sentiment: Positive\n
                    A suitable counterfactual would be: "I dislike this movie."\n\n
                    Now, generate a counterfactual for the following sentence:\n\n
                    Sentence: "{sentence}"\n
                    True sentiment: {sentiment}\n
                    Return only the counterfactual sentence, without any additional information."""
            ),
        },

        "cola": {
            "sys": "",
            "user": PromptTemplate.from_template(
                """You are an AI assistant tasked with generating counterfactuals for grammaticality judgment.\n
                    Given a sentence and its true grammaticality label (Acceptable or Unacceptable), your goal is to make the minimal necessary change to flip the grammaticality while preserving the structure and meaning as much as possible.\n\n
                    For example, if the input is:\n
                    Sentence: "She is going to the store."\n
                    True grammaticality: Acceptable\n
                    A suitable counterfactual would be: "She is go to the store."\n\n
                    Now, generate a counterfactual for the following sentence:\n\n
                    Sentence: "{sentence}"\n
                    True grammaticality: {sentiment}\n
                    Return only the counterfactual sentence, without any additional information."""
            ),
        },

        "imdb": {
            "sys": "",
            "user": PromptTemplate.from_template(
                """You are an AI assistant tasked with generating counterfactuals for sentiment analysis.\n
                    Given a sentence and its true sentiment label, your goal is to make the minimal necessary change to flip the sentiment while preserving the structure and meaning as much as possible.\n\n
                    For example, if the input is:\n
                    Sentence: "I love this movie."\n
                    True sentiment: Positive\n
                    A suitable counterfactual would be: "I dislike this movie."\n\n
                    Now, generate a counterfactual for the following sentence:\n\n
                    Sentence: "{sentence}"\n
                    True sentiment: {sentiment}\n
                    Return only the counterfactual sentence, without any additional information."""
            ),
        },

        "sentiment140": {
            "sys": "",
            "user": PromptTemplate.from_template(
                """You are an AI assistant tasked with generating counterfactuals for sentiment analysis.\n
                    Given a sentence and its true sentiment label, your goal is to make the minimal necessary change to flip the sentiment while preserving the structure and meaning as much as possible.\n\n
                    For example, if the input is:\n
                    Sentence: "I love this movie."\n
                    True sentiment: Positive\n
                    A suitable counterfactual would be: "I dislike this movie."\n\n
                    Now, generate a counterfactual for the following sentence:\n\n
                    Sentence: "{sentence}"\n
                    True sentiment: {sentiment}\n
                    Return only the counterfactual sentence, without any additional information."""
            ),
        },

        "yelp": {
            "sys": "",
            "user": PromptTemplate.from_template(
                """You are an AI assistant tasked with generating counterfactuals for sentiment analysis of Yelp reviews.\n
                Given a sentence (a Yelp review) and its true sentiment label (positive or negative), your goal is to make the minimal necessary change to flip the sentiment while preserving the structure and meaning as much as possible.\n\n
                For example, if the input is:\n
                Sentence: "This restaurant is fantastic, the food was amazing!"\n
                True sentiment: Positive\n
                A suitable counterfactual would be: "This restaurant is terrible, the food was awful!"\n\n
                Now, generate a counterfactual for the following sentence:\n\n
                Sentence: "{sentence}"\n
                True sentiment: {sentiment}\n
                Return only the counterfactual sentence, without any additional information."""
            ),
        },

        "amazon": {
            "sys": "",
            "user": PromptTemplate.from_template(
                """You are an AI assistant tasked with generating counterfactuals for sentiment analysis.\n
                    Given a sentence and its true sentiment label, your goal is to make the minimal necessary change to flip the sentiment while preserving the structure and meaning as much as possible.\n\n
                    For example, if the input is:\n
                    Sentence: "I love this movie."\n
                    True sentiment: Positive\n
                    A suitable counterfactual would be: "I dislike this movie."\n\n
                    Now, generate a counterfactual for the following sentence:\n\n
                    Sentence: "{sentence}"\n
                    True sentiment: {sentiment}\n
                    Return only the counterfactual sentence, without any additional information."""
            ),
        },
        
    }




prompt_templates_sst2_variants  = {
    "sst2_variant_1": {
        "sys": "",
        "user": PromptTemplate.from_template(
            """You're helping improve a sentiment analysis model by generating counterfactual examples.\n
               Given a sentence and its original sentiment, modify it minimally to reverse the sentiment while keeping the rest of the content and style intact.\n\n
               Example:\n
               Original: "The food was amazing!"\n
               Sentiment: Positive\n
               Counterfactual: "The food was terrible!"\n\n
               Now apply the same to:\n
               Sentence: "{sentence}"\n
               Sentiment: {sentiment}\n
               Provide only the revised sentence."""
        ),
    },
    "sst2_variant_2": {
        "sys": "",
        "user": PromptTemplate.from_template(
            """Your task is to rewrite a sentence so that its sentiment is flipped while keeping the overall structure, topic, and wording as close as possible.\n\n
               Example:\n
               Sentence: "This show is fantastic."\n
               Original Sentiment: Positive\n
               Revised Sentence: "This show is awful."\n\n
               Now revise the following:\n
               Sentence: "{sentence}"\n
               Sentiment: {sentiment}\n
               Output only the modified sentence."""
        ),
    },
    "sst2_variant_3": {
        "sys": "",
        "user": PromptTemplate.from_template(
            """Generate a sentiment-flipped version of the following sentence.\n
               Keep your changes minimal — just enough to reverse the sentiment polarity (positive ↔ negative) without altering the core content.\n\n
               Input:\n
               Sentence: "{sentence}"\n
               Sentiment: {sentiment}\n
               Respond only with the counterfactual sentence."""
        ),
    },
    "sst2_variant_4": {
        "sys": "",
        "user": PromptTemplate.from_template(
            """Please produce a counterfactual for the sentiment classification task.\n
               Slightly adjust the sentence so that it reflects the *opposite* sentiment, while maintaining coherence and staying close to the original wording.\n\n
               Example:\n
               "I really enjoyed this experience." → "I really hated this experience."\n\n
               Input sentence: "{sentence}"\n
               Original sentiment: {sentiment}\n
               Return only the transformed sentence."""
        ),
    },
}
