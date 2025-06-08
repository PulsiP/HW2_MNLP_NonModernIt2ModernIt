from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

"""
    PrometheusEval_AtM
    Evaluation module for translation tasks from Archaic Italian to Modern Italian using the Prometheus Evaluator model.
    The class loads and manages a quantized or full model, and provides a method to generate detailed feedback and a numerical score based on a strict rubric.
"""

class PrometheusEval_AtM:
    # System message defining the assistant's role and specialization.
    ABS_SYSTEM_PROMPT = (
        "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, "
        "ensuring each assessment reflects the absolute standards set for performance, and specialized in evaluating "
        "translations from Old Italian to Modern Italian."
    )

    # Prompt template used for generating evaluation feedback.
    ABSOLUTE_PROMPT = """###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
        3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
        4. Please do not generate any other opening, closing, and explanations.

        ###The source sentence:
        Evaluate how better the model has translated the sentence "{source_sentence}" from Ancient Italian to Modern Italian

        ###Response to evaluate:
        {predicted_sentence}

        ###Reference Answer (Score 5):
        {reference_sentence}

        ###Score Rubrics:
        {rubric}

    ###Feedback: """

    # Evaluation rubric defining scoring criteria from 1 to 5.
    RUBRIC = """
    criteria":"Is the model faithful and accurate in translating sentences into Modern Italian from Archaic Italian?"
    "score1_description":"Completely unacceptable translation: the translation has no pertinence with the original meaning, the generated sentence is either gibberish or something that makes no sense"
    "score2_description":"Severe semantic errors, omissions or substantial add ons on the original sentence. The errors are of semantic and syntactic nature. It’s still something no human would ever write"
    "score3_description":"Partially wrong translation, the translation is lackluster, it contains errors, but are mostly minor errors, like typos, or small semantic errors."
    "score4_description":"Good translation. The translation is mostly right, substantially faithful to the original text, but the style does not perfectly match the original sentence, still fluent and comprehensible, and could semantically acceptable"
    "score5_description":"Perfect translation. The translation is accurate, fluent, complete and oheret. It retained the original meaning as much as it could.."
    """

    def __init__(self, quantized: bool, device: str):
        """
        Initialize the evaluation class with the Prometheus model.

        Args:
            quantized (bool): Whether to load the model in 4-bit quantized mode for memory efficiency.
            device (str): Target device for computation (e.g., 'cuda' or 'cpu').
        """
        model_id = "prometheus-eval/prometheus-7b-v2.0"
        quant_config = None

        if quantized:
            # Configure quantization for 4-bit loading using BitsAndBytes
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16", 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self.device = device

        # Load the Prometheus model with or without quantization
        self.model_ = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config
        )
        self.model_.to(device)

        # Load the associated tokenizer
        self.tokenizer_ = AutoTokenizer.from_pretrained(model_id)

    def getEvaluation(self, source_sentence: str, predicted_sentence: str, reference_sentence: str):
        """
        Generate a structured evaluation for a given translation task.

        Args:
            source_sentence (str): The original sentence in Archaic Italian.
            predicted_sentence (str): The model-generated translation in Modern Italian.
            reference_sentence (str): The human-quality reference translation used as a gold standard.

        Returns:
            str: A string containing feedback and a numerical score in the format:
                 "Feedback: ... [RESULT] x"
        """
        
        # Format system and user messages for the chat template
        sys_content = self.ABS_SYSTEM_PROMPT
        user_content = self.ABSOLUTE_PROMPT.format(
            source_sentence=source_sentence,
            predicted_sentence=predicted_sentence,
            reference_sentence=reference_sentence,
            rubric=self.RUBRIC
        )

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

        # Tokenize the input prompt
        encodeds = self.tokenizer_.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)

        # Generate output from the model
        generated_ids = self.model_.generate(
            model_inputs, 
            max_new_tokens=1000, 
            do_sample=True
        )

        # Decode the output text
        decoded = self.tokenizer_.batch_decode(generated_ids)

        # Return the full raw output (can be refined to extract just feedback if needed)
        return decoded[0]




        