from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


"""
    prometheus eval for ancient italiano to modern italian sentences
"""
class PrometheusEval_AtM:
    ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance, and specialized in evaluating translations from Old Italian to Modern Italian."

    ABSOLUTE_PROMPT = """###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
        3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
        4. Please do not generate any other opening, closing, and explanations.

        ###The source sentence to evaluate:
        {source_sentence}

        ###Response to evaluate:
        {predicted_sentence}

        ###Reference Answer (Score 5):
        {reference_sentence}

        ###Score Rubrics:
        {rubric}

    ###Feedback: """

    RUBRIC = """
    criteria":"Is the model faithful and accurate in translating sentences into Modern Italian from Archaic Italian?"
    "score1_description":"Completely unacceptable translation: the translation has no pertinence with the original meaning, the generated sentence is either gibberish or something that makes no sense"
    "score2_description":"Severe semantic errors, omissions or substantial add ons on the original sentence. The errors are of semantic and syntactic nature. It’s still something no human would ever write"
    "score3_description":"Partially wrong translation, the translation is lackluster, it contains errors, but are mostly minor errors, like typos, or small semantic errors."
    "score4_description":"Good translation. The translation is mostly right, substantially faithful to the original text, but the style does not perfectly match the original sentence, still fluent and comprehensible, and could semantically acceptable"
    "score5_description":"Perfect translation. The translation is accurate, fluent, complete and oheret. It retained the original meaning as much as it could.."
    """
    
    def __init__(self, quantized:bool, device:str):
    
        model_id = "prometheus-eval/prometheus-7b-v2.0"
        quant_config = None

        if quantized:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16", 
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"  
            )

        self.device = device

        self.model_ = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  
            quantization_config=quant_config
        )
        self.model_.to(device)

        self.tokenizer_ = AutoTokenizer.from_pretrained(model_id)
    
    def getEvaluation(self,
                      source_sentence:str, 
                      predicted_sentence:str, 
                      reference_sentence:str):
        

        sys_content = self.ABS_SYSTEM_PROMPT
        user_content = self.ABSOLUTE_PROMPT.format(source_sentence=source_sentence,
                                                    predicted_sentence=predicted_sentence,
                                                    reference_sentence=reference_sentence,
                                                    rubric=self.RUBRIC) 
        

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]

        encodeds = self.tokenizer_.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model_.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer_.batch_decode(generated_ids)


        return decoded[0]



        