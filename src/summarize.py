from typing import List
from collections import defaultdict
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    BartTokenizer,
    BartForConditionalGeneration,
)
from os import environ as env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained(env["MODEL"])
tokenizer = BartTokenizer.from_pretrained(env["TOKENIZER"])
model.to(device)

def summarize_text(text: str, summary_length: int, exclude_words: List[str]) -> str:
    """
    Generate a summary for the given text while applying constraints on the summary generation.

    Args:
        text (str): The input text to be summarized.
        summary_length (int): The desired length (in tokens) of the summary.
        exclude_words (List[str]): A list of words to be excluded from the generated summary.

    Returns:
        str: The generated summary for the input text.

    This function generates a summary for the given text by utilizing the `generate_summaries_with_constraints()` function.
    It applies constraints on the summary generation process to ensure that the generated summary does not contain any
    excluded words.

    The function internally calls the `generate_summaries_with_constraints()` function, passing the input text, summary length,
    and excluded words. It returns the generated summary as a string.

    Returns the generated summary for the input text, considering the applied constraints.
    """
    pred_summaries_constrained = generate_summaries_with_constraints(
        model=model,
        tokenizer=tokenizer,
        docs_to_summarize=[text],
        word_validator=WordValidator(exclude_words=exclude_words),
        num_beams=6,
        max_length=summary_length,
        device=device
    )

    return pred_summaries_constrained[0]


class WordValidator:
    """A helper class used by ConsistentLogitsProcessor that maintains a set of excluded words 
    and provides a method to check the validity of a word."""

    def __init__(self, exclude_words: List[str]):
        self.exclude_words = set(exclude_words)

    def is_valid_word(self, word: str) -> bool:
        return word not in self.exclude_words


SPLIT_WORD_TOKENS = {
    " ",
    ".",
    ",",
    "_",
    "?",
    "!",
}


class ConsistentLogitsProcessor(LogitsProcessor):
    """
    Enforces constraints on the logits (scores) during summary generation. This class checks 
    the validity of each generated token and ensures that it does not contain excluded words. 
    It also handles subword splitting to validate complete words.
    """


    def __init__(self, tokenizer: AutoTokenizer, num_beams: int, word_validator: WordValidator):
        self.tokenizer = tokenizer
        self.word_validator = word_validator
        self.num_beams = num_beams
        self.failed_sequences = set()


    def is_valid_beam(
        self,
        sequence,
        token_id,
    ) -> bool:
        """
        Check whether the beam is valid based on the provided constraints.

        Args:
            sequence (List[int]): The sequence generated so far.
            token_id (int): The ID of the next token to be generated.

        Returns:
            bool: True if the beam is valid, False otherwise.

        The method checks the validity of the beam by considering the next token to be generated and the sequence generated so far.
        It ensures that the beam does not contain excluded words and handles subword splitting to validate complete words.

        If the predicted subword (token) ends with a character from SPLIT_WORD_TOKENS, it backtracks through the sequence to
        construct the full word. It then calls the word validator to check whether the constructed word is valid. If it is not valid,
        the method returns False.

        Returns True if the beam is valid and does not violate any constraints, and False otherwise.
        """
        current_subword = self.tokenizer.decode(token_id)
        backtrack_word = ""
        is_subword_ending = False
        for char in current_subword:
            if char in SPLIT_WORD_TOKENS:
                is_subword_ending = True
                break
            else:
                backtrack_word += char

        backtrack_done = False
        if is_subword_ending:
            prev_subword_idx = len(sequence) - 1
            while prev_subword_idx != 0 and not backtrack_done:
                prev_token_id = sequence[prev_subword_idx]
                prev_subword = self.tokenizer.decode(prev_token_id)
                prev_char_idx = len(prev_subword) - 1
                while prev_char_idx >= 0:
                    prev_char = prev_subword[prev_char_idx]
                    if prev_char not in SPLIT_WORD_TOKENS:
                        backtrack_word = prev_char + backtrack_word
                    else:
                        backtrack_done = True
                        break
                    prev_char_idx -= 1
                prev_subword_idx -= 1

            # Call validator to check whether the word is valid. Return False
            # if it is not valid
            if not self.word_validator.is_valid_word(backtrack_word):
                return False

        return True


    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process the logits (scores) for a given input and modify them based on the defined constraints.

        Args:
            input_ids (torch.LongTensor): The input token IDs.
            scores (torch.FloatTensor): The logits (scores) associated with the input.

        Returns:
            torch.FloatTensor: The processed logits (scores) after applying the constraints.

        This method is called to process the logits (scores) generated by the model. It iterates over each beam's scores and
        determines if the beam violates any constraints. If a beam contains an invalid word or violates the specified constraints,
        its score is set to negative infinity.

        The method utilizes the `is_valid_beam` method to check the validity of a beam based on the generated sequence and the
        next token to be generated. If the beam is determined to be invalid, the corresponding score is set to negative infinity.

        The method also keeps track of the failed sequences (inputs) where all beams are blocked by the constraints.

        Returns the modified logits (scores) after applying the constraints.
        """
        blocked_beams_by_input_idx = defaultdict(lambda: 0)
        for beam_idx, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            top_k = beam_scores.topk(k=5)
            for prob, idx in zip(top_k[0], top_k[1]):
                input_idx = beam_idx // self.num_beams
                if not self.is_valid_beam(beam_input_ids, idx.item()):
                    beam_scores[idx] = float("-inf")
                    blocked_beams_by_input_idx[input_idx] += 1

        for input_idx, n_blocked in blocked_beams_by_input_idx.items():
            if n_blocked == self.num_beams:
                self.failed_sequences.add(input_idx)

        return scores


def generate_summaries_with_constraints(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    docs_to_summarize: List[str],
    word_validator: WordValidator,
    num_beams: int = 4,
    max_length: int = 150,
    device: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> List[str]:
    """
    Generate summaries for a given list of documents while applying specified constraints.

    Args:
        model (AutoModelForSeq2SeqLM): The pre-trained model for sequence-to-sequence language modeling.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        docs_to_summarize (List[str]): The list of documents to be summarized.
        word_validator (WordValidator): An instance of the WordValidator class to validate words against constraints.
        num_beams (int, optional): The number of beams to be used in beam search during summary generation. Defaults to 4.
        max_length (int, optional): The maximum length (in tokens) of the generated summaries. Defaults to 150.
        device (torch.device, optional): The device on which to perform the summary generation. Defaults to GPU if available, otherwise CPU.

    Returns:
        List[str]: A list of generated summaries for the input documents.

    This function generates summaries for a given list of documents using the provided pre-trained model and tokenizer.
    It applies constraints on the generated summaries by utilizing the `ConsistentLogitsProcessor` class.

    The input documents are first tokenized and processed using the tokenizer. The input token IDs are then passed to the model
    for summary generation. During generation, the `ConsistentLogitsProcessor` is used to enforce constraints on the logits (scores)
    to ensure the generated summaries adhere to the specified constraints.

    The function handles both successful and failed generations. If a summary fails to satisfy the constraints and all beams are blocked,
    a specific failure message is included in the generated summaries.

    Returns a list of generated summaries for the input documents, considering the applied constraints.
    """
    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    input_token_ids = inputs.input_ids.to(device)
    consistency_forced = ConsistentLogitsProcessor(tokenizer, num_beams, word_validator)
    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
        logits_processor=LogitsProcessorList([consistency_forced]),
        max_length=max_length,
    )

    generated_summaries = [
        (
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if idx not in consistency_forced.failed_sequences
            else "<Failed generation: blocked all beams>"
        )
        for idx, ids in enumerate(model_output.sequences)
    ]
    return generated_summaries
