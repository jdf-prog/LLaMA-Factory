# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
from typing import TYPE_CHECKING, List, Sequence, Tuple

import difflib
import os
from typing import List
from transformers import AutoTokenizer
from ...extras.packages import is_pillow_available

if is_pillow_available():
    from PIL import Image


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL.Image import Image as ImageObject
    from transformers import ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def get_pixel_values(images: Sequence["ImageObject"], processor: "ProcessorMixin") -> "NDArray":
    r"""
    Processes visual inputs. (currently only supports a single image)
    """
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    image = images[0] if len(images) != 0 else Image.new("RGB", (100, 100), (255, 255, 255))
    return image_processor(image, return_tensors="pt")["pixel_values"][0]  # shape (C, H, W)


def get_paligemma_token_type_ids(input_len: int, processor: "ProcessorMixin") -> List[int]:
    r"""
    Gets paligemma token type ids for computing loss.
    """
    image_seq_length = getattr(processor, "image_seq_length")
    return [0] * image_seq_length + [1] * (input_len - image_seq_length)


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len




default_tokenizer = None
def diff_texts(text1: str, text2: str, tokenizer=None):
    if tokenizer is None:
        global default_tokenizer
        if default_tokenizer is None:
            default_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer = default_tokenizer
    # Encode the input texts to token IDs
    text1_tokens_ids = tokenizer.encode(text1, add_special_tokens=False)
    text2_tokens_ids = tokenizer.encode(text2, add_special_tokens=False)

    # Create a SequenceMatcher object
    matcher = difflib.SequenceMatcher(None, text1_tokens_ids, text2_tokens_ids)
    
    # Get the opcodes (operations) for the differences
    opcodes = matcher.get_opcodes()

    # Process the opcodes to create the merged_tokens list
    merged_tokens = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            merged_tokens.append((tokenizer.decode(text2_tokens_ids[j1:j2]), '+'))
            merged_tokens.append((tokenizer.decode(text1_tokens_ids[i1:i2]), '-'))
        elif tag == 'delete':
            merged_tokens.append((tokenizer.decode(text1_tokens_ids[i1:i2]), '-'))
        elif tag == 'insert':
            merged_tokens.append((tokenizer.decode(text2_tokens_ids[j1:j2]), '+'))
        elif tag == 'equal':
            merged_tokens.append((tokenizer.decode(text1_tokens_ids[i1:i2]), None))
        else:
            raise ValueError(f"Unknown tag: {tag}")
        
        if len(merged_tokens) >= 3 and \
            merged_tokens[-1][1] == None and \
            merged_tokens[-2][1] == '-' and \
            merged_tokens[-3][1] in ['+', None]:
            
            if merged_tokens[-2][0].endswith(merged_tokens[-3][0]):
                token_3 = merged_tokens.pop()
                token_2 = merged_tokens.pop()
                token_1 = merged_tokens.pop()
                
                if token_1[1] == None:
                    merged_tokens.append((token_1[0] + token_2[0][:-len(token_1[0])], token_2[1]))
                elif token_1[1] == '+':
                    merged_tokens.append((token_2[0][:-len(token_1[0])], token_2[1]))
                merged_tokens.append((token_1[0] + token_3[0], token_3[1]))
        elif len(merged_tokens) >= 2:
            if set([merged_tokens[-1][1], merged_tokens[-2][1]]) == set(['+', '-']):
                token_2 = merged_tokens.pop()
                token_1 = merged_tokens.pop()
                common_prefix = os.path.commonprefix([token_1[0], token_2[0]])
                common_suffix = os.path.commonprefix([token_1[0][len(common_prefix):][::-1], token_2[0][len(common_prefix):][::-1]])[::-1]
                if common_prefix:
                    token_1 = (token_1[0][len(common_prefix):], token_1[1])
                    token_2 = (token_2[0][len(common_prefix):], token_2[1])
                if common_suffix:
                    token_1 = (token_1[0][:-len(common_suffix)], token_1[1])
                    token_2 = (token_2[0][:-len(common_suffix)], token_2[1])
                
                if common_prefix:
                    merged_tokens.append((common_prefix, None))
                if token_1[0]:
                    merged_tokens.append(token_1)
                if token_2[0]:
                    merged_tokens.append(token_2)
                if common_suffix:
                    merged_tokens.append((common_suffix, None))

    # Merge adjacent tokens with the same category
    final_merged_tokens = [merged_tokens[0]]
    last_token_category = merged_tokens[0][1]
    for token, category in merged_tokens[1:]:
        if category == last_token_category:
            final_merged_tokens[-1] = (final_merged_tokens[-1][0] + token, category)
        else:
            final_merged_tokens.append((token, category))
        last_token_category = category
    return final_merged_tokens


def compare_lists(list1, list2):
    # Find common prefix
    prefix = []
    for x, y in zip(list1, list2):
        if x == y:
            prefix.append(x)
        else:
            break
    
    # Find common suffix
    suffix = []
    for x, y in zip(reversed(list1), reversed(list2)):
        if x == y:
            suffix.insert(0, x)  # Insert at the beginning to maintain order
        else:
            break
    
    return prefix, suffix

def diff_token_ids(text1_tokens_ids: List[int], text2_tokens_ids: List[int]):
    """
    Args:
        text1_tokens_ids: List of token IDs for the first text.
        text2_tokens_ids: List of token IDs for the second text.
    Returns:
        List of tuples, where each tuple contains a list of token ids and a category.
        The category is one of '+', '-', or None
    """
    # Create a SequenceMatcher object
    matcher = difflib.SequenceMatcher(None, text1_tokens_ids, text2_tokens_ids)
    
    # Get the opcodes (operations) for the differences
    opcodes = matcher.get_opcodes()

    # Process the opcodes to create the merged_tokens list
    merged_tokens = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            merged_tokens.append([text2_tokens_ids[j1:j2], '+'])
            merged_tokens.append([text1_tokens_ids[i1:i2], '-'])
        elif tag == 'delete':
            merged_tokens.append([text1_tokens_ids[i1:i2], '-'])
        elif tag == 'insert':
            merged_tokens.append([text2_tokens_ids[j1:j2], '+'])
        elif tag == 'equal':
            merged_tokens.append([text1_tokens_ids[i1:i2], None])
        else:
            raise ValueError(f"Unknown tag: {tag}")
        
        if len(merged_tokens) >= 3 and \
            merged_tokens[-1][1] == None and \
            merged_tokens[-2][1] == '-' and \
            merged_tokens[-3][1] in ['+', None]:
            

            # if merged_tokens[-2][0].endswith(merged_tokens[-3][0]):
            if merged_tokens[-3][0] == merged_tokens[-2][0][-len(merged_tokens[-3][0]):]:
                token_3 = merged_tokens.pop()
                token_2 = merged_tokens.pop()
                token_1 = merged_tokens.pop()
                
                if token_1[1] == None:
                    # merged_tokens.append((token_1[0] + token_2[0][:-len(token_1[0])], token_2[1]))
                    merged_tokens.append([token_1[0] + token_2[0][:-len(token_1[0])], token_2[1]])
                elif token_1[1] == '+':
                    # merged_tokens.append((token_2[0][:-len(token_1[0])], token_2[1]))
                    merged_tokens.append([token_2[0][:-len(token_1[0])], token_2[1]])
                # merged_tokens.append((token_1[0] + token_3[0], token_3[1]))
                merged_tokens.append([token_1[0] + token_3[0], token_3[1]])
        elif len(merged_tokens) >= 2:
            if set([merged_tokens[-1][1], merged_tokens[-2][1]]) == set(['+', '-']):
                token_2 = merged_tokens.pop()
                token_1 = merged_tokens.pop()

                common_prefix, common_suffix = compare_lists(token_1[0], token_2[0])

                if common_prefix:
                    # token_1 = (token_1[0][len(common_prefix):], token_1[1])
                    token_1 = [token_1[0][len(common_prefix):], token_1[1]]
                    # token_2 = (token_2[0][len(common_prefix):], token_2[1])
                    token_2 = [token_2[0][len(common_prefix):], token_2[1]]
                if common_suffix:
                    # token_1 = (token_1[0][:-len(common_suffix)], token_1[1])
                    token_1 = [token_1[0][:-len(common_suffix)], token_1[1]]
                    # token_2 = (token_2[0][:-len(common_suffix)], token_2[1])
                    token_2 = [token_2[0][:-len(common_suffix)], token_2[1]]
                
                if common_prefix:
                    # merged_tokens.append((common_prefix, None))
                    merged_tokens.append([common_prefix, None])
                if token_1[0]:
                    merged_tokens.append(token_1)
                if token_2[0]:
                    merged_tokens.append(token_2)
                if common_suffix:
                    # merged_tokens.append((common_suffix, None))
                    merged_tokens.append([common_suffix, None])

    # Merge adjacent tokens with the same category
    final_merged_tokens = [merged_tokens[0]]
    last_token_category = merged_tokens[0][1]
    for token, category in merged_tokens[1:]:
        if category == last_token_category:
            final_merged_tokens[-1] = [final_merged_tokens[-1][0] + token, category]
        else:
            final_merged_tokens.append([token, category])
        last_token_category = category

    text1_diff_tokens = [x for x in final_merged_tokens if x[1] in ['-', None]]
    text2_diff_tokens = [x for x in final_merged_tokens if x[1] in ['+', None]]
    text1_tokens_after_diff = []
    for x in text1_diff_tokens:
        text1_tokens_after_diff.extend(x[0])
    text2_tokens_after_diff = []
    for x in text2_diff_tokens:
        text2_tokens_after_diff.extend(x[0])
    assert text1_tokens_after_diff == text1_tokens_ids, f"After diff, the tokens in text1 are \n{text1_tokens_after_diff}\n but expected \n{text1_tokens_ids}\n"
    assert text2_tokens_after_diff == text2_tokens_ids, f"After diff, the tokens in text2 are \n{text2_tokens_after_diff}\n but expected \n{text2_tokens_ids}\n"
    return text1_diff_tokens, text2_diff_tokens

def get_diff_label_indices(text1_tokens_ids: List[int], text2_tokens_ids: List[int]):
    text1_diff_tokens, text2_diff_tokens = diff_token_ids(text1_tokens_ids, text2_tokens_ids)
    # only set labels for those with '-' in text1_diff_tokens
    text1_diff_labels = []
    for token, category in text1_diff_tokens:
        if category == '-':
            text1_diff_labels.extend([1] * len(token))
        else:
            text1_diff_labels.extend([0] * len(token))
    
    # only set labels for those with '+' in text2_diff_tokens
    text2_diff_labels = []
    for token, category in text2_diff_tokens:
        if category == '+':
            text2_diff_labels.extend([1] * len(token))
        else:
            text2_diff_labels.extend([0] * len(token))
    # print(f"text1_diff_tokens: {text1_diff_tokens}")
    # print(f"text1_diff_labels: {text1_diff_labels}")
    # print(f"text2_diff_tokens: {text2_diff_tokens}")
    # print(f"text2_diff_labels: {text2_diff_labels}")
    
    # convert to index so the tensor or numpy array can use these indices to slice
    text1_diff_label_indices= [i for i, x in enumerate(text1_diff_labels) if x == 1]
    text2_diff_label_indices = [i for i, x in enumerate(text2_diff_labels) if x == 1]
    return text1_diff_label_indices, text2_diff_label_indices