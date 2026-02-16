#################################################################################################
# 01 Prompt Templates for Pattern Extraction
#################################################################################################

PATTERN_TYPE = "Microservices"

optimized_prompt = f"""
You are an {PATTERN_TYPE} design pattern mining expert.

Extract all **true {PATTERN_TYPE} design patterns** mentioned in the following research text.

For each pattern, include:
- Pattern Name :str
- Problem :str
- Context :str
- Solution :str
- Result :str
- Related Patterns :str
- Uses: str
- Thinking: Explain briefly how you identified this as an {PATTERN_TYPE} design pattern from the text.

Return only a JSON array. Do not include markdown, extra text, or commentary.

Text:
{'{text}'}
"""



#################################################################################################
# 02 Retry Prompt
#################################################################################################

retry_prompt = """\
following is a list of patterns and thinking on how it was extracted in JSON format and paper text from which those patterns were extracted. 
Look for any patterns that are not identified from the paper. If there are any missing design patterns from the paper text, extract them as well and add to the below json array.
if there is any issue with bellow json format, correct it and return only the json array.
""" + optimized_prompt + """

Extracted patterns so far:
{extracted_patterns}
"""



#################################################################################################
# 03 Summary Prompt
#################################################################################################

summary_prompt = f"""
You are an expert in {PATTERN_TYPE} design patterns. 
Your task is to combine the following {PATTERN_TYPE} design patterns into a single, unified pattern. 
Use information from all patterns to produce one coherent pattern that includes:

- Pattern Name :str
- Problem :str
- Context :str
- Solution :str
- Result :str
- Related Patterns :str
- Category :str
- Uses: str

Return strictly as JSON. Do not add extra text, explanations, or formatting.

Patterns to combine:
{'{patterns_text}'}
"""



#################################################################################################
# 04 Merge Prompt
#################################################################################################

merge_prompt = f"""
Combine all the following JSON arrays of {PATTERN_TYPE} patterns into one deduplicated, coherent JSON array.
If multiple patterns describe similar problems or solutions, merge them carefully.
Return only the final JSON array.

All extracted pattern lists:
{'{partial_jsons}'}
"""
#################################################################################################