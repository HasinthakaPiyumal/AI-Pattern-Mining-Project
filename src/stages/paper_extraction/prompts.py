#################################################################################################
# 01 Prompt Templates for Pattern Extraction
#################################################################################################

optimized_prompt = """
You are an AI design pattern mining expert.

Extract all **true AI design patterns** mentioned in the following research text. Ignore general software engineering, DevOps, or data engineering patterns.

For each pattern, include:
- Pattern Name :str
- Problem :str
- Context :str
- Solution :str
- Result :str
- Related Patterns :str
- Category :str
- Uses: str
- Thinking: Explain briefly how you identified this as an AI design pattern from the text.

Categories field must be one of the following: 
1. Classical AI
2. Generative AI
3. Agentic AI
4. Prompt Design
5. MLOps (only if specific to ML workflows, not general deployment)
6. AI–Human Interaction
7. LLM-specific
8. Tools Integration
9. Knowledge & Reasoning
10. Planning
11. Personalization

Return only a JSON array. Do not include markdown, extra text, or commentary.

Text:
{text}
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

summary_prompt = """
You are an expert in AI design patterns. 
Your task is to combine the following AI design patterns into a single, unified pattern. 
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
{patterns_text}
"""



#################################################################################################
# 04 Merge Prompt
#################################################################################################

merge_prompt = """
Combine all the following JSON arrays of AI patterns into one deduplicated, coherent JSON array.
If multiple patterns describe similar problems or solutions, merge them carefully.
Return only the final JSON array.

All extracted pattern lists:
{partial_jsons}
"""
#################################################################################################