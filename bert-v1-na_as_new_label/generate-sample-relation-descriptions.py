from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

extract_trigger_prompt = """
Given a piece of text and two entities subject, object (not ordered), extract the relation trigger in the form of [Subject, Relation trigger, Object] from it.
Here are some examples:
Example 1:
Text: "he passed away on saturday ."
Subject, Object (not ordered): "he", "saturday"
Complete triplets: ["he", "passed away on", "saturday"]

Example 2:
Text: "as a substantial shareholder in cnac 's subsidiary air china , cathay pacific said late monday it would give serious consideration to joining cnac and form a strategic partnership with china eastern ."
Subject, Object (not ordered): "cnac", "cathay pacific"
Complete triplets: ["cathay pacific", "a substantial shareholder", "cnac"]

Now it's your turn! Please extract the relation trigger from the following text:
Text: "{text}"
Subject, Object (not ordered): "{head}", "{tail}"
Complete triplets:
""".strip()

relation_definition_prompt = """
Define the relationship in a relational triplet extracted from a given text.

Example 1:
Text: "he passed away on saturday ."
Triplet: ["he", "passed away on", "saturday"]
Definition:
- passed away on: The event of someone dying on a specific day.

Example 2:
Text: "as a substantial shareholder in cnac 's subsidiary air china , cathay pacific said late monday it would give serious consideration to joining cnac and form a strategic partnership with china eastern ."
Triplet: ["cathay pacific", "a substantial shareholder", "cnac"]
Definition:
- a substantial shareholder: The relationship between a company and another company in which the former owns a significant portion of the latter's shares.

Now it's your turn! Please define the relationship in the following relational triplet:
Text: "{text}"
Triplet: {triplet}
Definition:
- {trigger}:
""".strip()

file = "/Users/anhnguyenhoang/Desktop/vscode/python3.10/fcre/bert/data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt"

raw_data = []

with open(file) as f:
    for line in f:
        items = line.strip().split('\t')
        raw_data.append(items)

from tqdm import tqdm
import re
import json

# process and save to txt file line by line
with open("/Users/anhnguyenhoang/Desktop/vscode/python3.10/fcre/bert/data/CFRLTacred/CFRLdata_6_100_5_5/test_1.txt", "w") as f:
    for data in tqdm(raw_data):
        text, head, tail = data[2], data[3], data[5]
        prompt = extract_trigger_prompt.format(text=text, head=head, tail=tail)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            top_p=0.5,
            max_tokens=256,
            stream=False
        )
        content = response.choices[0].message.content.strip()

        try:
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, content)
            triplet = json.loads("[" + matches[0] + "]")
        except Exception as e:
            print(e)
            print(content)
            print(data)
            triplet = [head, "null", tail]

        if triplet[1] != "null":
            prompt = relation_definition_prompt.format(text=text, triplet=json.dumps(triplet), trigger=triplet[1])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                top_p=0.5,
                max_tokens=128,
                stream=False
            )
            relation_description = response.choices[0].message.content.strip()
            if f"- {triplet[1]}: " in relation_description:
                relation_description = relation_description.replace(f"- {triplet[1]}: ", "")
        else:
            relation_description = "null"
        
        data = data + [json.dumps(triplet), relation_description]

        line_to_write = "\t".join(data) + "\n"

        f.write(line_to_write)
