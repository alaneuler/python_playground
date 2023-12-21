import json
import re

poems = []
poem = ""
with open("data/tang_poems.txt") as f:
    for line in f:
        if line.strip() == "":
            continue

        if re.match(r"^\d{3}", line):
            if poem:
                poems.append(poem)
            poem = ""
        else:
            if line.endswith("\n\n"):
                line = line[:-1]
            poem += line
print(f"Poem number is {len(poems)}")
with open("data/tang_poems.json", "w") as f:
    json.dump(poems, f, ensure_ascii=False, indent=2)
