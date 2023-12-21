from tokenizer import decode, encode

s = "清心拂尘服"
input_ids = encode(s)
print(input_ids)

decoded = decode(input_ids)
print(decoded)
