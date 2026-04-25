from datasets import load_dataset

ds = load_dataset("Marqo/polyvore")

data = ds["data"]   # select split first
print(len(data))

set(data["category"])

CLOTHING_MAP = {

    # -------- TOPS --------
    "Blouses": "tops",
    "T-Shirts": "tops",
    "Tank Tops": "tops",
    "Sweaters": "tops",
    "Cardigans": "tops",
    "Tunics": "tops",
    "Hoodies": "tops",
    "Tops": "tops",

    # -------- BOTTOMS --------
    "Jeans": "bottoms",
    "Skinny Jeans": "bottoms",
    "Bootcut Jeans": "bottoms",
    "Flared Jeans": "bottoms",
    "Wide Leg Jeans": "bottoms",
    "Pants": "bottoms",
    "Shorts": "bottoms",
    "Skirts": "bottoms",
    "Mini Skirts": "bottoms",
    "Long Skirts": "bottoms",

    # -------- DRESSES --------
    "Day Dresses": "dresses",
    "Cocktail Dresses": "dresses",
    "Dresses": "dresses",
    "Wedding Dresses": "dresses",
    "Gowns": "dresses",

    # -------- OUTERWEAR --------
    "Jackets": "outerwear",
    "Coats": "outerwear",
    "Blazers": "outerwear",
    "Outerwear": "outerwear",
}

def keep_clothes(example):
    cat = example["category"]

    if cat in CLOTHING_MAP:
        example["norm_category"] = CLOTHING_MAP[cat]
        return example

    example["norm_category"] = None
    return example


data = data.map(keep_clothes)

data = data.filter(lambda x: x["norm_category"] is not None)
set(data["norm_category"])

data = data.filter(
    lambda x: x["norm_category"] in ["tops", "bottoms"]
)

set(data["norm_category"])

from collections import Counter

Counter(data["norm_category"])

print(data[0])


from collections import defaultdict

outfits = defaultdict(list)

for item in data:
    set_id = item["item_ID"].split("_")[0]   # extract outfit id
    outfits[set_id].append(item)

print("total outfits: ", len(outfits))

positive_pairs = []

for set_id, items in outfits.items():

    tops = [i for i in items if i["norm_category"] == "tops"]
    bottoms = [i for i in items if i["norm_category"] == "bottoms"]

    # create all valid outfit combinations
    for top in tops:
        for bottom in bottoms:

            positive_pairs.append({
                "top_id": top["item_ID"],
                "bottom_id": bottom["item_ID"],
                "label": 1
            })

print("Positive pairs:", len(positive_pairs))


all_tops = []
all_bottoms = []

for item in data:
    if item["norm_category"] == "tops":
        all_tops.append(item)

    elif item["norm_category"] == "bottoms":
        all_bottoms.append(item)

import random

negative_pairs = []

target = len(positive_pairs)

while len(negative_pairs) < target:

    top = random.choice(all_tops)
    bottom = random.choice(all_bottoms)

    top_set = top["item_ID"].split("_")[0]
    bottom_set = bottom["item_ID"].split("_")[0]

    # mismatch outfits
    if top_set != bottom_set:

        negative_pairs.append({
            "top_id": top["item_ID"],
            "bottom_id": bottom["item_ID"],
            "label": 0
        })

print("Negative pairs:", len(negative_pairs))

pairs = positive_pairs + negative_pairs
random.shuffle(pairs)

print(len(pairs))
print(pairs[0])

import json, os

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/pairs.json", "w") as f:
    json.dump(pairs, f)


set_ids = list({
    p["top_id"].split("_")[0]
    for p in pairs
})
random.shuffle(set_ids)

n = len(set_ids)

train_ids = set(set_ids[:int(0.8*n)])
val_ids   = set(set_ids[int(0.8*n):])

train_pairs = []
val_pairs = []

for p in pairs:

    set_id = p["top_id"].split("_")[0]

    if set_id in train_ids:
        train_pairs.append(p)
    else:
        val_pairs.append(p)

json.dump(train_pairs, open("data/processed/train_pairs.json","w"))
json.dump(val_pairs, open("data/processed/val_pairs.json","w"))