import json
from math import inf
from tqdm import tqdm
from operator import itemgetter


def main():
    k = 400
    json_paths = [
        "scores.json",
    ]

    scores_list = []
    for path in json_paths:
        with open(path, 'r') as file:
            scores = json.load(file)
        scores = {q: rs for q, rs in scores}
        scores_list.append(scores)
    
    concat_scores = {}
    for q in tqdm(list(scores_list[0].keys())):
        rs = {}
        for scores in scores_list:
            for r, s in scores[q]:
                rs[r] = max(s, rs.get(r, -inf))
        rs = [[r, s] for r, s in rs.items()]
        rs = sorted(rs, key=itemgetter(1), reverse=True)
        concat_scores[q] = rs
    
    concat_scores = [[q, rs[: k]] for q, rs in concat_scores.items()]
    
    with open("scores.json", 'w') as file:
        json.dump(concat_scores, file)
    

def main_divide_scores():
    path = "scores.json"
    k = 10
    with open(path) as file:
        scores = json.load(file)

    for i in range(k):
        _scores = scores[i::k]
        with open(f"scores_part-{i}.json", 'w') as file:
            json.dump(_scores, file)


if __name__ == "__main__":
    # main()
    main_divide_scores()
