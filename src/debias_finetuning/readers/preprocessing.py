from typing import List
from drqa.retriever.utils import normalize


def get_non_empty_lines(lines: List[str]) -> List[str]:
    return [line for line in lines if len(line.strip())]


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_gold(l):
    ret = set()
    for ev_group, page, line in l:
        ret.add((page,line))
    return ret


def preprocess(page_title):
    return page_title.replace("_", " ").replace("-LRB-", " ( ").replace("-RRB-", " ) ").replace("-COLON-", " : ")


def unpreprocess(content):
    return content.replace("-LRB-", " ( ").replace("-RRB-", " ) ").replace("-COLON-", " : ")


def format_line(page, line):
    return normalize(preprocess(page) + " : " + unpreprocess(line))