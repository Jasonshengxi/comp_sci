import describe
import sys

invalidate = sys.argv

data = describe.load_data()
for k, v in data.items():
    if "embeddings" in invalidate:
        v.embeddings = []
    if "answers" in invalidate:
        v.answers = []
describe.store_data(data)
