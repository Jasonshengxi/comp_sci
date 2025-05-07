import describe

data = describe.load_data()
for k, v in data.items():
    v.embeddings = []
describe.store_data(data)
