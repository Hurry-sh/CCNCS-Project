import os
import pickle
import pandas as pd
import gzip

folder = 'CCNCS_Internship\\CNN\\data\\'

target = os.listdir(folder)
cnt = 0
d = []
cls = []

for item in target:
    print(item)
    payload = os.path.join(folder, item)
    print(f"\nProcessing: {payload}")

    if os.path.isfile(payload):
        d.append([])
        fname = item
        with open(payload, 'r', encoding='UTF-8') as reader:
            data = reader.read()
            if not data:
                print(f"Empty file: {item}")
            lines = data.strip().split('\n')
            print(f"Lines: {lines}")

            for row in lines[1:]:
                api = row.split(',')[0]
                if api != 'Class':
                    d[cnt].append(api)

        if 'mal' in item:
            cls.append(1)
        elif 'be' in item:
            cls.append(0)
        d[cnt] = list(set(d[cnt]))
        d[cnt].insert(0, fname)

        cnt += 1

print(f"Data list: {d}")
all = pd.DataFrame(d)
print(f"DataFrame before filling NA: {all}")
all = all.fillna(0)
all['class'] = cls

# print(f"Final DataFrame: {all}")

print(all)

with gzip.open('CCNCS_Internship/CNN/output_data/output.pickle', 'wb') as f:
    pickle.dump(all, f)

