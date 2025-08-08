import pandas as pd

origin_code = pd.read_csv("/workspace/dacon_choisunggyu/baseline_submit_tuned.csv")
recovered_code = pd.read_csv("/workspace/dacon_choisunggyu/baseline_submit_tuned_verified.csv")
counter = 0

for i in range(len(origin_code)):
    if origin_code["answer"][i] == recovered_code["answer"][i]:
        pass
    else:
        counter = counter + 1

print(f"문항의 개수 : {len(origin_code)}")
print(f"틀린 개수 : {counter}")

