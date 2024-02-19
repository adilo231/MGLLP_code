import subprocess
from threading import Thread

rej_baseline_out = open("log/rej_baseline/rej_baseline_out", "a")
rej_baseline_err = open("log/rej_baseline/rej_baseline_err", "a")

#gcnfn_out = open("log/baseline/gcnfn_out", "a")
#gcnfn_err = open("log/baseline/gcnfn_err", "a")

#gnn_out = open("log/baseline/gnn_out", "a")
#gnn_err = open("log/baseline/gnn_err", "a")

#gnncl_out = open("log/baseline/gnncl_out", "a")
#gnncl_err = open("log/baseline/gnncl_err", "a")

def worker(baseline_method, model, dataset, out_file, err_file): # 1/ 16/512
        if model in ["gcn", "gat", "sage"]:
            subprocess.run(["python", f"rej_baseline/{baseline_method}.py", f"--dataset={dataset}", f"--model={model}"], stdout=out_file, stderr=err_file)
        else:
            subprocess.run(["python", f"rej_baseline/{baseline_method}.py", f"--dataset={dataset}"], stdout=out_file, stderr=err_file)


datasets = ["politifact", "gossipcop", "pheme", "weibo"]
baseline_methods = ["bigcn", "gcnfn", "gnncl"]

all_exps = [(dataset, method) for dataset in datasets for method in baseline_methods]
# politifact.
for dataset, method in all_exps:
    baseline = Thread(target = worker, args = (method, None, dataset, rej_baseline_out, rej_baseline_err))
    baseline.start()
    baseline.join()

# for gnn wich contains 3-sub methodes: gcn, gat, sage 
gnns = ["gcn", "gat", "sage"]


all_exps = [(dataset, model) for dataset in datasets for model in gnns]

for dataset, model in all_exps:
    baseline = Thread(target = worker, args = ("gnn", model, dataset, rej_baseline_out, rej_baseline_err))
    baseline.start()
    baseline.join()

