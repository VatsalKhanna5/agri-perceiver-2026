import pandas as pd

df = pd.read_csv("gpu_usage.log", header=None)
df.columns = ["timestamp","gpu","util","mem_util","mem_used","mem_total"]

print("Avg GPU Util:", df["util"].mean())
print("Avg Mem Util:", df["mem_util"].mean())
