import re

with open("src/distance/metrics.rs", "r") as f:
    text = f.read()

# Lines causing errors are around 753. Let's inspect those manually via regex.
text = re.sub(r'dispatch_metric!\(alg::lcs_seq_similarity,\s+&a,\s+&b\)', 
              r'dispatch_metric!(alg::lcs_seq_similarity, &a, &b, None)', text)

text = re.sub(r'dispatch_metric!\(alg::lcs_seq_distance,\s+&a,\s+&b\)', 
              r'dispatch_metric!(alg::lcs_seq_distance, &a, &b, None)', text)

# There may also be `av` and `bv` variants:
text = re.sub(r'dispatch_metric!\(alg::lcs_seq_similarity,\s+&av,\s+&bv\)', 
              r'dispatch_metric!(alg::lcs_seq_similarity, &av, &bv, None)', text)

text = re.sub(r'dispatch_metric!\(alg::lcs_seq_distance,\s+&av,\s+&bv\)', 
              r'dispatch_metric!(alg::lcs_seq_distance, &av, &bv, None)', text)

with open("src/distance/metrics.rs", "w") as f:
    f.write(text)

print("regex fixed")
