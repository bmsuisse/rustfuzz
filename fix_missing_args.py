import re

with open("src/distance/metrics.rs", "r") as f:
    text = f.read()

# Fix lcs_seq_distance which is called with only two arguments natively
text = text.replace("dispatch_metric!(alg::lcs_seq_distance, &av, &bv)", 
                    "dispatch_metric!(alg::lcs_seq_distance, &av, &bv, None)")

# Fix lcs_seq_similarity which is called with only two arguments internally
text = text.replace("dispatch_metric!(alg::lcs_seq_similarity, &av, &bv)",
                    "dispatch_metric!(alg::lcs_seq_similarity, &av, &bv, None)")

# Fix indel_distance if we missed any
text = text.replace("dispatch_metric!(alg::indel_distance, &av, &bv)",
                    "dispatch_metric!(alg::indel_distance, &av, &bv, None)")

with open("src/distance/metrics.rs", "w") as f:
    f.write(text)

print("Fixed missing argument dispatches")
