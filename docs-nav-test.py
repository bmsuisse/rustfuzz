import tomli_w
import tomllib

with open("zensical.toml", "rb") as f:
    config = tomllib.load(f)

# Reformat nav to a list of dicts
config["project"]["nav"] = [
    {"Home": "index.md"},
    {"Internals": [
        {"Architecture & Design": "architecture.md"},
        {"Developer & Contributor Guide": "developer_guide.md"}
    ]},
    {"Cookbook": [
        {"Introduction": "cookbook/01_introduction.md"},
        {"Advanced Matching": "cookbook/02_advanced_matching.md"},
        {"Benchmarks": "cookbook/03_benchmarks.md"},
        {"Vector DB Hybrid Search": "cookbook/04_hybrid_search.md"},
        {"LangChain Integration": "cookbook/05_langchain.md"},
        {"Real-World Examples": "cookbook/06_real_world.md"},
        {"Fuzzy Full Join": "cookbook/07_fuzzy_join.md"},
        {"3-Way Hybrid Search": "cookbook/08_3way_hybrid_search.md"},
        {"Filter & Sort": "cookbook/10_filter_sort.md"},
        {"EmbedAnything Integration": "cookbook/11_embed_anything.md"}
    ]}
]

with open("zensical.toml", "wb") as f:
    tomli_w.dump(config, f)
