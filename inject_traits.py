import re

with open("src/algorithms.rs", "r") as f:
    text = f.read()

# We need to change:
# pub trait HashableChar: Copy + Eq + Hash + Sized + Into<u64> {}
# impl<T: Copy + Eq + Hash + Sized + Into<u64>> HashableChar for T {}
# To include Into<u32>.
# Let's just find `Into<u64>` and replace with `Into<u32> + Into<u64>` inside the top of algorithms.rs.

text = text.replace("pub trait HashableChar: Copy + Eq + Hash + Sized + Into<u64> {}", "pub trait HashableChar: Copy + Eq + Hash + Sized + Into<u32> + Into<u64> {}")
text = text.replace("impl<T: Copy + Eq + Hash + Sized + Into<u64>> HashableChar for T {}", "impl<T: Copy + Eq + Hash + Sized + Into<u32> + Into<u64>> HashableChar for T {}")

with open("src/algorithms.rs", "w") as f:
    f.write(text)

print("Injected Trait bounds.")
