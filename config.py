from src.interestingness.save_sampled_KB_for_interestingness import save_sampled_KB_for_interestingness
from src.interestingness.interestingness_GB import save_count_sample

print("Saving sample KG for interestingness ...")
save_sampled_KB_for_interestingness(20000)

print("Saving interestingness ...")
save_count_sample()
