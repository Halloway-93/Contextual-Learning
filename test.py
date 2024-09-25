import numpy as np

N = 240
l = np.array([0] * (N // 2) + [1] * (N // 2))
# Generate a list of 0 and 1 with Bernoulli distribution
# l = stats.bernoulli.rvs(0.5, size=240)
np.random.shuffle(l)
print(l)
# Initialize counters
count_1_followed_by_1 = 0
count_1 = 0

# Iterate through the list to count occurrences
for i in range(len(l) - 1):
    if l[i] == 1:
        count_1 += 1
        if l[i + 1] == 1:
            count_1_followed_by_1 += 1

# Compute the conditional probability p(i+1|i)
if count_1 > 0:
    p_i_plus_1_given_i = count_1_followed_by_1 / count_1
else:
    p_i_plus_1_given_i = 0  # To avoid division by zero

print(f"The conditional probability p(i+1|i) is: {p_i_plus_1_given_i}")
