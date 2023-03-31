import openai
import os
import time
import matplotlib.pyplot as plt
import random
import pickle as pkl

MOST_RECENT_PROJECT_EULER_ID = 825
openai.api_key = os.environ.get("OPENAI_API_KEY")

answers = {1: 55944}


def get_gpt_answer(gpt_version):
    success = False
    backoff_seconds = 3
    while not success:
        try:
            response = openai.ChatCompletion.create(
                model=gpt_version,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful, highly technically accomplished assistant.",
                    },
                    {
                        "role": "user",
                        "content": """Here is a numerical problem for you: If we list all the natural numbers below 20 that are multiples of 6 or 15, we get 6, 12, 15 and 18. The sum of these multiples is 51.Find the sum of all the multiples of 6 or 15 below 1000.""",
                    },
                    {
                        "role": "assistant",
                        "content": f"99501",
                    },
                    {
                        "role": "user",
                        "content": """Great, that's correct, and in the right format. Now give me the numerical solution to this problem:If we list all the natural numbers below 20 that are multiples of 9 or 18, we get 9, 18. The sum of these multiples is 27.Find the sum of all the multiples of 9 or 18 below 1000. Give me the numerical answer and nothing else.""",
                    },
                ],
                max_tokens=30,
            )
            success = True
            print(response)
        except openai.error.RateLimitError as e:  # type: ignore
            backoff_seconds *= backoff_seconds + random.uniform(-1, 1)
            print(f"Got error {e}, waiting {backoff_seconds}")
            time.sleep(backoff_seconds)
    return response.choices[0].message["content"].strip(), response["choices"][0]["finish_reason"]  # type: ignore


def check_range(lower, upper, version):
    if version == "gpt-4":
        num_checks = 10
    else:
        num_checks = 30
    results = {}
    for problem_number, correct_answer in answers.items():
        if problem_number < lower or problem_number > upper:
            continue
        print(problem_number)
        success_count = 0
        gpt_answers = []
        why_finisheds = []
        for _ in range(num_checks):
            gpt_answer, why_finished = get_gpt_answer(version)
            if gpt_answer == str(correct_answer):
                success_count += 1
            gpt_answers += [gpt_answer]
            why_finisheds += [why_finished]
        results[problem_number] = {
            "correct_answer": correct_answer,
            "success_rate": success_count / num_checks,
            "gpt_answers": gpt_answers,
            "finish_reason": why_finisheds,
        }
        time.sleep(2)
    return results


lower = 0
upper = 5
results = check_range(lower, upper, "gpt-4")
# Since we use id 1, 2, in the prompt, we remove them for the plot
problem_ids = list(results.keys())
success_rates = [result["success_rate"] for result in results.values()]
print(results)
plt.bar(
    problem_ids,
    success_rates,
    linewidth=2,
    edgecolor="k",
    color="tab:red",
    label="GPT-4",
    alpha=0.75,
)
plt.xlabel("Problem ID")
plt.ylabel("Success Rate")
plt.title("Success Rate for Project Euler Problems")

results = check_range(lower, upper, "gpt-3.5-turbo-0301")
problem_ids = list(results.keys())
success_rates = [result["success_rate"] for result in results.values()]
print(results)
plt.bar(
    problem_ids,
    success_rates,
    linewidth=2,
    edgecolor="k",
    color="tab:blue",
    label="GPT-3.5-turbo-0301",
    alpha=0.75,
)
plt.legend()

plt.show()
