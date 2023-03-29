import openai
import os
import time
import matplotlib.pyplot as plt
import random
import pickle as pkl

MOST_RECENT_PROJECT_EULER_ID = 825
openai.api_key = os.environ.get("OPENAI_API_KEY")

answers = pkl.load(open("answers.pkl", "rb"))


def get_gpt_answer(problem_number, gpt_version):
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
                        "content": f"Give me the numerical solution to Project Euler problem number 1. Give me the numerical answer and nothing else.",
                    },
                    {
                        "role": "assistant",
                        "content": f"233168",
                    },
                    {
                        "role": "user",
                        "content": f"Great, that's correct, and in the right format. Now give me the numerical solution to Project Euler problem number 2. Give me the numerical answer and nothing else.",
                    },
                    {
                        "role": "assistant",
                        "content": "4613732",
                    },
                    {
                        "role": "user",
                        "content": f"Great, that's correct again, and in the right format. Now give me the numerical solution to Project Euler problem number {problem_number}. Give me the numerical answer and nothing else.",
                    },
                ],
                max_tokens=30,
            )
            success = True
        except openai.error.RateLimitError as e:  # type: ignore
            backoff_seconds *= backoff_seconds + random.uniform(-1, 1)
            print(f"Got error {e}, waiting {backoff_seconds}")
            time.sleep(backoff_seconds)
    return response.choices[0].message["content"].strip(), response["choices"][0]["finish_reason"]  # type: ignore


def check_range(lower, upper, version):
    if version == "gpt-4":
        num_checks = 2
    else:
        num_checks = 5
    results = {}
    for problem_number, correct_answer in answers.items():
        if problem_number < lower or problem_number > upper:
            continue
        print(problem_number)
        success_count = 0
        gpt_answers = []
        why_finisheds = []
        for _ in range(num_checks):
            gpt_answer, why_finished = get_gpt_answer(problem_number, version)
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


lower = 300
upper = 350
results = check_range(lower, upper, "gpt-4")
# Since we use id 1, 2, in the prompt, we remove them for the plot
results.pop(1, None)
results.pop(2, None)
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
results.pop(1, None)
results.pop(2, None)
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
