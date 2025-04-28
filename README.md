# Ranked Voting based Self-Consistency of Large Language Models

## Quick Links

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Evaluation](#evaluation)
  - [Step 1: Generate Model Outputs](#step-1-generate-model-outputs)
  - [Arguments Explanation](#arguments-explanation)
  - [Step 2: Perform Ranked Voting and Evaluation](#step-2-perform-ranked-voting-and-evaluation)
- [Expected Output](#expected-output)

## Project Structure
```
.
├── data/                        # Directory to store input and output data files, we provide a example data generate by Gemma-2-9b-it on CommonsenseQA (CSQA)
├── figs/                        # Folder for storing figures or plots
├── lm-evaluation-harness/       # Evaluation framework (forked/modified from EleutherAI)
├── RankedVotingSC/              # Source code for Ranked Voting based Self-Consistency
├── LICENSE                      # License file
├── README.md                    # Project description and usage instructions
└── evaluate.py                  # Script for performing Ranked Voting and evaluation
```

## Getting Started

Clone our repository and install the required environment:

```bash
# Clone the repository
git clone https://github.com/szu-tera/RankedVotingSC.git
cd lm-evaluation-harness
# Install lm-evaluation-harness
pip install -e .
```

## Evaluation

First, define a task YAML file under `code/lm-evaluation-harness/lm_eval/tasks/`.  
For example, the task configuration for **Gemma-2-9b-it** on **CommonsenseQA (CSQA)** can be found [here](https://github.com/szu-tera/RankedVotingSC/blob/main/lm-evaluation-harness/lm_eval/tasks/CSQA/CSQA.yaml) (If you need to implement a new task, please refer to [this guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)).

Next, we provide a step-by-step demonstration using **Gemma-2-9b-it** on the **CSQA** dataset (example outputs [here](https://github.com/szu-tera/RankedVotingSC/blob/main/data/samples.json)) as an example:

---

### Step 1: Generate Model Outputs

Run the following command to generate outputs from the model:

```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=[path/to/your/model],dtype=bfloat16 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path RankedVotingSC/data \
    --tasks CSQA-RankedVoting \
    --batch_size auto
```

> **Note:** Replace `[path/to/your/model]` with your local model path or the Huggingface model ID.  

---

### Arguments Explanation

- `--model`: Specify the format of the model to use.
  - `hf`: Use Huggingface format.

- `--model_args`: Specify the model path and the precision type.
  - `pretrained`: Path to the model on Huggingface or local directory.
  - `dtype`: Precision type for loading (e.g., bfloat16).

- `--apply_chat_template`: Apply the chat template formatting to input prompts.

- `--fewshot_as_multiturn`: Format few-shot examples as multi-turn dialogues.

- `--log_samples`: Save generated samples for further filtering and voting.

- `--output_path`: Specify the directory where output files (e.g., `RankedVotingSC/data`) will be saved.

- `--tasks`: Specify the evaluation task (e.g., `CSQA-RankedVoting`).

- `--batch_size`: Specify the batch size for generation.
  - `auto`: Automatically adjust the batch size based on available GPU memory.

---

### Step 2: Perform Ranked Voting and Evaluation

After generating the outputs, use the following command to conduct Ranked Voting and evaluate performance:

```bash
python evaluate.py path/to/your/samples.json
```

### Expected Output

```
Get Key! Done!
Extract Answer! Done!
Vote! Done!
-------- test -------
+--------+----------+
| Method | Accuracy |
+--------+----------+
| IRV    | 0.891638 |
+--------+----------+
| BCV    | 0.890785 |
+--------+----------+
| MRR    | 0.894198 |
+--------+----------+
```
