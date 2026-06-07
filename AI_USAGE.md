# Generative AI Usage Documentation

Generative AI tools were used for this project and are documented for attribution. AI was used as a support tool for implementation, debugging, formatting, and brainstorming; it was not used as a substitute for project ownership, technical judgment, or interpretation of results.

## Tools Used

- ChatGPT / Codex

## Uses

- Quickly triaging and summarizing relevant papers during project brainstorming
- Brainstorming experiment plans and analysis directions
- Debugging Python, PyTorch, and Modal training/evaluation scripts
- Generating and refactoring code, with author review before use
- Creating and revising the model architecture diagram
- Generating plotting scripts for threshold tradeoff and temporal ablation figures
- Formatting LaTeX tables, figures, and report layout
- Proofreading and editing report language for clarity


## Representative Historical Prompts

The following are representative examples of prompts used during the project. They are abbreviated only for readability and are intended to document the kinds of AI assistance requested.

- "Restructure this notebook to a repository, and call the current model the baseline model."
- "Run the model remotely in Modal; help me install and authenticate Modal."
- "Extract EDA into `scripts/eda.py` and save the visualization of the data distribution."
- "Reduce the prediction horizon to 1.0s and use class-weighted cross entropy for all training paths."
- "For negative examples, only generate sliding windows before the action onset."
- "Add LE2I using its annotation files, combine frame-based and timestamp-based labels consistently, and verify FPS before constructing windows."
- "Add CAUCAFall into the staged training pool with GMDCSA24 and LE2I, preprocess it the same way, and extract pose and V-JEPA latent features."
- "Create V-JEPA baseline and V-JEPA predictive models; use a weighted classification plus predictive latent loss with cosine loss."
- "Create a simple pose plus V-JEPA fusion model by projecting each modality to 256 dimensions and concatenating them."
- "Train Pose Transformer, V-JEPA baseline, V-JEPA predictive, and fusion models on Modal GPUs, report validation/test metrics, and tune thresholds using F2 and balanced accuracy."
- "Compute confusion matrix counts, positive precision, positive recall, F1, F2, balanced accuracy, and threshold-tuned summaries."
- "Create Precision-Recall and Recall-vs-False-Positive-Rate curves for the staged+unstaged models."
- "Run temporal ablation for the staged+unstaged models and create a table explaining which observation step is most important."
- "Help interpret why V-JEPA may outperform pose, including dataset bias, LE2I scene-context shortcuts, and staged versus unstaged distribution shift."
- "Create a higher-resolution model architecture diagram for the paper and clarify that the predictive latent loss uses cosine distance."
- "Turn these result tables into LaTeX format suitable for an academic paper and add them to the Experiments section."
- "Go through the Experiments, Results, and Discussion section and help create a coherent narrative from the experiment and analysis results."

## Boundaries

The project authors selected the research questions, data preprocessing strategy, model designs, experiment settings, reported results, written analysis, and final interpretation. During the project, the authors actively inspected model behavior, questioned unexpected results, revised dataset construction choices, compared threshold policies, identified dataset bias concerns, and decided which experiments were scientifically meaningful to include. Generative AI suggestions were treated as drafts or implementation assistance and were reviewed, modified, accepted, or rejected by the authors. Generative AI was not used to fabricate data, fabricate metrics, write substantive report content, or replace the authors' conclusions. Reported results come from the implemented training and evaluation pipeline.

## Documentation

Prompts and transcripts are retained in the ChatGPT/Codex conversation history associated with this project. The prompts show iterative technical decision-making by the authors, including asking why results looked suspicious, whether data splits were valid, how label definitions affected the task, whether model comparisons were fair, and which analyses best supported the final claims. AI-assisted artifacts include portions of code, LaTeX formatting, and editing suggestions, all reviewed by the project authors before inclusion.
