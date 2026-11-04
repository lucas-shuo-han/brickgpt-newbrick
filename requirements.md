Meeting signup, HW9-P4 prerequisite, and proposal draft
Book two required team appointments with Matteo: appointment calendar （在新标签页中打开） or Prof. Ranade. A team books one slot for each appointment and come together. Both appointments are required and graded.

First appointment — Week of November 2: date and time to be confirmed by Matteo.

Second appointment — Week of November 16: date and time to be confirmed by Matteo.

Complete Homework 9 Problem 4 (Hardware Check) BEFORE your meeting. For the separate P4 submission, upload only hardware-results.csv to Homework 9 — Hardware Check CSV （在新标签页中打开）. HW9 is due November 4 at 10:59 p.m. PT; if your meeting is earlier, finish P4 before the meeting. See the HW9 announcement (#294) （在新标签页中打开） and P4 questions (#292) （在新标签页中打开）.

Bring a printed draft proposal. Form a team of four students. Come ready to discuss your research question, planned experiments, and available compute.

Project Direction: Theoretical/empirical study of in-context learning
In-context learning (ICL) is a model's ability to learn a new task from examples or experience placed in its context, with no weight update. Since GPT-3 it has become one of the main ways deployed models adapt to a new task. A coding agent today spends a lot of its tokens on context it did not write (tool outputs, memory files, summaries, skills, earlier attempts), assembled by the harness around the model, such as Claude Code or Codex. At the same time, the efficient sequence layers that are replacing attention (linear attention, DeltaNet, Mamba, test-time-training layers) are now designed as in-context learners: each one solves a small regression problem online, the same view of attention as ridge regression that you saw in lecture. Where a model should store what it learns at test time, in its context, in a fixed-size state, or in its weights, is one of the main open questions in the field.

Read the late 2022 paper “What Can Transformers Learn In-Context? A Case Study of Simple Function Classes” （在新标签页中打开） by Garg, Tsipras, Liang, and Valiant. It trains small transformers from scratch to learn simple function classes (e.g. linear functions) in context, and shows that they match optimal estimators such as least squares. The setting fits on a single GPU and is still the standard testbed of the field. Then read “Test-time regression: a unifying framework for designing sequence models with associative memory” （在新标签页中打开） by Wang, Shi, and Fox (2025), which shows that softmax attention, linear attention, and state-space models are different ways of solving the same regression problem in context.

Since 2022 the picture has grown a lot, and many basic questions are still open. Below are some rough directions to get you started. They are not project descriptions: part of the project is to find your own question and write your own pre-proposal. Every project must train something (a model from scratch, a soft prompt or prefix, a LoRA adapter, or a small module of a larger system), so that you can show training and validation curves. Projects that only call an API are NOT accepted.

What algorithm does a trained model run in context? One layer of linear attention trained on linear regression provably implements a step of preconditioned gradient descent （在新标签页中打开）. Deeper or differently parametrized models have been found to do other things (Newton-like steps, principal component regression, Bayesian estimators), and pretrained LLMs do not seem to behave like gradient descent （在新标签页中打开）. Last year we suggested new function classes (e.g. kernelized linear models) and distribution shift between training and test prompts; both now have a literature, so start from what is known and look for what is missing. Re-running Garg et al. on a new function class without a clear question is NOT encouraged.

Which architectures can learn in context, and why? Some tasks separate attention from recurrent models, such as associative recall （在新标签页中打开）, copying, and state tracking, while in-context linear regression does not. Newer layers such as Gated DeltaNet （在新标签页中打开） and Mamba-3 are built as in-context regression solvers, and the flash-linear-attention （在新标签页中打开） library implements most of them. Last year we pointed to the architectural choices of the nanoGPT speedrun （在新标签页中打开）. The speedrun has moved a lot since (FP8, multi-token prediction, a very large n-gram embedding table), so if you go this way, fix a snapshot of the recipe, and keep in mind that some additions store in the weights what a model would otherwise learn in context.

How do training choices shape ICL? ICL can appear suddenly and then fade away with more training （在新标签页中打开）, and whether it appears depends on task diversity （在新标签页中打开） and on the training dynamics （在新标签页中打开）. Little is known about how the optimizer (e.g. Muon vs AdamW) or the parametrization (e.g. muP) changes this. Combining with Project (1) is welcome.

In context or in weights? [Note that small open models such as Qwen3.5 0.8B–9B fit on one GPU.] Prompts with hundreds of examples （在新标签页中打开） compete with fine-tuning （在新标签页中打开） and with test-time training （在新标签页中打开）. When is it better to put information in the context, and when to change the weights? Can one be turned into the other, for example a small set of trained tokens that replace a long context （在新标签页中打开）?

What does an agent learn from the context its harness builds? In an agent, everything the model learns at test time passes through the context assembled by the harness: tool outputs, memory files, summaries, skills, retrieved past trajectories (see Anthropic's note on context engineering （在新标签页中打开）). A harness is a policy that decides what a frozen in-context learner sees under a token budget. In one study, changing the harness changed coding-benchmark results more than changing the model （在新标签页中打开）, yet controlled measurements are rare and most advice comes from blog posts. Simple settings, where the best summary of the context is known, may help. The trained part can be the context itself (a soft prompt, a prefix, learned compression tokens) or a small module of the harness; text-only prompt optimization such as GEPA （在新标签页中打开） is a baseline, not the trained part.

Can small models learn in context from real assembly sequences? Building sequences of LEGO models (public LDraw （在新标签页中打开） data) are long and structured, and each step can be checked exactly for collisions and connections. What do small sequence models learn in context there, for example when a sub-assembly built earlier is repeated? This connects ICL to models that design and assemble physical objects and robots. Can model learn pose estimation tasks in context?

More relevant papers:

In-context Learning and Induction Heads （在新标签页中打开）: Finds the attention circuit behind much of ICL in transformers (induction heads) and shows that it forms in a sudden phase change during training.

Transformers learn in-context by gradient descent （在新标签页中打开）: Builds linear-attention weights that run gradient descent in the forward pass and finds trained models close to them.

Do pretrained Transformers Learn In-Context by Gradient Descent? （在新标签页中打开）: Argues that the gradient-descent view does not carry over to pretrained LLMs. Read it together with the paper above.

What learning algorithm is in-context learning? Investigations with linear models （在新标签页中打开）: Shows that trained transformers behave like gradient descent, ridge regression, or least squares depending on depth and noise.

Data Distributional Properties Drive Emergent In-Context Learning in Transformers （在新标签页中打开）: Shows that ICL emerges from bursty data with many rare classes, and that it competes with in-weights learning.

Training Dynamics of In-Context Learning in Linear Attention （在新标签页中打开）: Exact learning curves for linear attention: the parametrization decides whether ICL appears in one jump or in several.

Strategy Coopetition Explains the Emergence and Transience of In-Context Learning （在新标签页中打开）: Explains why ICL can appear and then disappear later in training.

Zoology: Measuring and Improving Recall in Efficient Language Models （在新标签页中打开）: Shows that associative recall separates attention from efficient recurrent models; comes with a small synthetic benchmark.

Gated Delta Networks: Improving Mamba2 with Delta Rule （在新标签页中打开）: A recurrent layer designed as an online regression solver, now used in recent large open models.

From Markov to Laplace: How Mamba In-Context Learns Markov Chains （在新标签页中打开）: Shows how Mamba learns the optimal predictor for Markov chains in context.

Decomposing Prediction Mechanisms for In-Context Recall （在新标签页中打开）: Toy problems that mix in-context regression on linear dynamical systems with associative recall; finds two mechanisms with different learning dynamics.

Many-Shot In-Context Learning （在新标签页中打开）: With hundreds or thousands of examples in context, results keep improving and pretraining biases can be overridden.

On the generalization of language models from in-context learning and finetuning: a controlled study （在新标签页中打开）: With matched data, ICL generalizes more flexibly than fine-tuning; adding in-context inferences to the fine-tuning data closes the gap.

The Surprising Effectiveness of Test-Time Training for Few-Shot Learning （在新标签页中打开）: Training on the in-context examples at test time beats plain ICL on new reasoning tasks such as ARC.

GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning （在新标签页中打开）: Improves prompts by reflecting on rollouts in natural language, and matches or beats RL with far fewer rollouts.

Meta-Harness: End-to-End Optimization of Model Harnesses （在新标签页中打开）: Treats the harness code itself (what is stored, retrieved, and shown to the model) as something to optimize.

Effective context engineering for AI agents (Anthropic, 2025) （在新标签页中打开）: Defines context engineering and describes compaction, note taking, and sub-agents.

Project Proposal

Please write a one-two page proposal on your project idea; including a high-level overview of your investigation, code bases you’ll build upon, how you will generate training/test data, and your resource budget — an awareness of how much compute power you can muster and use is very important.

Please include all the group members in your proposal, but only one person needs to submit the proposal.

Final Report Requirements:

Must have a coherent story that clearly lays out the question being investigated, cites relevant literature while being a self-contained treatment, and shows clear and systematic analysis/exploration.

Must cite relevant work to put your work into an intellectual context.

For example, you need to provide loss curves for training and validation and compare as needed to tell your story. (Here, it can be fine to partner up with other project teams in this theme to share baselines, etc.) Meaningful error-bars are required.

Provide a link to your github repo with all the code used for this project as well as trained model checkpoints sufficient to efficiently replicate all plots and tables in your paper.

Course Requirements

The shared deadlines in #287 （在新标签页中打开） are November 23 (report V1), November 30 (individual peer reviews), December 7 (report V2), December 9 (second individual peer reviews), and December 12 (final report, code, and checkpoints), all at 10:59 p.m. PT unless updated there. The oral/poster-session details are still TBD. Each person must complete both peer-review rounds; skipping a required review earns a zero on the project.

Please also follow #287 （在新标签页中打开）'s requirements for the self-review checklist and disclosure of generative-AI use. Use this thread for in-context-learning questions and topic-specific teaming, the coordination thread (#267) （在新标签页中打开） to find teammates across topics, and #287 （在新标签页中打开） for administrative questions.

#287
Hi everyone,


It's time to start thinking about your final project. The topics will be released in more detail shortly:

(0) Your own project from outside EECS: only for PhD students outside EECS and their team members (see the Project 0 thread)

(1) Optimizers 

(2) In-context learning

(3) Prompt-injection security

(4) AI Safety 


The topics give rough directions, and finding your own question is part of the work. Every project must train something. Projects that only call an API will not be accepted.


Deadlines:

Same for every topic. All deadlines are at 10:59 PM PT unless noted.


Starting Mon Nov 2: first required 10-minute 1:1 meeting with staff. Bring a printed copy of your draft proposal.

(0) Gireeja Ranade: calendar （在新标签页中打开）

(1) Mert Cemri, calendar （在新标签页中打开）

(2) Matteo Guarrera, calendar （在新标签页中打开）

(3) Sizhe Chen, calendar （在新标签页中打开）

Wed Nov 12: project proposal due, including confirmed teams. Submit on Microsoft CMT and Gradescope.

Starting Mon Nov 16: second required 10-minute 1:1 meeting with staff, for feedback.

Mon Nov 23: final report V1 due for peer review (Monday of Thanksgiving week)

Mon Nov 30: individual peer reviews of V1 due on Microsoft CMT

Mon Dec 7: final report V2 due, addressing the reviews (Monday of RRR week)

Wed Dec 9: second round of individual peer reviews due on Microsoft CMT, with new reviewer assignments

[TBD]: oral exam/poster session

Sat Dec 12: final project submission: report addressing all review comments, plus your GitHub repo with code and checkpoints


Project proposal requirements:

Team: list the 4 students. Only one person submits.

Abstract: at most 6 sentences.

Introduction and Background: a literature review of at least 8 papers related to your project, with at least a one-sentence summary of each paper's key point.

Key questions: a short section with the actual questions you are trying to answer. What problem are you trying to solve, or what phenomenon are you trying to understand or replicate? Back this up with the previous section.

Hypothesis: what do you expect to happen? It can be simple, e.g. "I expect the phenomenon described in reference XYZ to replicate in a different setting." It is important to be expecting something, but it's fine if something different ends up happening.

Methods: how you will test your hypothesis, which experiments you will run, and the expected compute.

Write the proposal in LaTeX and submit it as a PDF, with one-inch margins and 12-point font. See the topics doc for the page length. The purpose of the proposal format is to help structure your thoughts.


Meetings with staff:

Each meeting is 10 minutes per team. Sign up for a slot on the staff calendars listed below. The Nov 2 meetings help guide you toward a good proposal: come with your draft proposal (or proposal options) printed and in hand. The Nov 16 meetings give you feedback to make your work better. Both meetings are required and graded.


This is not a class in which a project can be completed overnight or even in a week. Start early and expect all four team members to work hard. Deep learning being what it is, there will be setbacks, confusion and lots of iteration, on top of a lot of reading and learning on your own.


Peer review:

Peer reviews are done on Microsoft CMT, in two rounds. Each of you writes individual reviews of other teams' reports: V1 over the Thanksgiving break (due Nov 30), and V2 with new reviewer assignments (due Dec 9). Use the reviews you receive to improve your report. We'll release peer-review guidance and rubrics later.


Oral presentation/poster session:

We will host a poster session/oral presentation session. Each team presents their project to course staff, and we ask questions about your code, your results and the details of your work. Every team member should be ready to answer questions about any part of the project. [TBD: date, length, sign-up, what to bring]


Staff calendars and office hours by topic:

Use these calendars to sign up for the 1:1 meetings and to find the best office hours for your topic. Students doing cybersecurity projects are strongly encouraged to meet with Sizhe. 


(0) Gireeja Ranade: calendar （在新标签页中打开）

(1) Mert Cemri, calendar （在新标签页中打开）

(2) Matteo Guarrera, calendar （在新标签页中打开）

(3) Sizhe Chen, calendar （在新标签页中打开）


Other logistics:

Teams must have 4 people. Exceptions for 3 or 5 can be granted if someone drops or to pick up students left without a team, but this case is very very rare.

Project grades are shared by the team but multiplied by the quality of your individual peer reviews. Do a bad job as a reviewer and you lose points. Skip a required peer review and you get a zero on the project. You can also earn participation credit for helpful interaction with other teams in your topic's Ed thread.

This thread is for administrative and logistics questions. Each topic will have its own thread for teaming and discussion.


Guidance and advice:

You are encouraged to use generative AI tools to write and modify code, to seed your literature search with "Deep Research"-style tools, and to sharpen your ideas. Cite the tools you used and add an appendix describing how you used them during your project. The appendix doesn't count toward page limits.


It is also fine to use generative AI tools to help with writing, figures, etc., as long as you describe this in the appendix. However, hallucinations in your report will be severely penalized: you are expected to understand and double-check everything you submit, and the oral exam will ask about your code. The point of permitting these tools is to improve overall quality and to help you do the best project you can in the given time. We live in 2026! 


With each submission you'll also submit a peer-review form for your own project: a checklist that helps get your submission in good shape before others review it.