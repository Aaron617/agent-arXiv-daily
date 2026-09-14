# Agent arXiv Daily

**Last Updated:** 2026-09-14 04:49:19

**Total Papers:** 107

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>ARC: Autonomous Robotics Compliance A Three-Layer Governance Architecture for Deployed Autonomous Systems</strong> - Tord Eide, Einar Holt - [[pdf]](https://arxiv.org/pdf/2609.12932)</summary>

**Abstract:** Proposed governance framework for autonomous robotic systems, introducing a three-layer compliance architecture (ARC) instantiated through model safety validation, cognitive certification benchmarks, and operational authorization standards.

**arXiv ID:** 2609.12932
</details>

<details>
<summary><strong>Size Doesn't Matter: Material-State Reinforcement Learning for Excavator Transferable Soil Manipulation</strong> - Lennart Werner, Pol Eyschen, Sean Costello, Pierluigi Micarelli, Andrei Cramariuc, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2609.12677)</summary>

**Abstract:** Earthmoving tasks such as excavation, backfilling, or embankment construction require deliberate repositioning of deformable soil. For these tasks, human operators use all shovel faces, while autonomous systems so far are limited to excavation and dumping. Current methods often rely on heuristic models but do not incorporate soil mechanics. We address this shortcoming by using Reinforcement Learning in a GPU-parallelized Material Point Method particle simulation. Our controllers are conditioned on material state such as shape and compactness, enabling skills that use multiple contact faces of the tool and displace material both inside and outside of the shovel. To use the same learned weights across machines, our policies operate in a normalized end-effector space and are deployed through a calibrated machine interface. We evaluate this calibrated transfer on an 11.5t hydraulic excavator and a 500g tabletop robot. We validate performance through autonomous construction of a 42m long, 2.1m high embankment in 45min, executing 201 individual policy strokes without failure, retry, or operator intervention. In a direct comparison, the autonomous controller matches an expert operator's progression speed and produces a higher, more consistent embankment. Additional qualitative backfilling and compaction experiments demonstrate the material-state awareness and calibrated transfer across machines.

**arXiv ID:** 2609.12677
</details>

<details>
<summary><strong>Plans They Abandon, Reports They Author: The Narrative Layer of Autonomous Agents</strong> - Obada Kraishan, Kulsawasd Jitkajornwanich - [[pdf]](https://arxiv.org/pdf/2609.12205)</summary>

**Abstract:** When a coding agent finishes a task, the developer reviews a summary the agent wrote about itself, not a display someone designed. We ask how much of the agent's work that summary carries, and whether it drifts toward the plan the agent stated when execution departed from it. Across 5,851 real developer sessions and 355,942 tool calls, a self-report referred to about one action in eleven, and a reader working from the report alone recovered roughly a fifth of the action log. Neither figure depended on whether the session later needed human correction. Reports did not generally resemble the stated plan more than the executed one, but they did so increasingly as execution diverged from the plan. We hand-validate both measurement steps that use a language model, report the one that failed alongside the one that passed, and draw conclusions only from measures that survived.

**arXiv ID:** 2609.12205
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (29 papers)</h2></summary>

<details>
<summary><strong>GTA: Graph Theory Agent and Benchmark for Algorithmic Graph Reasoning with LLMs</strong> - Zixiang Xu, Yanbo Wang, Chenxi Wang, Lang Gao, Zirui Song, Yue Huang, Zhaorun Chen, Xiangliang Zhang, Xiuying Chen - [[pdf]](https://arxiv.org/pdf/2609.12265)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly asked to reason over structured data such as graphs, yet how reliably they can carry out multi-step graph algorithms in language remains unclear. Existing evaluations tend to use simple tasks on small graphs, to score code generation rather than reasoning over the graph itself, or to fix a single input format. We introduce Graph Theory Bench (GT Bench), a benchmark covering 24 classical graph problems in 44 task-structure settings, with over 100,000 examples across four representations: natural language, structured language, adjacency list, and adjacency matrix. Evaluating eight LLMs on GT Bench shows that accuracy is strongly tied to the input representation, that the best representation shifts with graph density, size, and topology as well as with the model, and that this sensitivity persists, attenuated, in the strongest reasoning models. Building on these observations, we propose the Graph Theory Agent (GTA), which pairs a preference-trained representation selector with plan-and-decompose scaffolding around a frozen executor LLM. GTA lifts Phi-4 from 53.5% to 69.1% on the benchmark's easy split and from 33.0% to 41.5% on its hard split, outperforming eight prompting and agent baselines, and transfers without retraining to GraCoRe and NLGraph. Code for benchmark generation and evaluation: this https URL. The project homepage is available at this https URL.

**arXiv ID:** 2609.12265
</details>

<details>
<summary><strong>Affective Agent: On-Device Personalized Intervention Reasoning for Wearable Systems</strong> - Reina Mun, Zishen Wan, Vijay Janapa Reddi - [[pdf]](https://arxiv.org/pdf/2609.12322)</summary>

**Abstract:** Affective computing has advanced wearable state inference, but on-device reasoning about whether, when, and how to intervene remains challenging. We present Affective Agent, a three-layer reference architecture for personalized intervention reasoning under uncertainty on wearable-class hardware. It combines a compact sub-billion-parameter language model with physiological evidence, context, and user history to decide whether, when, and how to intervene, without cloud dependency or per-user retraining. The architecture is organized into three interacting layers (perception, personalization, and reasoning), adapting to individual users through host-managed structured memory evolution rather than per-user weight updates. We instantiate Affective Agent in indoor environmental quality control and evaluate it on held-out, simulator-generated longitudinal scenarios spanning physiological variation, context, signal quality, and intervention history. Results show that memory-driven personalization and two-pass structured reasoning improve intervention decisions within this synthetic evaluation. By moving the decision layer on-device, this work demonstrates a path from wearable state inference toward closed-loop, personalized intervention on wearable-class hardware.

**arXiv ID:** 2609.12322
</details>

<details>
<summary><strong>SoK: Rethinking Jailbreaking in the Era of Agentic AI: Attacks, Defenses, and Practical Consideration</strong> - Md Jueal Mia, Yanzhao Wu, Selcuk Uluagac, M. Hadi Amini - [[pdf]](https://arxiv.org/pdf/2609.12413)</summary>

**Abstract:** Large language models (LLMs) are rapidly evolving from conversational assistants into agentic AI systems that reason, plan, invoke tools, maintain persistent memory, communicate with other agents, and execute multi-step tasks. At the same time, modern models exhibit substantially stronger native safety alignment than earlier generations on which many jailbreak attacks and defenses were originally studied. This shift raises a fundamental question: \textit{which established jailbreak-security findings remain valid in the era of modern LLMs and agentic AI?} We address this question through a Systematization of Knowledge (SoK) that reframes jailbreak security around the full agentic execution pipeline. We develop unified taxonomies of attacks and defenses spanning user interaction, planning and reasoning, memory, tool use, and inter-agent communication, and introduce a security--utility--efficiency evaluation framework that separates native harmful-prompt safety, adversarial jailbreak robustness, and agent-level security outcomes. We further conduct a controlled empirical study of representative attacks and defenses within a common agentic framework. Our results reveal three important gaps. First, strong native alignment does not imply robustness to adversarial jailbreaks. Second, defense effectiveness is highly model-, attack-, and component-dependent and can come at substantial cost in over-refusal, utility, and latency. Third, low final-response attack success can mask severe intermediate compromise: planning, memory, and tool interactions may remain unsafe even when the final response is successfully filtered. These findings motivate a shift from response-centric jailbreak defense toward cross-layer, execution-aware security that protects agent state, component transitions, and external actions while preserving practical utility and efficiency.

**arXiv ID:** 2609.12413
</details>

<details>
<summary><strong>Skill Issue: Lessons from Optimizing Repository SKILLs for Coding Agents</strong> - Mykhailo Kozyrev, Andrei Kozyrev, Anton Podkopaev - [[pdf]](https://arxiv.org/pdf/2609.12742)</summary>

**Abstract:** Coding agents increasingly read repository knowledge from SKILLs --- plain \texttt{.md} files versioned alongside the code. Recent work synthesizes these files automatically, by optimizing the document against a benchmark. A bare repository comes with no benchmark, and the synthetic tasks prior work builds are small enough that a capable agent saturates them with no document at all. We mine harder tasks --- merged pull requests of the repository, reverted at a single frozen base commit; and score a candidate document by whether the same agent does better with it than without it. On three Kotlin repositories, the documents GEPA finds raise this score by $4.9$pp on average, and the ones SkillOpt finds leave it where it started, $0.1$pp above the seed. The GEPA gain matches what prior work reports with the same optimizer, and at the dataset size a single repository supplies it cannot be separated from the agent's run-to-run variance; settling that would take more tasks than one repository's history yields. The documents themselves read better than the score: a maintainer of one repository found in them knowledge one only gets by working in the project.

**arXiv ID:** 2609.12742
</details>

<details>
<summary><strong>K-Bench: A Benchmark for LLM Unlearning in Agentic Deployments</strong> - Guangsheng Yu, Yanna Jiang, Qin Wang, Baihe Ma, Xu Wang - [[pdf]](https://arxiv.org/pdf/2609.12808)</summary>

**Abstract:** Unlearning benchmarks such as TOFU and MUSE certify forgetting by reading the model's final answer, where a model that refuses to answer already counts as having forgotten. We show that this model-level certificate does not transfer once the model is deployed as an agent. We introduce K-Bench, a benchmark that scores LLM unlearning under agentic deployment. K-Bench inspects all six channels a ReAct agent exposes, including its chain-of-thought (CoT), tool calls and tool observations, and elicited summary. A query counts as leaked if the secret appears in any of them. Each experiment places the secret in exactly one of the agent's three sources (the weights, the prompt, or the retrieval store). The K-Score is computed separately for each source and credits forgetting only when the agent remains usable. Clearing the answer channel does not make the secret unrecoverable. On structured retrieval, the secret stays verbatim in the tool-observation channel and the aggregate leak rate is unchanged. When the secret lives in the prompt or the retrieval store, TOFU and MUSE report no leakage, while the deployed agent still leaks it on 22--86\% of queries. When the secret is in the weights, none of the twenty evaluated published methods demonstrably removes it, and only an input-corruption intervention reaches selective forgetting under the evaluated observer. The top-ranked method changes across base models. A refusal-tuning method resists the evaluated extraction without verified knowledge removal.

**arXiv ID:** 2609.12808
</details>

<details>
<summary><strong>Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval</strong> - Junghyun Min, Huseyin Uzunalioglu, Mohamed Trabelsi - [[pdf]](https://arxiv.org/pdf/2609.13073)</summary>

**Abstract:** Recent breakthroughs in LLM-based systems and their abilities in problem solving and coding have allowed progress in the AI for Science paradigm, potentially replacing human roles in machine learning (ML) research. However, while several frameworks of fully autonomous end-to-end ML research have been proposed, successful implementations of them are often limited to problems with narrow search spaces, like language modeling or biomedical ML benchmarks. In this paper, we explore how autonomous research can be adapted to solve open-ended, industry-grade ML problems, by considering a case study: telecom ticket retrieval, an open-ended task with degrees of freedom in representation, architecture, and training data generation. We discover that autonomous research for open-ended problems with commercial and open-source agents shows both promise and limitations: while autonomous research can excel in narrow hyperparameter optimization, it lacks human-like intuition and creativity and requires operational overhead. Even with minimal human supervision, autonomous research can reach $90\%$ of state-of-the-art performance (0.34 vs. 0.38 Recall@1) in a much shorter time period (10 weeks vs. 10 months of human work) at a modest cost (up to \$200 per Cursor campaign). Our empirical evidence recommends that human researchers and autonomous research frameworks work together for best results in ML research.

**arXiv ID:** 2609.13073
</details>

<details>
<summary><strong>Embodied-BenchForge: A Closed-Loop Agentic Workflow for Embodied Benchmark Construction</strong> - Baoyang Jiang, Fengchun Zhang, Leyuan Wang, Haotian Li, Yida Wang, Zhe Ji, Jinshan Lai, Xi Ren, Danyang Li, Zheng Yang, Jianwei Hu, Qiang Ma - [[pdf]](https://arxiv.org/pdf/2609.13082)</summary>

**Abstract:** Agentic systems offer a promising way to automate embodied benchmark construction, but existing approaches typically cover isolated stages or remain specialized to predefined environments and task families. More importantly, multi-step construction produces dependent intermediate artifacts that are often passed downstream without artifact-specific verification, allowing local defects to propagate into the final benchmark. We present Embodied-BenchForge, an agentic framework that transforms user-specified evaluation intents into complete embodied benchmark artifacts. It formulates construction as Closed-Loop Benchmark Synthesis, integrating forward artifact synthesis with backward verification and repair. Skill-Orchestrated Artifact Synthesis composes typed and reusable skills into executable workflows, while an artifact dependency graph records intermediate outputs and their dependencies. Requirement-Guided Verification and Repair applies artifact-specific contracts throughout construction and uses provenance to trigger local re-execution or upstream rollback when verification fails. Embodied-BenchForge constructs six benchmarks covering diverse embodied scenarios in the Offline EQA Track, together with one interactive benchmark containing 220 executable tasks in the Interactive Embodied Track. Evaluations of representative MLLMs and embodied agents show that the benchmarks distinguish model capabilities in both observation-based understanding and closed-loop execution. Quality assessment and ablations validate benchmark quality and the effectiveness of verification and repair, while repair and skill-reuse analyses demonstrate efficient localized recovery and cross-benchmark reusability.

**arXiv ID:** 2609.13082
</details>

<details>
<summary><strong>When Agent Metrics Measure Different Things: An Evidence-Grounded Audit of the Praxa AI Pipeline</strong> - Stefan G. Creadore, Peyton Woakz - [[pdf]](https://arxiv.org/pdf/2609.12017)</summary>

**Abstract:** Agent evaluations can be numerically correct while measuring a different construct from the one implied by their labels. We present a retrospective measurement audit of selected Praxa AI implementation files, historical evaluation artifacts, and operational records. A 139-case offline routing report contains 112 passes and 27 failures despite zero gating failures, because known gaps are explicitly exempted from the gate. An identifier-free export represents 8,843 tool-attempt rows: 8,395 recorded durations and 448 missing values. Of the durations, 121 equal the signed 32-bit maximum and carry abandoned-client labels; inspected database code clamps elapsed lifecycle age. The pooled recorded 99th percentile is 2,147,483,647 ms, versus 38,118.31 ms among server-observed completed calls. This is a stratum contrast, not a treatment effect. In a documented single-trajectory compaction pilot, the reported follow-up input reduction is 94.39%, but the reduction across the trigger and follow-up calls together is 46.54%. We reproduce the descriptive calculations, verify 91 timing statistics through a separate weighted rational-arithmetic implementation, and execute 13 scoring-function and 12 analysis-verifier tests. Finite-completion bounds show how missing durations limit all-row timing statements without imputing values. The contribution is a source-linked case study and reusable verification package for separating gate policy, lifecycle timing, and request-level accounting from broader agent-performance claims. Historical provider runs and the full current pipeline were not independently reproduced; general capability superiority and population-level statistical significance are not established.

**arXiv ID:** 2609.12017
</details>

<details>
<summary><strong>Reality Is the Final Verifier: On Two Key Gaps in Agentic Software Engineering</strong> - Alexander Krentsel, Shubham Agarwal, Mert Cemri, Shu Liu, Sidharth Sankhe, Ziming Mao, Matei Zaharia, Ion Stoica - [[pdf]](https://arxiv.org/pdf/2609.12039)</summary>

**Abstract:** Software development follows an implementation-verification loop in which developers or agents iteratively revise an implementation until an evaluator, such as a test suite, accepts it. The evaluator checks the implementation against a set of requirements under a model of the deployment environment. Yet even a formal proof that the implementation satisfies the requirements under the model cannot guarantee acceptable behavior after deployment. Requirements only approximate stakeholder intent, and the model only approximates the real deployment environment. We call these together - requirement gap and model gap - the two-gap framework, which unifies the main failure modes of agentic software engineer-ing: reward hacking exploits omissions in the requirements or model, while hallucination widens the gaps by fabricating requirements or environment assumptions.
Because neither gap can generally be certified closed in an open, changing world, the goal shifts from closing them to continuously narrowing them. We therefore propose an assurance-revision loop that uses deployment evidence to revise the requirements, model, or evaluator when stakeholders reject the resulting behavior. We then cast assured agentic development as a resource-allocation problem over human judgment, agent capability, and compute. The two principal bottlenecks mirror the two gaps: human judgment for the requirement gap and faithful, costly evaluation for the model gap. Reality remains the final verifier: acceptable behavior under actual deployment conditions is the ultimate test, while predeployment evaluations remain proxies for it.

**arXiv ID:** 2609.12039
</details>

<details>
<summary><strong>UniPart: Towards Zero-shot Language-Grounded 3D Part Segmentation for Embodied Interaction</strong> - Xinqiang Yu, Zekun qi, Jiawei He, Wenyao Zhang, Xuchuan Chen, Guaocai Yao, Li Yi, Zhaoxiang Zhang, He Wang - [[pdf]](https://arxiv.org/pdf/2609.12898)</summary>

**Abstract:** Fine-grained robotic manipulation depends on understanding parts, not only whole objects. Existing 3D foundation models tend to be either generalized but object-aware, or part-aware but limited to closed-set taxonomies, which weakens zero-shot transfer. We study text-conditioned 3D part segmentation, where a free-form phrase selects a functional part on point cloud. We introduce UniPart, a feed-forward cross-modal 3D Transformer that conditions CLIP text embedding. To scale supervision, we build LangPart-1M with 160K+ Objaverse assets and 8M text to part pairs using multi-view consistent part generation. We further manually label a high-quality subset, LangPart-4K, for fine-tuning and evaluation. UniPart achieves strong zero-shot results on open-vocabulary part benchmarks and transfers to language-conditioned part grasping in real world.

**arXiv ID:** 2609.12898
</details>

<details>
<summary><strong>MP-Bench: Evaluating Voice Agents as a Multiparty Conversation Participant</strong> - Yi-Jen Shih, Shih-Yun Shan Kuan, Guan-Ting Lin, Kai-Wei Chang, Siddhant Arora, Shu-wen Yang, Abdelrahman Mohamed, Shinji Watanabe, Hung-yi Lee, David Harwath - [[pdf]](https://arxiv.org/pdf/2609.13076)</summary>

**Abstract:** Conversational voice agents have advanced significantly, offering increasingly natural human-machine interactions through both cascaded and end-to-end architectures. However, while recent benchmarks extensively evaluate dyadic interactions and passive audio comprehension, they largely overlook a prevalent real-world scenario: multi-party conversations. Evaluating agents in these settings is fundamentally more challenging than in dyadic interactions due to the exponentially greater conversational complexity. For voice agents to integrate seamlessly into human group dynamics, they must not only generate contextually appropriate responses but also demonstrate a nuanced understanding of open turn-taking. To address this gap, we introduce Multiparty Bench (MP-Bench), the first benchmark specifically designed to objectively evaluate conversational speech systems as active participants within multi-party contexts. MP-Bench assesses agent behavior along two primary dimensions: turn-taking awareness and response appropriateness. Additionally, we incorporate comprehension-based question-answering tasks as a complementary evaluation. By benchmarking 12 voice agents, we find that real-time voice agents stay at or below 22% on multiparty comprehension and remain near chance on multiparty turn-taking, exposing an open challenge for real-time voice agents under multiparty scenario.

**arXiv ID:** 2609.13076
</details>

<details>
<summary><strong>Capable but Careless: Do Computer-Use Agents Follow Contextual Integrity?</strong> - Anmol Goel, Iryna Gurevych - [[pdf]](https://arxiv.org/pdf/2606.23189)</summary>

**Abstract:** Computer-use agents (CUAs) now act on a user's behalf across personal applications such as email, calendars, and to-do lists. This cross-application access is useful, but it also creates a privacy risk that has been largely overlooked: when an agent works in one context, it can pull in information from another that is inappropriate in that context. Hence, we introduce AgentCIBench, an evaluation harness that turns this risk into executable, deterministically scored scenarios. We target three common failure modes in CUAs: visual co-location, where the agent pulls in prohibited items that sit next to the task target in the UI; task-ambiguity overshare, where the agent dumps dense personal state in response to an under-specified prompt; and recipient misalignment, where the agent sends content to an addressee for whom it is inappropriate. We evaluate 15 frontier agents and find a surprisingly high failure rate: 11 of 15 leak on more than 50% of scenarios, with an average leakage of 67.9%, and the same failures persist when agents act end-to-end in the environment to complete the task. We release AgentCIBench to encourage the development of safer computer-use agents and position contextual disclosure testing as a pre-deployment safety check.

**arXiv ID:** 2606.23189
</details>

<details>
<summary><strong>Measurement Without Validity: The Compounding Reliability Problem in Agentic AI Evaluation</strong> - William Caban - [[pdf]](https://arxiv.org/pdf/2608.00794)</summary>

**Abstract:** Agentic AI evaluation pipelines produce benchmark scores that justify deployment decisions, safety certifications, and regulatory compliance claims. No formal framework has yet characterized how validity degrades across the stages of these pipelines. We present a three-layer compounding validity model, $V_{total} \leq V_1 \times V_2 \times V_3$, that captures multiplicative degradation across task generation ($V_1$), human-simulator calibration ($V_2$), and automated judgment ($V_3$). Under empirically grounded estimates, a pipeline retaining 70% validity at each stage is at most 34% valid against the intended construct (range 0.17-0.54 across the empirical estimate bounds).
We examine the model's predictions against a structured survey of 55 published agentic evaluation papers, finding that approximately 82% of papers in this purposive sample apply structurally mismatched, incomplete, or absent inter-rater reliability (IRR) metrics, a pattern consistent with systematic $V_3$ collapse. We further identify empirical evidence of $V_1$ failures (task validity flaws in 7 of 10 popular benchmarks) and $V_2$ miscalibration (up to 9 percentage points inter-simulator variance, with systematic demographic disparities for non-Standard American English speakers).
We derive eight prescriptions grounded in psychometric science and domain-stratified reliability thresholds (ICC $\geq$ 0.70; $\alpha \geq$ 0.67/0.70/0.80 by consequence level) that practitioners and benchmark authors can apply immediately. The framework provides a tractable knowledge-based tool for diagnosing and correcting evaluation pipeline validity before deployment decisions are made.

**arXiv ID:** 2608.00794
</details>

<details>
<summary><strong>SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?</strong> - Yuqiao Tan, Shizhu He, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2609.09113)</summary>

**Abstract:** While research on recursive self-improvement (RSI) has predominantly automated model training pipelines, reliable autonomous development demands a missing pillar: post-hoc monitoring and auditing to understand what models learn and ensure safe alignment. Mechanistic interpretability tools are essential to bridge this gap, among which Sparse Autoencoders (SAEs) serve as a cornerstone by isolating interpretable features for model inspection and steering. In this paper, we introduce SAEScientist-Bench to evaluate whether AI agents can act as scientists utilizing SAE tools for autonomous mechanistic discovery. Given a target concept, an agent designs contrastive probes and navigates a Gemma Scope dictionary of 131K+ features in Gemma-2-9B-IT to discover the optimal feature, evaluated against curated expert reference features anchored on Neuronpedia across activation rank, concept selectivity on contrastive texts, and causal steering. Across 10 agent configurations and 20 tasks, frontier agents demonstrate genuine discovery capabilities and lead different evaluation dimensions, but remain well behind the expert baseline, approaching expert levels on separating target concepts from contrastive controls while lagging substantially in causal generation steering. Further analysis reveals that although agents can design contrasts to rule out spurious candidates, they frequently misinterpret experimental measurements. These results establish experimental model understanding as a measurable capability for closed-loop autonomous AI R&D. Our code is available at this https URL.

**arXiv ID:** 2609.09113
</details>

<details>
<summary><strong>Mr.LHDR: A Benchmark for Multimodal Real-World Long-Horizon Deep Research Agents</strong> - Minghao Guo, Meng Cao, Sui Zhao, Siyu Ning, Xin Wang, Haoze Zhao, Jiaxuan Yang, Haihong Hao, Mingfei Han, Shunlin Rong, Haijun Wu, Xiaodan Liang, Xiaojun Chang - [[pdf]](https://arxiv.org/pdf/2609.11318)</summary>

**Abstract:** Deep research agents are increasingly capable of web search, tool use, multimodal evidence analysis, and information synthesis. However, existing benchmarks mainly evaluate medium-horizon exploration and rarely test whether agents can sustain long, dependency-heavy research processes. We introduce Mr. LHDR (Multimodal real-world Long-Horizon Deep Research), a benchmark for evaluating real-world deep research over long, irreducible chains of interdependent evidence across eight categories. Each question is constructed from a hidden Node-Relation graph and requires an average of 12.1 necessary intermediate conclusions with a mean dependency depth of 10.4 before reaching a short, unique, and verifiable answer. Questions incorporate multimodal evidence, including images, maps, PDFs, logos, charts, tables, and video frames, with at least one non-text element that changes the reasoning state. Mr. LHDR evaluates both final answers and the correctness of intermediate conclusions under annotated dependencies. We evaluate general models, deep research systems, and agent frameworks using Overall Accuracy (OA), Strict Accuracy (SA), Checklist Score (CS), and Dependency-Aware Checklist Score (DACS). Results show that even the strongest system achieves only 43.1% OA and 34.3% SA, indicating that final-answer accuracy substantially overestimates complete research success. Removing images reduces DACS by 12.6 points, demonstrating the importance of multimodal evidence, while SA consistently declines as reasoning chains become longer. These findings reveal sustained, dependency-consistent evidence integration, rather than isolated fact retrieval, as a key bottleneck for current deep research agents.

**arXiv ID:** 2609.11318
</details>

<details>
<summary><strong>The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation</strong> - Makoto Fukushima, Hua-Dong Xiong, Ehsan Moradi Pari - [[pdf]](https://arxiv.org/pdf/2609.11489)</summary>

**Abstract:** Cooperative AI agents are evaluated against other AIs, yet human cooperation relies on implicit conventions -- shared protocols for reading meaning beyond the literal message -- which AI-AI benchmarks may not capture. We propose the convention gap, the difference between the failure probability predicted from the literal content of communication and the observed failure rate, as a metric of implicit communication. In the card game Hanabi, the finite deck and deterministic hint constraints make this posterior exactly computable. We replayed about 101,000 play actions from three public datasets of human-human (an online Hanabi platform), AI-AI (HOAD), and human-AI (HanabiData) games. The gap was +26.2 percentage points (pp) in human pairs, -0.7 pp in AI pairs, and +16.4 pp in human-AI pairs, and was concentrated on plays of cards that had received no hints (+46 pp in human pairs). Within human-AI play, the literal information available to humans was similar across the three AI partners (mean predicted failure 38-41%), but human failure rates ranged from 14.4% to 34.4% and the gap from +24.1 to +6.2 pp; the partner eliciting the largest gap produced the fewest human failures. Game score carried different information: it depended on each corpus's roster composition, whereas the gap separated human from AI play at the agent level. As a known-answer check, Off-Belief Learning agents, whose convention content is controlled by construction, gave a gap of +1.6 pp at the convention-free level, rising monotonically to +21.7 pp. These results suggest that convention compatibility, rather than AI-AI performance, may predict an AI's effectiveness with human partners.

**arXiv ID:** 2609.11489
</details>

<details>
<summary><strong>Right Family, Wrong Skill: Evaluating Risk Exposure in Agent Skill Retrieval</strong> - Jiandong Ding, Honglei Ji, Ming Liu, Tao Duan - [[pdf]](https://arxiv.org/pdf/2606.10388)</summary>

**Abstract:** Agent skill libraries are becoming routable software assets: a retrieved skill can contribute instructions, scripts, resource bindings, and execution assumptions to an agent. This makes retrieval failures more specific than broad irrelevance. A system can find the right capability family yet expose the wrong same-capability representative. We study this failure as same-capability risk-exposure retrieval. Each benchmark unit pairs a helpful skill with a query-specific risky sibling that shares the capability family but differs on an execution-controlling contract, such as the required resource, precondition, procedure, or artifact. We introduce SameCapRisk-Bench, an auditable benchmark with 1,190 skill-risk units and 1,686 evaluation query cases: 694 marked-sibling units under public library pressure and 496 hard role-flip units where the same two skills swap helpful/risky roles across paired queries. The release records admission evidence, cue/leakage checks, source hashes, family relations, and fixed candidate pools. The benchmark reports helpful ranking together with harmful sibling rate (HSR@K), the top-K exposure of the marked risky sibling. On this benchmark, public SkillRouter, SkillRet, and R3-Skill retrieve helpful skills at high Recall@3 (0.848--0.888) but also expose marked risky siblings frequently (HSR@3 0.346--0.372). A fully public score-and-cluster pipeline lowers HSR@3 to 0.128--0.182, with Recall@3 of 0.713--0.776. Under a benchmark-trained reference scorer, public text-cluster and controlled resolvers reach HSR@3 0.012 and 0.007; the latter attains Recall@3 0.833. Skill retrieval should therefore report both capability matching and same-family risk exposure, with HSR serving as a targeted exposure certificate for fixed skill libraries.

**arXiv ID:** 2606.10388
</details>

<details>
<summary><strong>EvoHarnessBench: Can Your Agents Keep Pace with an Evolving Harness?</strong> - Zixuan Ke, Vaidehi Patil, Haizhou Shi, Yang Li, Ye Liu, Sarath Shekkizhar, Anurag Koul, Jiayu Wang, Xuan Phi Nguyen, Semih Yavuz, Mohit Bansal, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2609.04280)</summary>

**Abstract:** Modern LLM-based agents operate through a harness of tools, reusable skills, and specialist agents that shapes what they observe and what they can do. In practice, this harness continually evolves as new capabilities are added. We introduce EVOHARNESSBENCH, a benchmark for evaluating agents under controlled harness evolution across three axes (tools, skills, and agents). Unlike existing continual-learning benchmarks for agents, which typically place non-stationarity (i.e., what changes over time) in the task stream while keeping the harness fixed, EVOHARNESSBENCH places non-stationarity in the externally supplied harness itself. It contains 17 multi-stage harness streams constructed deterministically from verifier-based benchmarks, comprising 802 tasks, 520 tools, 42 skills, and 62 agents. We evaluate two complementary settings corresponding to the central challenges of harness evolution: deployment evaluation, which isolates retention of previously accessible competence as the harness expands, and self-evolving adaptation evaluation, which tests whether accumulated experience remains useful as new capabilities are introduced. Our results reveal three persistent gaps. First, harness expansion alone can degrade performance on previously solved tasks, producing harness-induced forgetting. Second, gains from self-evolving adaptation remain inconsistent across stages of harness evolution, capability axes, and environments. Third, retention and adaptation can pull in different directions: preserving earlier competence does not necessarily improve adaptation to newly introduced capabilities, and vice versa. These results establish harness evolution as a distinct challenge for building agents that can keep pace with an evolving harness while preserving previously effective behavior.

**arXiv ID:** 2609.04280
</details>

<details>
<summary><strong>Continue, Adapt, or Yield: In-Turn Adaptation to Overlapping Speech in Full-Duplex Agents</strong> - Yunqi Lu, Tyler Baumgartner, Nikhil Johri, Brandon Tai, Candice Fan, Luc Debaupte, Ruben Aguilar, Bill Wang, Yi Zhong - [[pdf]](https://arxiv.org/pdf/2609.13117)</summary>

**Abstract:** Full-duplex evaluation often emphasizes whether an agent keeps speaking or stops. That binary cannot express a third response humans use routinely: continuing to speak while incorporating what the listener just contributed. The contribution may be a missing word, a correction or a clarification. We introduce Duplex Cue, an evaluation of this \emph{in-turn adaptation} in full-duplex voice agents. Duplex Cue separates listener intent (backchannel, collaboration, or interruption) from speaker behavior: continuing unchanged, adapting within the turn, or yielding. Adaptation includes acknowledgment as well as content revision. In a single-model case study using 300 human-confirmed cues from unscripted English conversations, we compare recorded human responses with PersonaPlex continuations generated while replaying the listener's audio. We retain 208 pairs with the ongoing speaker active at cue onset and a scorable response in each condition. On the 66 collaborative pairs, recorded speakers adapt in 68.2\% of cases, compared with 34.8\% for PersonaPlex. The model otherwise continues unchanged (42.4\%) or yields (22.7\%). These findings show why evaluating natural voice interaction requires measuring how an agent responds to a listener's contribution as well as whether it keeps speaking.

**arXiv ID:** 2609.13117
</details>

<details>
<summary><strong>Is Bash All You Need? An Empirical Study of Tool Interfaces for Enterprise Digital Worker Agents</strong> - Hazel Mak, Susheel Suresh, Sahil Bhatnagar, Barry Wang, Chhaya Methani, Alejandro Gutierrez Munoz - [[pdf]](https://arxiv.org/pdf/2609.11999)</summary>

**Abstract:** In this study, we examine whether a general shell can outperform specialized tools on enterprise tasks. Shell-based agents have shown strong results in coding, but enterprise work also involves moving between applications and services, coordinating with coworkers, and performing professional analysis. We compare five tool interfaces on TheAgentCompany and APEX-Agents using Opus-4.8 and GPT-5.5: typed tools, typed tools plus bash, bash alone, bash with persistent agent-synthesized tools, and programmatic tool calling (PTC), which runs programs whose actions are restricted to a typed tool catalog. Bash alone outperforms typed tools on both benchmarks, improving score by 21.8-24.5 pp on TheAgentCompany and 4.8-7.4 pp on APEX-Agents while using 19-72% fewer total tokens. Adding typed tools or persistent tool synthesis to bash produces no detectable pooled score gain. PTC uses fewer tokens than direct typed calls with broadly similar task performance, but generally underperforms bash alone in both quality and cost efficiency. For enterprise practitioners, these results favor bash alone when arbitrary execution can be isolated and PTC when security or compliance policies require a fixed tool catalog.

**arXiv ID:** 2609.11999
</details>

<details>
<summary><strong>UrduFactCheck: An Agentic Fact-Checking Framework for Urdu with Evidence Boosting and Benchmarking</strong> - Sarfraz Ahmad, Hasan Iqbal, Momina Ahsan, Numaan Naeem, Muhammad Ahsan Riaz Khan, Arham Riaz, Muhammad Arslan Manzoor, Yuxia Wang, Preslav Nakov - [[pdf]](https://arxiv.org/pdf/2505.15063)</summary>

**Abstract:** The rapid adoption of Large Language Models (LLMs) has raised important concerns about the factual reliability of their outputs, particularly in low-resource languages such as Urdu. Existing automated fact-checking systems are predominantly developed for English, leaving a significant gap for the more than 200 million Urdu speakers worldwide. In this work, we present UrduFactBench and UrduFactQA, two novel hand-annotated benchmarks designed to enable fact-checking and factual consistency evaluation in Urdu. While UrduFactBench focuses on claim verification, UrduFactQA targets the factuality of LLMs in question answering. These resources, the first of their kind for Urdu, were developed through a multi-stage annotation process involving native Urdu speakers. To complement these benchmarks, we introduce UrduFactCheck, a modular fact-checking framework that incorporates both monolingual and translation-based evidence retrieval strategies to mitigate the scarcity of high-quality Urdu evidence. Leveraging these resources, we conduct an extensive evaluation of twelve LLMs and demonstrate that translation-augmented pipelines consistently enhance performance compared to monolingual ones. Our findings reveal persistent challenges for open-source LLMs in Urdu and underscore the importance of developing targeted resources. All code and data are publicly available at this https URL.

**arXiv ID:** 2505.15063
</details>

<details>
<summary><strong>GRACE-DS: a Guarded Reward-guided Agent Correction Environment in Data Science</strong> - Aleksandr Tsymbalov, Danis Zaripov, Artem Epifanov, Anastasiya Palienko - [[pdf]](https://arxiv.org/pdf/2606.16000)</summary>

**Abstract:** We introduce GRACE-DS, a Guarded Reward-guided Agent Correction Environment in Data Science for pre-deployment evaluation of LLM-powered AutoML agents. GRACE-DS is a set of evaluation metrics in an isolated environment that can be applied to tabular ML tasks specific to a particular organization. It exposes agents to realistic workflow stages, from planning and data inspection through feature engineering, model development, validation, and code repair to final submission, while hidden executable validators measure not only final predictive performance but also leakage avoidance, reproducibility, protocol validity, correction behavior, and reward alignment. The strongest structured regime, flexible iterative interaction (our approach), achieves higher end-to-end normalized hidden-test quality than single-shot generation, unstructured interaction, and restart-based baselines, while also improving protocol-valid completion. Validated across more than 7,000 episodes, these results establish GRACE-DS as a robust platform for assessing the capacity of LLM-based AutoML agents to execute machine learning workflows under production-like conditions and in accordance with organization-specific requirements.

**arXiv ID:** 2606.16000
</details>

<details>
<summary><strong>ParaRecover: A Process-Level Benchmark for Error Localization and Recovery in Parallel Tool-Use Agents</strong> - Bowen Guan, Zhentao Yin, Yanming Shen - [[pdf]](https://arxiv.org/pdf/2609.12345)</summary>

**Abstract:** Existing agent benchmarks mainly evaluate final task success or tool-call correctness, providing limited insight into whether agents can reliably diagnose and recover from intermediate execution failures. This limitation becomes particularly critical in multi-turn parallel tool-use scenarios, where errors may propagate across dependent branches and trigger cascading failures. We introduce ParaRecover, a process-level benchmark for evaluating error localization and recovery in multi-turn parallel tool-use agents. Built upon a fine-grained taxonomy of 14 error types covering planning dependencies, tool selection, and argument matching, the benchmark comprises 10,626 instances spanning two difficulty levels. To enable finegrained, process-oriented evaluation, we further propose the SDE rubric, which measures structural integrity, diagnostic reasoning, and evolutionary strategy during agent this http URL across more than ten mainstream LLMs reveal that even state-of-the-art models still struggle with multi-turn error propagation,implicit tool-use failures, and precise replanning. Moreover, we demonstrate that the SDE rubric provides effective supervision signals for improving agents' reflective recovery capabilities. Our data and code are available at this https URL.

**arXiv ID:** 2609.12345
</details>

<details>
<summary><strong>Scenario-Independent Criticality Assessment and Prediction for Vulnerable Road Users in Autonomous Driving</strong> - Jörg Gamerdinger, Victor Schwarzenberger, Philipp Schmid, Sven Teufel, Oliver Bringmann - [[pdf]](https://arxiv.org/pdf/2609.11947)</summary>

**Abstract:** Increasing safety is the primary objective of automated vehicles. Achieving this goal requires reliable safety metrics that incorporate safety-relevant factors such as object type, velocity, and criticality. A key capability of such metrics is the distinction between critical and non-critical objects, which is addressed through criticality or relevance estimation. Existing criticality metrics are typically designed for specific scenarios and primarily focus on vehicle-to-vehicle interactions. In this paper, we therefore propose a novel criticality metric tailored to vulnerable road users (VRUs), which require special consideration due to their less predictable motion behavior. Furthermore, to avoid the complexity introduced by scenario-specific metrics, we introduce a scenario-independent criticality prediction framework applicable to all traffic participant classes. The effectiveness of both the proposed VRU-centric criticality metric and the criticality prediction framework is evaluated using the DeepAccident dataset, which contains a diverse set of safety-critical traffic scenarios. The proposed VRU-centric criticality metric improves pedestrian criticality classification performance by up to 50 %. In addition, the proposed criticality prediction framework outperforms state-of-the-art metrics by 275 %, achieving an F1-score of 0.96 and enabling scenario-independent criticality assessment across all object classes. These results demonstrate the strong potential of the proposed approaches to enhance criticality assessment for safety evaluation in automated driving systems.

**arXiv ID:** 2609.11947
</details>

<details>
<summary><strong>Multi-Objective Agent-Based Model Predictive Controller for Plug-and-Play Vehicle Control</strong> - Jiaming Zhong, Ladan Khoshnevisan, Shucheng Huang, Mohammad Pirani, Yash Vardhan Pant, Amir Khajepour - [[pdf]](https://arxiv.org/pdf/2609.12108)</summary>

**Abstract:** Functional integration is a growing trend in vehicle control, often involving the coordination of multiple controllers to achieve various objectives simultaneously. The need for flexibility and reliability has led to a "plug-and-play" approach in control system design, which presents challenges for traditional integrated model predictive control (MPC). Agent-based model predictive control (AMPC) has recently emerged as a distributed solution that treats controllers as agents, creating a collaborative framework among them to reach a common goal. However, this approach struggles to manage distributed conflicting objectives when agents are coupled or interdependent. To address this, we propose a novel, practical distributed control scheme called multi-objective AMPC, which adapts the alternating direction method of multipliers (ADMM) into a general control strategy that approximates global optimization while decoupling objectives. We systematically develop three formulations that maintain convergence while addressing control regularization and inequality constraints, applying them to complex vehicle control systems for the first time. The proposed method has been tested on two vehicle control scenarios with a multi-objective topology. Different formulations are compared through simulations, and the most computationally efficient one was implemented on an electric vehicle for real-world evaluations. The results demonstrate that the proposed multi-objective AMPC can converge approximately to the same global optimum as integrated MPC with greater flexibility and the potential to reduce computational costs.

**arXiv ID:** 2609.12108
</details>

<details>
<summary><strong>Guardrailed Meta-Agent Loops: Stress-Testing Policy Pinning, Budget Bounds, and Crash Recovery</strong> - Qinzhen Ma, Jialin Wu - [[pdf]](https://arxiv.org/pdf/2609.12216)</summary>

**Abstract:** Self-improving agent workflows create an audit problem when the same controller can change both its behavior and the conditions under which that behavior is judged. We present GuardrailLoop, a simulation-based testbed that makes three operational contracts jointly testable: preservation of human-defined policy, compute accounting at every recorded execution prefix, and recovery of a specified scientific state after crashes. A hash-pinned policy fixes goals, scope, evaluation identity, budget, and release conditions; machine-directed evolution is restricted to a code-owned feature catalog and bounded knobs. The contribution is an executable boundary and an evaluation protocol that separates useful adaptation, state recovery, and repeated execution. In a paired 50-seed 2 x 2 study, round-stage growth changes target attainment by +1.00 and restricted mean compute to target by -56.97 simulated GPU-hours (95% paired-bootstrap interval [-58.91,-54.70]); idle growth has zero measured utility effect. Across 240 enumerated crash injections, all runs recover the defined outcome, but only 210 preserve the normalized trace: 30 pre-commit crashes repeat a planner call. Resource-drift, kill-switch, integrity, and output-guard matrices satisfy their specified checks. These findings show why successful outcome recovery is insufficient evidence of exactly-once execution. They establish conformance within one calibrated deterministic testbed, rather than general safety or real-world self-improvement.

**arXiv ID:** 2609.12216
</details>

<details>
<summary><strong>READ: Learning Risk-Informed Fields for End-to-End Autonomous Driving</strong> - Zhiyuan Liu, Yuanxin Tian, Zehong Ke, Jinhao Li, Hao Cheng, Zhenhua Xu, Wenhao Yu, Jianqiang Wang - [[pdf]](https://arxiv.org/pdf/2609.12371)</summary>

**Abstract:** Autonomous driving requires more than recognizing what is present in a scene: a planner must determine how road structure, surrounding agents, and their motion states should influence a future maneuver. Existing learning-based planners can capture these influences through latent scene features and trajectory decoders, but the relationship between environmental factors and candidate actions often remains implicit. This limits the ability to inspect, diagnose, or refine how scene context affects the safety of a predicted trajectory. Classical safety fields provide an explicit spatial representation of this relationship, but their risk shapes and relative weights are prescribed in advance and do not adapt to each scene. We introduce READ, a framework that learns an explicit, planning-aligned risk representation from complementary geometric and behavioral constraints. READ instantiates this representation as a continuous spatiotemporal field, enabling differentiable queries along candidate trajectories. The learned field connects scene understanding with action selection by encouraging predicted trajectories to align with low-risk regions, while retaining a differentiable interface for trajectory evaluation and refinement. READ integrates with both end-to-end planners and Vision-Language-Action models. Experiments on NAVSIM show consistent gains across matched end-to-end backbones and strong performance in a VLA setting; READ also achieves competitive results on NAVSIM v2. These results establish learned spatial risk as an explicit, adaptable representation for safe planning.

**arXiv ID:** 2609.12371
</details>

<details>
<summary><strong>Quantifying Spectral Differences in Vehicle Between Production Autonomous and Human-Driven Vehicles Across Driving Scenarios</strong> - Peiyi Fang, Xiangyu Li, Yonglin Weng, Ke Ma - [[pdf]](https://arxiv.org/pdf/2609.12609)</summary>

**Abstract:** Differences in vehicle kinematic characteristics between production autonomous vehicles (PAVs) and human-driven vehicles (HVs) have been limitedly investigated by empirical studies. Most recent studies rely on simulation-based models, while some further investigate low-level adaptive cruise control (ACC) systems in controlled experiments. These methods commonly adapt some time-domain metrics to characterize PAV-HV differences across limited driving conditions. However, current PAVs equipped with high-level autonomous driving systems generate driving behaviors in a black box using data-driven models. These fundamentally different mechanisms for generating behaviors may produce distinct kinematic characteristics in traffic. More importantly, these time-domain metrics cannot reflect frequency-related traffic dynamics across different driving scenarios. Thus, this study adapted a real-world PAV dataset with four PAV platforms and developed a frequency-domain framework to quantify kinematic differences between PAVs and HVs across diverse driving scenarios, including varying driving states, lighting, weather, and vehicle densities. The framework transforms kinematic signals into the frequency domain and extracts spectral features, and then compares these features between PAVs and HVs based on kernel density estimation and Wasserstein distance. The results reveal clear scenario-dependent PAV-HV spectral differences. Specifically, speed-related differences were consistently smaller during car-following than cruising, while rainy conditions consistently enlarged acceleration-related differences compared with clear conditions. These findings highlight the necessity of multi-scenario evaluations and demonstrate the value of frequency-domain analysis for characterizing PAV-HV kinematic differences under real-world conditions.

**arXiv ID:** 2609.12609
</details>

<details>
<summary><strong>ForkSCOPE: Charting the Agentic Garden of Forking Paths</strong> - Arjun Balaji, Batuhan Duru Yeltekin, Tian Zheng - [[pdf]](https://arxiv.org/pdf/2609.12438)</summary>

**Abstract:** Even with a fixed dataset and research question, data analysis involves many defensible decisions. Understanding how these choices influence the results is scientifically important but remains challenging. Crowdsourcing and agentic AI can generate hundreds of end-to-end analyses, but scaling generation alone can create a processing bottleneck and an analytic ``black hole.'' A common workaround is to impose a shared fixed decision taxonomy, which can limit insight and understate uncertainty. We present ForkSCOPE, a human-AI collaboration framework that induces structure bottom-up from the code corpus of end-to-end analyses, without a taxonomy fixed before or after generation, so the organization and evaluation of the garden can scale with the corpus. ForkSCOPE surfaces the charted garden of forking paths through a human-AI collaboration pipeline and an evidence-linked interactive viewer for steering and verification: it spotlights organically identified forks and structures and produces a derived taxonomy and decision map compatible with existing multiverse tools.

**arXiv ID:** 2609.12438
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework</strong> - Ioannis Prokopiou, Athanasios Aidinis, Panagiotis-Christos Kyrmpatsos, Pantelis Vikatos - [[pdf]](https://arxiv.org/pdf/2609.12746)</summary>

**Abstract:** Agentic pipelines for structured-query generation are rapidly expanding, but it is unclear which part of the loop produces the gain. We use LAST-CQ -- a five-agent, training-free, execution-grounded Text-to-Cypher framework -- as an instrumented testbed, running three counterfactuals over 2,471 live-database queries and six backbones spanning three vendor scale tiers. Removing correction is worth between 3.1% aggregate execution-BLEU against the single-pass system and 12.3% against a no-refinement counterfactual (up to 80.7% for the weakest backbone). Replacing schema-grounded, LLM-synthesised feedback with raw database error strings costs almost nothing (20.9% vs. 19.9% naive exact match; <0.2% end-to-end; equivalent within $\pm 0.075$ set-F1 by two one-sided tests). Spending the same call budget on parallel sampling degrades quality by 10-11%. What works is detecting failure and routing it to a retry, not the feedback sophistication or number of samples. LAST-CQ itself recovers 91.7% of queries that fail under single-pass generation, while a query that succeeds first time still costs exactly one LLM call. We also show that n-gram overlap on serialised results is not a bound in either direction: it over-scores against set equivalence on 65.9% of results while under-scoring against judged semantics. Finally, we calibrate our LLM judge against blind human labels and find it optimistic by 9 points.

**arXiv ID:** 2609.12746
</details>

<details>
<summary><strong>Agentic TCAD Calibration Workflow for Oxide Semiconductor Transistors</strong> - Gyujun Jeong, Junmo Lee, Sungwon Cho, Woohyun Hwang, Kwangyou Seo, Suhwan Lim, Wanki Kim, Daewon Ha, Rishi Ranade, Kihang Youn, Ram Cherukuri, Yiyi Wang, Asif Khan, Shimeng Yu - [[pdf]](https://arxiv.org/pdf/2609.12184)</summary>

**Abstract:** Experimental TCAD calibration is essential for predictive technology modeling of emerging oxide semiconductor transistors. However, it remains time-consuming and expert dependent because of model ambiguity. Multiple physical models and parameter sets can reproduce the same measured transfer characteristics, while local fitting alone cannot uniquely identify the underlying device physics. We present the first demonstration of an agentic TCAD calibration workflow for a fabricated bottom-gate In--W--O (BG-IWO) transistor. Starting from the measured transfer curve and device information, the workflow uses measurement--TCAD residuals and local sensitivity tests to select bounded parameter corrections or evaluate additional physical models, and accept only updates that improve device metrics. The LLM agent orchestrates the workflow, while Sentaurus governs the device physics. For the 2\%-W reference device, five agent-suggested updates yield a fixed calibrated model, reducing the multi-metric device objective $J$ by 14.3$\times$. Maximum $V_{\mathrm{th}}$/$I_{\mathrm{on}}$ errors are 36.1~mV/0.022 decade for varying-drain-bias tests and 46.2~mV/0.062 decade for varying-channel-length tests, demonstrating model transferability across bias and geometry rather than a local parameter fit. W-composition tests provide process-sensitive insight. This agentic workflow provides a faster route to model development for emerging device technologies.

**arXiv ID:** 2609.12184
</details>

<details>
<summary><strong>Behavior Quotient Learning for Low-Rank Adaptation of LLM Agents</strong> - Pengyang Zhou, Xiaobin Tu, Zhengxi Liu, Rongkun Xue, Haochen Li, Miancan Liu, Ziyuan Chen, Yinggui Wang, Jinkui Ren, Xiantao Zhang - [[pdf]](https://arxiv.org/pdf/2609.12896)</summary>

**Abstract:** LLM-based agents rely on heterogeneous interaction capabilities to accomplish complex tasks. Existing approaches often distribute these capabilities across multiple LoRA adapters, which increases adapter storage requirements and introduces routing overhead during inference. A single LoRA avoids this overhead, but learning from diverse agent trajectories under a fixed rank budget presents two challenges. First, trajectories with different interaction traces and parameter gradients can induce equivalent changes in decision distributions, causing repeated updates to overemphasize redundant behavioral changes. Second, an aggregated update may exceed the rank budget of the adapter, and approximating it in weight space can distort the decision changes that it is intended to produce. We propose BQ-LoRA, a low-rank adaptation framework that organizes trajectory updates through a local behavior quotient manifold. It contains two modules, i.e., behavior quotient balancing (BQB) and decision preserving compression (DPC). BQB constructs the quotient manifold from decision distributions and reweights trajectory update directions according to their local density in the quotient tangent space. DPC projects the balanced gradient onto the intrinsic fixed rank tangent space and refactorizes the resulting target by jointly controlling effective weight error and distortion of decision distributions. Experiments on AppWorld and BrowseComp-Plus compare BQ-LoRA with standard LoRA and recent low-rank adaptation methods, while separate ablations evaluate the complementary contributions of both components.

**arXiv ID:** 2609.12896
</details>

<details>
<summary><strong>Graph-of-Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills</strong> - Dawei Liu, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, Lichao Sun - [[pdf]](https://arxiv.org/pdf/2604.05333)</summary>

**Abstract:** As LLM agents act across personal applications, web browsers, and other interfaces, their reusable skill libraries can scale to thousands of skills. This scale introduces two challenges. First, loading the full library saturates the context window, driving up token costs, hallucination, and latency. Second, semantic retrieval surfaces topically relevant skills but can miss upstream and downstream prerequisite skills, creating a prerequisite gap that leaves the retrieved bundle insufficient for execution. We present Graph-of-Skills (GoS), an inference-time structural retrieval layer for large skill libraries. GoS constructs an executable skill graph offline from skill packages, then retrieves a bounded, dependency-aware bundle through hybrid semantic-lexical seeding, reverse-aware Personalized PageRank, and context-budgeted hydration. Across SkillsBench and ALFWorld, with three model families (Claude Sonnet 4.5, MiniMax M2.7, and GPT-5.2 Codex), GoS attains the highest average reward in all six model-benchmark blocks, at a fraction of the token cost of loading the full library. On SkillsBench with GPT-5.2 Codex it raises average reward by 7.0 absolute points over full skill loading, a 25.6% relative gain, while cutting total tokens by 56.7%. Ablations isolate the mechanism: replacing reverse traversal with forward propagation costs 9.1 reward points, a larger loss than removing the graph altogether. The gain thus comes from traversing dependencies backwards, not from graph diffusion as such. A budget-matched retrieval study holding seeding, reranking, hydration, and context budget fixed reproduces the same ordering, with dependency-pair co-recovery falling from 0.654 to 0.362. Code is available at this https URL

**arXiv ID:** 2604.05333
</details>

<details>
<summary><strong>OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents</strong> - Kaicheng Zhang, Wen Ge, Lei Jiang, Weixin Yang, Jordan Langham-Lopez, Jialin Yu, Lukasz Szpruch, Hao Ni - [[pdf]](https://arxiv.org/pdf/2606.26350)</summary>

**Abstract:** Although large language model agents are increasingly applied to quantitative-finance workflows, their evaluation remains fragmented across isolated tasks, while the financial relevance of benchmark tasks is often overlooked. Yet financial workflows are inherently multi-stage, spanning interdependent tasks such as forecasting, strategy construction, risk management, and trading. Existing platforms typically focus on a single task, and can therefore overstate agent competence and fail to reveal weaknesses in generalization, real-market interaction, and financially meaningful decision-making. We introduce OpenFinGym, a unified gym environment for quantitative-finance agent development that covers forecasting, market generation, real-time trading, and fraud detection under a single execution and verification interface. OpenFinGym additionally provides an automated task-construction pipeline that turns quantitative finance publications into executable task packages; a containerised runtime with a host-side verifier service that supports scalable agent rollouts and prevents runtime train-test leakage; a paper trading engine with a low-latency data-stream design; deferred-resolution support for long-horizon and event-market forecasts; and integration for SFT and RL post-training

**arXiv ID:** 2606.26350
</details>

<details>
<summary><strong>SimSkill: A Self-Evolving LLM Agent for Skill and Knowledge Accumulation in Traffic Simulation</strong> - Qi Liu, Qinzheng Wang, Can Li, Yiming Bie, Wanjing Ma - [[pdf]](https://arxiv.org/pdf/2609.03753)</summary>

**Abstract:** Cumulative culture enables humans to preserve, reuse, and extend knowledge and skills across experiences and generations. Inspired by this principle, we introduce \textit{SimSkill}, a self-evolving agent built around the Simulation of Urban MObility (SUMO) traffic simulator. SimSkill continually identifies capability gaps, generates and solves environment-grounded tasks, verifies solutions through an action--critic loop, and consolidates experience into episodic, procedural, and semantic memory. Through autonomous exploration, it builds a library of reusable skills and knowledge spanning major stages of the traffic-simulation workflow. We evaluate SimSkill on two held-out benchmarks across three backbone LLMs, with each result independently verified. It improves verified success by up to 25 percentage points, and ablations show complementary contributions from procedural and semantic memory. Its benefits remain backbone- and budget-dependent, as memory does not improve every model or uniformly reduce inference cost. More broadly, SimSkill illustrates a natural-language-centered design paradigm for LLM-based agent systems. Its high-level control logic, operating principles, and accumulated knowledge are expressed in natural language, while an LLM integrates them with executable tools and code to realize precise and reproducible execution. All code and experimental data are publicly available at this https URL.

**arXiv ID:** 2609.03753
</details>

<details>
<summary><strong>Look Before You Leap: Pre-Action Verification for LLM Agents</strong> - Asaad Althoubi - [[pdf]](https://arxiv.org/pdf/2609.11957)</summary>

**Abstract:** An LLM agent acts on the world by emitting actions: shell commands to run, edits to apply. A wrong action does not always fail loudly; it can fail silently, producing a plausible but incorrect effect that raises no error. We argue that a cheap deterministic check, run before an action takes effect, is an effective and underused form of agent oversight, and we study it across two action modalities in one framework. The idea is to fix an action's correct effect by construction, before any executor runs, so that silent failure is measured directly and the verifier may abstain rather than guess. For shell commands, a static verifier over 9930 commands and 482 tools catches 95.8% of invalid commands at a 10.0% false-positive rate. Its syntax and binary checks are oracle-exact, giving zero false positives while catching half of all errors; the flag check is bounded only by help-text coverage and accounts for every false positive. For code edits, a benchmark of 640 edits over 224 files isolating the apply step exposes a sharp split. Content-anchored formats such as search/replace and diff fail cleanly, whereas location-anchored formats fail silently: line numbers corrupt 99.1% of files under a one-line shift, and function-name edits hit the wrong function 12.7% of the time. In both settings a refuse-when-unsure policy turns silent failures into recoverable ones at a tunable cost in applicability: selective grounding reaches 0.958 recall at 7.0% false positives, and an anchor-and-verify applier records one silent misapplication in 8320 trials (0.01%). We release both benchmarks, the verifiers, and the guards.

**arXiv ID:** 2609.11957
</details>

<details>
<summary><strong>GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents</strong> - Umesh Bodhwani, Thanh Tran, Kai Wei - [[pdf]](https://arxiv.org/pdf/2609.12191)</summary>

**Abstract:** Comparing and selecting task-oriented LLM agents increasingly relies on a low-cost offline evaluation gate: persona-driven LLM user-simulators converse with each candidate, an LLM-as-a-judge scores the transcripts, and the higher-scoring agent is promoted. We introduce GAUGE, a reusable offline protocol that measures whether this gate's ranking matches a grounded verifiable reward across 25 agents from six providers on the $\tau^2$-bench and SimulatorArena benchmarks, separating two kinds of evaluation validity that release practices conflate: ranking validity and construct validity. First, a satisfaction-success gap: satisfaction carries essentially no information about task success, as conversations rated satisfied by our blind panel are decorrelated from actual success, with 57.5% of them failing the customer's task, a pattern consistent across five rater populations, both benchmarks, and every subjective dimension we rated. Second, while the gate's ranking is robust across the broad capability span, it loses resolution among the near-equal strong agents: this decision-disagreement rate jumps from $<$1% on wide-reward pairs to 31% on close pairs. The gate is thus human-validated yet mis-anchored. As a remedy, we propose a calibrate-then-trust cadence in which a judge-free completion bit is a zero-cost tripwire for truncation regressions.

**arXiv ID:** 2609.12191
</details>

<details>
<summary><strong>LifeMem: Enabling Lifelong Experience Reuse for LLM Agents</strong> - Yuli Qiu, Yutong Li, Wei Su, Zeming Liu, Wanxiang Che, Heyan Huang, Haifeng Wang, Yuang Guo - [[pdf]](https://arxiv.org/pdf/2609.12655)</summary>

**Abstract:** Large language model agents are expected to continuously adapt to new tasks and environments over their lifetime by reusing past experience. However, existing memory-based agents struggle to transfer reusable experience across environments and suffer from catastrophic forgetting as experience accumulated. To address these challenges, we propose LifeMem, a lifelong learning framework that enables agents to transfer knowledge across multiple environments. During learning, LifeMem clusters accumulated interaction trajectories based on underlying workflows to extract reusable skills. When solving a new task at inference time, the agent recalls relevant skills and trajectories to guide actions. To validate our method, we conduct experiments across 10 environments and over 13k tasks with 2k newly annotated interaction trajectories. Results show that LifeMem enables effective experience reuse in lifelong learning, achieving both reduced forgetting on learned tasks and superior cross-task transfer. Further analysis reveals that task streaming impacts learning, while consolidating structurally similar trajectories within memory boosts performance.

**arXiv ID:** 2609.12655
</details>

<details>
<summary><strong>Granularity-Adaptive Credit Assignment for Long-Horizon LLM Agent Reinforcement Learning</strong> - Taoran Liang, Yang Liu, Shang Luo, Yingguang Yang, Rongrong Zhang, Yingzong Min, Yulin Huang, Jianshen Zhang, Yongzhi Qi, Kefu Xu, Congjing Ran, Bin Chong - [[pdf]](https://arxiv.org/pdf/2609.12424)</summary>

**Abstract:** Reinforcement learning is now the standard way to train large language model agents on long-horizon tasks, where dozens of interdependent actions precede a single sparse reward. Critic-free, group-relative methods such as GRPO suit this regime, but they broadcast one trajectory-level scalar to every step and cannot say which decision drove the outcome. GiGPO recovers a step-level signal by grouping time steps that share an anchor state, yet it merges the step- and episode-level estimates under one fixed weight, spending the same resolution on a pivotal branching decision as on a routine, near-deterministic transition. We argue that the right resolution is state-dependent, and propose GACA, a critic-free estimator whose granularity follows an uncertainty-based criticality proxy. GACA scores every step by the negative log-likelihood its own rollout already records, then blends the two advantages with a per-step weight that grows with that score, so the gradient places more weight on the fine-grained signal at above-average NLL and on the episode-level signal below it. We derive an exact risk decomposition for the implemented mixture and show that sufficiently small modulation improves on fixed mixing under positive directional alignment. A separate conditional result bounds local action-value variation using expected NLL, while an error-projection analysis characterizes when mixing adds value beyond scalar uncertainty reweighting. On ALFWorld and WebShop, GACA improves task success over GRPO and GiGPO at both 1.5B and 7B scales.

**arXiv ID:** 2609.12424
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>AIM: A Privacy-Aware Interoperable Memory Framework for Multi-Agent Multi-User LLM Systems</strong> - Zachary Johnson, Nigel Boachie Kumankumah, Somya Chatterjee, Tejas Sathyamurthi, Min Chen, Xinyi Alice Li, Xiao Wang, Emily Morgan Gelchie, Jessica Lin, Sadid A. Hasan, Sulaiman Vesal - [[pdf]](https://arxiv.org/pdf/2609.12320)</summary>

**Abstract:** Traditional large language models (LLMs) are scoped to individual user sessions, limiting their knowledge to a single conversation and preventing them from learning user preferences that evolve over time. Existing agentic memory systems address this limitation but generally operate at the individual-user level, restricting the public knowledge that could be shared across users to improve downstream responses. We introduce AIM (Agentic Interoperable Memory), a unified, privacy-aware memory framework that enables multi-agent, multi-user LLM systems to persistently manage private and shared memory. AIM dynamically classifies information as private, scoped to one user and inaccessible to others, or public, accessible to all users. It enforces index-level access controls so that private memories are retrievable only by their owner, protecting sensitive data while allowing beneficial shared knowledge to improve coordination and consistency. We also introduce MUMBench (Multi-User Memory Benchmark), a dataset of multi-user interactions containing private and shareable information across four domains. To our knowledge, MUMBench is the first public dataset designed to evaluate multiple memory operations, including retrieval, creation, update, and deletion, in a multi-user environment. Across three independent runs on MUMBench, AIM achieves 96.0% visibility classification accuracy, 58.8% strict operation accuracy, and 70.5% state-aware operation accuracy.

**arXiv ID:** 2609.12320
</details>

<details>
<summary><strong>Hierarchical Belief Modeling for Zero-Shot Opponent Adaptation in Partially Observable Multi-Agent Navigation</strong> - Kowei Shih, Lu Cheng, Zeyu Wang, Yeyun Xu, Kejian Tong - [[pdf]](https://arxiv.org/pdf/2609.12422)</summary>

**Abstract:** Lux AI Season 3 requires agents to act under partial observability, randomized episode level dynamics, and a best of five match structure that rewards both tactical execution and fast adaptation. We present HORIZON, a hierarchical agent that combines symmetry aware spatial perception, dual memory belief tracking, relic centric graph attention, information gain driven exploration, and an opponent conditioned policy mixture. HORIZON separates short horizon control from cross match meta reasoning, while auxiliary belief and world model objectives stabilize learning. Trained with PPO in a large scale JAX simulator, the resulting agent explicitly infers hidden game parameters and opponent style. Experiments show consistent gains in match win rate, episode win rate, adaptation gain, and league rating over strong recurrent and feed forward baselines.

**arXiv ID:** 2609.12422
</details>

<details>
<summary><strong>Information Specialization and Constrained Synthesis in Multi-Agent LLM Forecasting: A Prospective Live-Study of the 2026 FIFA World Cup</strong> - Julian Varghese, Lucas Bickmann, Sarah Sandmann - [[pdf]](https://arxiv.org/pdf/2609.12495)</summary>

**Abstract:** Large language models are being organized into multi-agent systems with specialized roles, but whether such specialization produces distinct forecasts and whether subsequent synthesis improves utility remains unclear. In this study, we carried out a live, prospective evaluation over the final 56 matches of the information-dense 2026 FIFA World Cup, keeping a frontier foundation model constant while assigning two primary forecasting agents contrasting specialist roles: a quantitative specialist focusing on structured performance statistics and a news specialist focusing on current injuries, tactics and information from press conferences. Their forecasts were then reviewed by a separate critic before being combined by a meta-agent, resulting in a sequential four-agent model. Forecasts from the betting market served as an external benchmark. The news specialist obtained the highest mean probability-weighted Top-3 utility and matched the betting market in Top-3 exact-score hits. Nevertheless, the two specialist forecasters agreed on at least two of the three scorelines in 50 out of 56 matches, and the meta-agent never generated more than one scoreline outside the specialists' forecast set. These findings show that rapidly changing, unstructured information can provide a valuable forecasting signal alongside structured statistics, whereas adding critic and meta-agent stages does not necessarily create complementary information or improve on the strongest specialist.

**arXiv ID:** 2609.12495
</details>

<details>
<summary><strong>MAIA: Multi-Agent Intent Articulation for Requirement Discovery in Art Commissions</strong> - Yu-Chao Wang, Yanhong Lu, Yingjie Victor Chen, Tim McGraw - [[pdf]](https://arxiv.org/pdf/2609.12097)</summary>

**Abstract:** In bespoke art commissions, laypeople know what they feel but lack the words to specify it: one participant wanted a laid-off truck driver depicted as "a ghost in his own machine" but left the medium, scale, and palette unsaid. We frame this as an articulation bottleneck at an under-served upstream stage: requirement discovery, which precedes any artist or image generator and forces the commissioner to constitute intent in the first place. We present MAIA (Multi-Agent Intent Articulation), a multi-agent system that scaffolds this stage through Socratic inquiry under a "Verification over Invention" rule, turning vague affect into a text-only brief of visual terms the user verifies. In a within-subjects study (N = 16), the full configuration produced a large, significant gain in Cognitive Support over a minimal baseline (r = 0.96, p_FDR = 0.015; LMM p_FDR < 0.001). Thematic analysis traces the same mechanism, and a validator gate structurally blocks unratified content. A complementary blind review by three professional concept artists on a sampled set of briefs corroborates this improvement from the artist's side: AI rewriting improved visual completeness and executability in all eight sampled tasks (task-level Wilcoxon p = 0.008; FDR q = 0.010), with directionally larger gains under MAIA than under the baseline (underpowered, d = 1.4-2.6).

**arXiv ID:** 2609.12097
</details>

<details>
<summary><strong>NDT Factory: Synthesizing Verified Network Digital Twins from Semantic Models via Multi-Agent LLM</strong> - Sudipta Acharya, Petar Djukic, Burak Kantarci - [[pdf]](https://arxiv.org/pdf/2609.12170)</summary>

**Abstract:** Autonomous network management requires systems that can evaluate Network Service Intents (NSIs) under varying conditions without manual implementation of analysis logic, as envisioned in TM Forum Level~4 (L4) autonomy. Behavioral Network Digital Twins (NDTs) enable such evaluation, but existing NDTs rely on pre-defined analytical logic, limiting adaptability for evolving closed-loop control. This paper introduces the NDT factory, a multi-agent software system that synthesizes executable behavioral NDTs on demand from semantic models using Large Language Model (LLM). We validate the system using a Call Admission Control (CAC) case study, where deterministic what-if analysis serves as the admission decision process. The NDT factory generates a complete CAC NDT through parallel synthesis and orchestration, achieving 100% compilation and test pass rates across multiple runs. Simulation over 300 NSIs shows 99.3% decision agreement with a reference implementation, 90% admission rate, and correct attribution of all rejections, demonstrating reliable synthesis with deterministic, verifiable execution.

**arXiv ID:** 2609.12170
</details>

<details>
<summary><strong>WorkflowPerturb: Calibrated Stress Tests for Evaluating Multi-Agent Workflow Metrics</strong> - Madhav Kanda, Sharad Agarwal, Rodrigo Fonseca, Alok Gautam Kumbhare, Pedro Las-Casas - [[pdf]](https://arxiv.org/pdf/2602.17990)</summary>

**Abstract:** Multi-agent LLM systems that generate structured workflows from natural-language requests are now deployed in production across cloud automation, DevOps, and enterprise orchestration. Operating them exposes a recurring change-management problem. Routine updates, such as re-running an input, swapping the LLM, or refactoring an agent's prompt or orchestration code, often produce workflows that differ substantially from validated references. Engineers then lack a principled way to decide whether a change is safe to ship. Automatic workflow evaluation is the natural tool, but in practice metric scores are poorly calibrated, and a numeric change rarely communicates the severity of the underlying degradation. We introduce WorkflowPerturb, a controlled benchmark that applies realistic, graded perturbations to golden workflows. It contains 4,973 golden workflows and 44,757 perturbed variants across three perturbation types (Missing Steps, Compressed Steps, Description Changes) at severities of 10%, 30%, and 50%. We benchmark multiple metric families, analyzing their sensitivity and calibration using expected score trajectories and instance-level alert rates. Our results characterize systematic differences across families and support severity-aware interpretation of workflow evaluation scores in change-management settings. WorkflowPerturb is publicly available at this https URL .

**arXiv ID:** 2602.17990
</details>

<details>
<summary><strong>CoSkill: Joint Reinforcement Learning of Reasoning and Meta-Skill Agents for Hierarchical Skill Evolution</strong> - Jinyuan Feng, Dongmin Li, Yiqun Chen, Yang Gao, Xing Chen, Huimu Wang, Zhiqiang Pu - [[pdf]](https://arxiv.org/pdf/2609.04865)</summary>

**Abstract:** Skill libraries improve the sample efficiency of agentic reinforcement learning (RL) by enabling large language model (LLM) agents to reuse procedural knowledge. Yet existing paradigms exhibit structural shortcomings: they either decouple skill evolution from policy optimization or instantiate meta-skills as fixed workflows. Both treat skills as passive objects to be managed, limiting the flexible evolution of skills and their co-adaptation with the reasoning agent. To address the limitations, we propose CoSkill, a unified multi-agent RL framework that recasts the static meta-skill workflow as a learnable Meta-Skill Agent and jointly trains it with a Reasoning Agent over a hierarchical skill library. By modeling the Reasoning and Meta-Skill Agents as a cooperative team sharing a single backbone, CoSkill enables end-to-end co-adaptation: the Reasoning Agent conditions its actions on a retrieved task skill and step skills selected from its child set, while its task performance guides the Meta-Skill Agent in refining those step skills. Experiments on ALFWorld and WebShop show that CoSkill substantially outperforms prior skill-based and RL baselines, achieving success rates of 98.4% and 90.6%, respectively (+3.5 and +6.2 pp). As shown in Figure 1, CoSkill achieves superior early-stage sample efficiency, asymptotic performance, and wall-clock efficiency. Our code is available at this https URL.

**arXiv ID:** 2609.04865
</details>

<details>
<summary><strong>QuantumQUBO Agent: Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language</strong> - Niloy Kumar Mondal, Md Rizwan Parvez - [[pdf]](https://arxiv.org/pdf/2609.10629)</summary>

**Abstract:** Quadratic Unconstrained Binary Optimization (QUBO) is a central formulation for combinatorial optimization and has gained increasing attention due to its compatibility with quantum, hybrid quantum-classical, and quantum-inspired solvers. However, translating natural-language problem descriptions into correct QUBO formulations remains difficult, requiring the identification of binary variables, constraints, objective functions, penalty terms, and suitable penalty weights. This process is time-consuming and often demands substantial domain expertise. To address this challenge, we propose an end-to-end multi-agent framework that automatically generates QUBO formulations from natural-language problem descriptions, supported by structured or unstructured test cases. To evaluate its performance, We also introduce QUBOBench, a benchmark containing 100 combinatorial optimization problems across 12 application domains, curated from peer-reviewed literature, competitions, and canonical NP-hard problems. Experimental results show that our framework achieves 68% accuracy on QUBOBench, outperforming a direct single-call baseline by 22%. Further analysis identifies iterative self-repair as the most important component contributing to improved performance. The data and code are open-sourced at this https URL.

**arXiv ID:** 2609.10629
</details>

<details>
<summary><strong>El Agente Quntur: A research collaborator agent for quantum chemistry</strong> - Juan B. Pérez-Sánchez, Yunheng Zou, Jorge A. Campos-Gonzalez-Angulo, Marcel Müller, Ignacio Gustin, Andrew Wang, Han Hao, Tsz Wai Ko, Changhyeok Choi, Eric S. Isbrandt, Mohammad Ghazi Vakili, Hanyong Xu, Chris Crebolder, Varinia Bernales, Alán Aspuru-Guzik - [[pdf]](https://arxiv.org/pdf/2602.04850)</summary>

**Abstract:** Quantum chemistry is a foundational enabling tool for the fields of chemistry, materials science, computational biology and others. Despite of its power, the practical application of quantum chemistry simulations remains in the hands of qualified experts due to methodological complexity, software heterogeneity, and the need for informed interpretation of results. To bridge the accessibility gap for these tools and expand their reach to chemists with broader backgrounds, we introduce El Agente Quntur, a hierarchical, multi-agent AI system designed to operate not merely as an automation tool but as a research collaborator for computational quantum chemistry. Quntur was designed following three main strategies: i) elimination of hard-coded procedural policies in favour of reasoning-driven decisions, ii) construction of general and composable actions that facilitate generalization and efficiency, and iii) implementation of guided deep research to integrate abstract quantum-chemical reasoning across subdisciplines and a detailed understanding of the software's internal logic and syntax. Although instantiated in ORCA, these design principles are applicable to research agents more generally and easily expandable to additional quantum chemistry packages and beyond. Quntur supports the full range of calculations available in ORCA 6.0 and reasons over software documentation and scientific literature to plan, execute, adapt, and analyze in silico chemistry experiments following best practices. We discuss the advances and current bottlenecks in agentic systems operating at the research level in computational chemistry, and outline a roadmap toward a fully autonomous end-to-end computational chemistry research agent.

**arXiv ID:** 2602.04850
</details>

<details>
<summary><strong>MedCollab: IBIS-Guided Multi-Agent Collaboration with Hierarchical Disease Relation Chains for Clinical Diagnosis</strong> - Yuqi Zhan, Xinyue Wu, Tianyu Lin, Yutong Bao, Xiaoyu Wang, Weihao Cheng, Huangwei Chen, Feiwei Qin, Zhu Zhu - [[pdf]](https://arxiv.org/pdf/2603.01131)</summary>

**Abstract:** Clinical diagnosis is a gradual process of evidence integration, in which physicians move from symptoms and medical history to examinations, competing hypotheses, disease relations, and treatment decisions. Large language models have advanced medical text understanding and generation. Yet their clinical use remains limited by weak evidence grounding, opaque reasoning, and inconsistent links among differential diagnosis, final diagnosis, diagnostic basis, and treatment planning. We introduce MedCollab, a multi-agent framework for full-cycle clinical diagnosis and report generation. MedCollab coordinates specialist and examination agents according to patient records. It structures agent deliberation with an Issue-Based Information System (IBIS) protocol, so that each diagnostic position is supported by patient-specific evidence and medical knowledge. It also builds Hierarchical Disease Relation Chains (HDRC) to connect accepted hypotheses through progression, complication, and comorbidity relations. During multi-round deliberation, a verifier-guided consensus module evaluates evidence support, medical plausibility, and logical conflicts. It then adjusts agent contributions and filters unsupported reasoning. Experiments on ClinicalBench and MIMIC-IV show that MedCollab outperforms leading LLMs and medical multi-agent baselines in diagnostic accuracy, evidence consistency, and clinical reasoning quality. These results indicate that structured and auditable collaboration can produce more faithful and clinically coherent diagnostic reports.

**arXiv ID:** 2603.01131
</details>

<details>
<summary><strong>PhysCodeBench: Benchmarking Physics-Aware Symbolic Simulation of 3D Scenes via Self-Corrective Multi-Agent Refinement</strong> - Tianyidan Xie, Peiyu Wang, Hu Jiaxin, Yuyi Qian, Yuxuan Wang, Shenyi Wang, Rui Ma, Yanlun Peng, Lanjun Wang, Ying Tai, Jian Yang, Zili Yi - [[pdf]](https://arxiv.org/pdf/2604.23580)</summary>

**Abstract:** Translating natural-language descriptions of physical phenomena into executable simulation code requires both programming expertise and physical reasoning. Current large language models (LLMs) lack this combination: they frequently produce code that runs but simulates the wrong physics. We introduce PhysCodeBench, the first benchmark for this task, with 1,200 expert-validated examples spanning four physical domains. Its evaluation suite, PhysCodeEval, goes beyond executability and visual fidelity to measure physical correctness directly from the engine state via conservation-law residuals and expert-written assertions, and supports cross-engine evaluation to disentangle physics reasoning from API fluency. As a reference method, we propose the Self-Corrective Multi-Agent Refinement Framework (SMRF), which decouples physics-aware error correction from code generation through specialized agents. This design is motivated by our finding that targeted correction, rather than generic iterative refinement, is the key driver of physical accuracy. SMRF nearly triples the physical-assertion pass rate of the best proprietary baseline (70.6\% vs.\ 23.8\%) and retains its advantage under cross-engine transfer.

**arXiv ID:** 2604.23580
</details>

<details>
<summary><strong>Orchestra: Corroboration-Based Regulatory Candidate Discovery via Composed Bioinformatics MCP Agents</strong> - Jose A. Bird - [[pdf]](https://arxiv.org/pdf/2609.05496)</summary>

**Abstract:** Orchestra composes two independently built bioinformatics MCP servers -- RegNetAgents, which infers gene regulatory
network topology from ARACNe networks, and CASCADE, which supplies four independent evidence sources (LINCS knockdown,
DepMap essentiality, super-enhancer status, DoRothEA transcription-factor confidence) -- into one multi-agent
workflow exposed via the Model Context Protocol. Its central architectural claim is that requiring RegNetAgents'
topology evidence and CASCADE's experimental evidence to agree on a candidate regulator yields a more trustworthy
candidate than either alone -- not previously tested directly, since RegNetAgents' own validation asked only whether
its candidate lists beat chance.
We test this on the TCGA tumor-acquired regulator tier (regulators in a gene's tumor ARACNe network but absent from
the GREmLN population-averaged baseline), selecting candidates by ARACNe mutual-information (MI) edge weight. On
RegNetAgents' published BRCA/COAD focal-gene panel plus matched negative controls, agreement among at least 2 of the 4
CASCADE sources predicts OncoKB cancer-gene status among focal genes (odds ratio 2.89, Benjamini-Hochberg-adjusted
p=0.0166) but not among negative controls (p=0.0721); a single source is not diagnostic for either group. The pattern
replicates and strengthens in a third cancer type, STAD, on a separately constructed panel (odds ratio 5.82), and
against an independently curated ground truth (the Sanger COSMIC Cancer Gene Census). MI edge weight is the strongest
single predictor overall (p=0.0003); a logistic-regression likelihood-ratio test confirms corroboration adds value
beyond it in both panels (p=0.0234; p=0.0001). Every experiment invokes Orchestra's real agentic entry point.

**arXiv ID:** 2609.05496
</details>

<details>
<summary><strong>ETHOS: Towards a Modular Ethics Framework for Clinical Multi-Agent Systems</strong> - Rakesh Sharma, Sydney Pugh, Cameron Beeche, Pankhuri Singhal, Rachel Wu, Margaret Eby, Jeffrey Duda, James Gee, Kyra O'Brien, Hersh Sagreiya, Marina Serper, Victoria Gershuni, Angela Bradbury, Anurag Verma, Eric Eaton, Kevin B. Johnson, Walter Witschey - [[pdf]](https://arxiv.org/pdf/2608.15424)</summary>

**Abstract:** The rapid adoption of large language models has enabled the development of clinical multi-agent systems (MAS) capable of integrating multimodal patient data and supporting increasingly complex clinical decision-making. However, the deployment of these systems in real-world healthcare settings raises critical ethical concerns related to safety, fairness, accountability, transparency, and patient trust. While numerous organizations, including the World Health Organization, the National Academy of Medicine, and the FUTURE-AI consortium, have proposed ethical frameworks and governance principles for healthcare AI, these efforts remain largely conceptual. To address this challenge, we present ETHOS (Ethics and Trust through Hierarchical Oversight System), a modular ethics framework designed as a governance meta-agent that can be integrated with any existing multi-agent system without requiring changes to its underlying architecture. ETHOS translates stakeholder-informed ethical requirements into executable runtime oversight through a layered governance approach consisting of deterministic checks, contextual reviews, and a final ethics critic. These components continuously evaluate intermediate reasoning steps and final outputs, enabling the system to identify ethical risks, request revisions, or suppress responses that fail predefined safety and trustworthiness criteria. We demonstrate ETHOS within a hepatology clinical decision-support MAS. Results show that ETHOS improves decision reliability by detecting incomplete, inconsistent, or out-of-scope evidence and appropriately increasing abstention when safe recommendations cannot be supported. By embedding ethical governance directly into system operation, ETHOS provides a practical and auditable mechanism for transforming high-level AI ethics principles into deployable safeguards.

**arXiv ID:** 2608.15424
</details>

<details>
<summary><strong>NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis</strong> - Wuche Liu, Yiran Qiao, Linlin Hou, Rui Yang, Shusen Pu, Song Wang, Jing Ma - [[pdf]](https://arxiv.org/pdf/2609.01971)</summary>

**Abstract:** AI is rapidly advancing neuroscience, yet many laboratories fail to fully unleash its potential due to significant interdisciplinary barriers. While pre-trained neural models for physiological data are progressing quickly, their heterogeneous architectures and modality-specific constraints hinder systematic integration, selection, and evaluation. Despite recent advances in large language model (LLM)-based agent systems for intelligent scientific applications, existing approaches often still lack the domain expertise required to effectively select and coordinate diverse neuroscience pre-trained models and handle unique data types in this domain. We present NS-Copilot, an LLM-driven multi-agent system for neuroscience analysis that autonomously supports end-to-end workflows for diverse professional tasks. It unifies domain-specific pre-trained models and supports key neuroscience modalities, including EEG and extracellular spike data, through a natural-language interface. Given raw data and a task description, NS-Copilot orchestrates agents with specialized roles for planning, adaptive control, code generation, and result synthesis, enabling analysis without dataset-specific heuristics. We evaluate NS-Copilot on neuroscience benchmarks spanning Alzheimer's disease, Parkinson's disease, and working memory spike decoding. Across 8 trials per task, the system consistently outperforms strong baselines on the primary metric, demonstrating the ability of NS-Copilot for effective and scalable neuroscience analysis. Our code is publicly available at this https URL.

**arXiv ID:** 2609.01971
</details>

</details>

<details open>
<summary><h2>Other Agent Research (16 papers)</h2></summary>

<details>
<summary><strong>Niching Agents in The Core</strong> - Gary B. Parker, Jim O'Connor, John Asaro - [[pdf]](https://arxiv.org/pdf/2609.12398)</summary>

**Abstract:** The Core is a unique competitive co-evolution algorithm that allows agents to evolve autonomous control without utilizing a traditional fitness function. The agents evolve via local interactions through tournament selection, crossover, and mutation, producing offspring by evolving better controllers. Previous works have shown The Core's ability to evolve agents capable of combat and navigation in the Xpilot video game. This research expands upon that premise by niching agents to specific subsets of the original environment The Core was tested in. Our results demonstrate the niched agents capacity for success over agents niched to the entire system and agents niched to different sub-environments.

**arXiv ID:** 2609.12398
</details>

<details>
<summary><strong>When Does AI Augment Work? A Workflow-Level Framework for Human-Agent Collaboration</strong> - CIVIC-AI Collaboration, Jiaying Wu, Caleb Ziems, Raymond Chan, Nancy F. Chen, Corlyss Chua, Gerard Chung, Jungpil Hahn, Wee Sun Lee, Zhengyuan Liu, Jamie Ng, Desmond C. Ong, Jeryl Ong, Da Ren Soon, Tianqi Song, Zhi-Xuan Tan, Sixing Tao, Emily Yang, Yajing Yang, Stella Xin Yin, Min-Yen Kan, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2609.12482)</summary>

**Abstract:** We aim to characterise the value of artificial intelligence in the workplace. Current studies largely measure this value in terms of the current automation capabilities and public adoption of AI. However, such metrics ignore the greater impacts of human--agent collaboration in transforming the nature of work. To account for this, we must expand the scope of our analysis beyond atomised tasks of today, and instead focus on how AI can augment entire workflows of the future. To ground this analysis, we establish a precise definition of AI augmentation comprising six conditions, spanning durable net value, meaningful human control, accountability and recovery, and long-term human development through learning, career pathways, and job purpose. We elaborate on these conditions and apply the framework in a case study of AI-mediated social surveys. We conclude by outlining how organisations, researchers, and government leaders can use this framework to make sense of the future of work.

**arXiv ID:** 2609.12482
</details>

<details>
<summary><strong>Unified Agentic Video Editing Across Levels of Complexity and Creativity</strong> - Surabhi S. Nath, Kim Ferres, Milan Petrović, Lion Schulz - [[pdf]](https://arxiv.org/pdf/2609.12769)</summary>

**Abstract:** Editing is a core component of video production, requiring creative planning and decisions under multiple constraints. Here, we report methods for agentic tooling for automated video editing across three tasks varying in editorial goal, complexity and creativity, namely scene previews, video summaries and cinematic trailers. We evaluate the outputs and discuss implications for automation and agency.

**arXiv ID:** 2609.12769
</details>

<details>
<summary><strong>Adaptive Agent Design</strong> - Raj Kiriti Velicheti, Subhonmesh Bose, Tamer Başar - [[pdf]](https://arxiv.org/pdf/2609.12486)</summary>

**Abstract:** We consider an agent acting against a general non-Markovian environment. The agent maintains its agent states, but is free to choose a transition kernel across those states and optimize its state-feedback control policies. We study the bi-level agent design problem that optimizes the transition kernel and the policy it induces, given said kernel with offline data of observations and actions obtained via a behavioral policy. For general environments, we show that a soft $Q$-learning algorithm converges almost surely to the fixed point of a soft Bellman equation defined by the stationary averages that the behavioral policy and the chosen kernel induce, and we delineate what separates the resulting policy from an optimal one. In partially observed Markov decision problems, we analyze convergence properties of parametrized transition kernel design via zero-th order and Bayesian optimization techniques.

**arXiv ID:** 2609.12486
</details>

<details>
<summary><strong>Tinker Tales: A Tangible Dialogue System for Child-AI Co-Creative Storytelling</strong> - Nayoung Choi, Jiseung Hong, Peace Cyebukayire, Ikseon Choi, Jinho D. Choi - [[pdf]](https://arxiv.org/pdf/2602.04109)</summary>

**Abstract:** Conversational AI agents are increasingly explored as creative partners, yet how conversation design shapes child-AI dialogue in co-creative settings remains underexplored. We present Tinker Tales, a tangible dialogue system for child-AI collaborative storytelling, in which educational frameworks (narrative development and social-emotional learning) are instantiated as conversation design, shaping how the agent engages children across four narrative stages. The system combines a physical storytelling board, NFC-embedded toys, and a mobile app mediating multimodal interaction through tangible manipulation and voice-based dialogue. We conducted a home-based user study with 10 children (ages 6-8) across two conversation design conditions varying in how the agent structured elaboration, with and without educational scaffolding. Our findings show that prompt framing shapes the form and consistency of children's narrative contributions, structuring how they participate in co-creative dialogue with AI.

**arXiv ID:** 2602.04109
</details>

<details>
<summary><strong>Agent as Policy for Robotic Manipulation</strong> - Mengzhao Jia, Yang Lin, Xixin Zhang, Zhihan Zhang, Xiaobai Liu, Meng Jiang - [[pdf]](https://arxiv.org/pdf/2609.12541)</summary>

**Abstract:** We demonstrate that a general-purpose agent can directly drive a physical robot throughout task execution without any task-specific or environment-specific training. We introduce Agent as Policy (AGP), which places task planning and execution under the agent's control. Given a task and a robot interface, the agent interprets visual evidence, writes executable programs, issues motion commands, and revises its actions in response to physical outcomes. This brings the agent's reasoning and programming capabilities into continuous interaction with the physical world. We study AGP across multiple real-world manipulation tasks spanning precision manipulation, dynamic motions, and deformable objects. These include assembly from human videos, block construction from goal images, die reorientation, targeted throwing, and bimanual towel folding. AGP achieves success rates of 100%, 100%, and 80% on three block construction configurations. These findings establish a path for general-purpose agents to act as robotic policies, extending their autonomy to physical manipulation through runtime reasoning, programming, and interaction.

**arXiv ID:** 2609.12541
</details>

<details>
<summary><strong>A Data-Driven Distributed Control Scheme: Learning Multi-Objective Agent-Based MPC for Path-Tracking</strong> - Jiaming Zhong, Reza Valiollahi Mehrizi, Yash Vardhan Pant, Amir Khajepour - [[pdf]](https://arxiv.org/pdf/2609.12142)</summary>

**Abstract:** Agent-based model predictive control (AMPC) has recently been proposed for vehicle systems with various controllers, such as differential braking and torque vectoring, where controllers are regarded as distributed agents contributing to the same objective. However, this scheme is challenging in handling multiple conflicting objectives with coupled agents. A common approach for such tasks is the integrated MPC, where all objectives and agents are stacked together in one optimization. Nevertheless, as more agents and objectives are involved, the integrated MPC will face challenges like computational burdens and maintenance difficulties in practice. To this end, this paper proposes a learning multi-objective AMPC that can improve design flexibility and computing efficiency. First, under the assumption of information exchange, a multi-objective AMPC tailored from the alternating direction method of multipliers (ADMM) is proposed to decouple the system and achieve the same performance as the integrated scheme iteratively. Second, a learning-based method for initializing iterations is proposed to accelerate convergence. In addition, a data management method is proposed for real-time efficiency, and an authentication module is designed for learning reliability. We compare the proposed scheme against the integrated scheme via a combined path-tracking simulation for autonomous vehicles with various controllers. The proposed scheme achieves the same control performance as the integrated one while reducing the computational time by 43.5%. Furthermore, the learning-based method saves 88.6% more computational time than without learning, making it suitable for real-time implementation.

**arXiv ID:** 2609.12142
</details>

<details>
<summary><strong>Mission Performance: Automatic and Adaptive Race Pace Progression for Autonomous Racing</strong> - Giovanni Lambertini, Matteo Pini, Nicola Musiu, Ayoub Raji, Francesco Iacovacci, Marko Bertogna - [[pdf]](https://arxiv.org/pdf/2609.12292)</summary>

**Abstract:** In this paper, we describe the Mission Performance module implemented for a fully autonomous racing car to automatically manage the longitudinal, lateral, and combined performances, aiming to speedup the laptime progression while assuring safety. Motivated by the difficulty and risks of applying the real-time estimation of the grip to critical modules like the motion planner and controller, the Mission Performance guides these modules adapting their target performance instead of changing the vehicle model parameters. The module is formed by pre-defined progressions to warm up the tires at the beginning of a run. Then, the system continuously monitors safety and vehicle dynamics metrics on a per-sector basis to adaptively reduce, maintain, or increase the performance levels for each sector, progressively converging toward the maximum allowed value. The solution's effectiveness is demonstrated on the EAV-25, a fully autonomous Dallara Superformula, at the Yas Marina Circuit during the Abu Dhabi Autonomous Racing League (A2RL) Season 2.

**arXiv ID:** 2609.12292
</details>

<details>
<summary><strong>PATH: Continuous Target Sensing among Autonomous Cooperative Drones</strong> - Heegyeong Kim, Alice James, Avishkar Seth, Endrowednes Kuantama, Jane Williamson, Yimeng Feng, Richard Han - [[pdf]](https://arxiv.org/pdf/2609.12456)</summary>

**Abstract:** Continuous target sensing by uncrewed aerial vehicles (UAVs) is constrained by limited flight endurance, motivating the transfer of tracking responsibility between cooperating UAVs. Such a handoff requires the receiver to identify the same physical target currently tracked by the sender despite differences in viewpoint, scale, and target appearance. Existing approaches based on global target localization or appearance-based cross-view association are limited by positioning uncertainty or ambiguous visual features. This paper presents Perspective Alignment \& Tracking Handoff (\textbf{PATH}), a platform-agnostic, geometry-assisted sensing and verification framework for target handoff between two moving UAVs. The sender reconstructs the tracked target as a metric 3D point using RGB-D sensing, while the receiver estimates its relative pose from a fiducial observation and projects the transmitted target point into its own image as a spatial prior for target acquisition. The receiver-generated candidate is then returned to the sender and verified through a cross-view Mutual Agreement Handshake before tracking responsibility is transferred. Real-world UAV experiments show mean relative-position and target-position errors of 0.047~m and 0.030~m, respectively. Under visually ambiguous conditions, PATH achieves 96.0\% frame-level receiver-side target acquisition accuracy, with 2.0\% false-positive and 2.0\% false-negative rates. A sensor-error sensitivity analysis shows that relative-pose uncertainty is the dominant contributor to receiver-view projection error. The implementation operates at video rate with compact inter-UAV communication below 16~kB/s at 60~Hz, demonstrating the feasibility of lightweight geometry-assisted target handoff on resource-constrained UAV platforms.

**arXiv ID:** 2609.12456
</details>

<details>
<summary><strong>Autonomous Precision Milling of Biological Structures via Generic Anatomical Priors and Active Boundary Perception</strong> - Enduo Zhao, Xiaofeng Lin, Yifan Wang, Yuhan Song, Weihan Li, Saul Alexis Heredia Perez, Kanako Harada - [[pdf]](https://arxiv.org/pdf/2609.12530)</summary>

**Abstract:** Autonomous precision milling of biological structures is challenged by incomplete knowledge of target geometry, local material thickness, and critical internal boundaries. Subject-specific preoperative models can address geometric and thickness variations, but static models cannot determine boundary status encountered during execution, while repeated target-specific imaging limits scalability. This article presents an uncertainty-aware autonomous milling framework that assigns complementary roles to generic anatomical priors and active boundary perception. A generic anatomical prior provides conservative global guidance and is transformed through semantic-guided registration and hybrid vision-force calibration into robot-executable guidance for individual targets. As milling approaches uncertain boundaries, the robot actively probes the remaining structure and uses relative stiffness changes to estimate boundary status and structural detachability. A state-adaptive controller governs transitions between active perception and spatially selective incremental refinement, repeating this cycle until the termination criterion is satisfied. Hierarchical experiments on biological surrogates and in vivo mouse cranial window creation demonstrate accurate anatomical prior transfer, reliable boundary adaptation, and autonomous precision milling of biological structures.

**arXiv ID:** 2609.12530
</details>

<details>
<summary><strong>Driving Context-guided Model Predictive Planning and Control for Autonomous Car Racing at the Limit and Beyond</strong> - Ayoub Raji, Federico Sacco, Nicola Musiu, Marko Bertogna - [[pdf]](https://arxiv.org/pdf/2609.12660)</summary>

**Abstract:** This paper presents a Model Predictive Control-based motion planning and control pipeline for autonomous car racing capable of adapting to different driving contexts, such as overtaking, nominal driving, and countersteering. A Cost Blending state machine manages the identification of different driving contexts and the selection of their predefined weights to be applied to the Model Predictive Planning (MPP) and Control (MPC) modules. The two optimization-based solutions share the same problem formulation and model, differing only in horizon length, rate, tuning, and in their open-loop versus closed-loop approach to maximize the effectiveness of their interaction. The work is validated on the fully autonomous open-wheel racecar Superformula EAV-25, with a lap time achieved that is within 2% of the best human driver reference. The results demonstrate the capability of the solution in driving at the limit of handling, smoothly executing overtaking maneuvers, and quickly reacting to high oversteering conditions to recover the vehicle stability.

**arXiv ID:** 2609.12660
</details>

<details>
<summary><strong>Layered Safety: Enhancing Autonomous Collision Avoidance via Multistage CBF Safety Filters</strong> - Erina Yamaguchi, Ryan M. Bena, Gilbert Bahati, Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2603.00338)</summary>

**Abstract:** This paper presents a general end-to-end framework for constructing robust and reliable layered safety filters that can be leveraged to perform dynamic collision avoidance over a broad range of applications using only local perception data. Given a robot-centric point cloud, we begin by constructing an occupancy map which is used to synthesize a Poisson safety function (PSF). The resultant PSF is employed as a control barrier function (CBF) within two distinct safety filtering stages. In the first stage, we propose a predictive safety filter to compute optimal safe trajectories based on nominal potentially-unsafe commands. The resultant short-term plans are constrained to satisfy the CBF condition along a finite prediction horizon. In the second stage, instantaneous velocity commands are further refined by a real-time CBF-based safety filter and tracked by the full-order low-level robot controller. Assuming accurate tracking of velocity commands, we obtain formal guarantees of safety for the full-order system. We validate the optimality and robustness of our multistage architecture, in comparison to traditional single-stage safety filters, via a detailed Pareto analysis. We further demonstrate the effectiveness and generality of our collision avoidance methodology on multiple legged robot platforms across a variety of real-world dynamic scenarios.

**arXiv ID:** 2603.00338
</details>

<details>
<summary><strong>AD-MPCC: Adaptive Differentiable Model Predictive Contouring Control for Autonomous Racing</strong> - Nam T. Nguyen, Binh Nguyen, Ahmad Amine, Thanh Vo-Duy, Rahul Mangharam, Truong X. Nghiem - [[pdf]](https://arxiv.org/pdf/2607.00141)</summary>

**Abstract:** This paper presents Adaptive Differentiable Model Predictive Contouring Control (AD-MPCC), a framework for autonomous racing that integrates differentiable MPCC with online parameter estimation to handle varying road-surface conditions. For online parameter estimation, we leverage a parameterized Pacejka Magic Formula together with a regularized moving-horizon estimation scheme with exponentially decaying weights to capture road interactions and update parameters in real time. Furthermore, we propose a differentiable MPCC (Diff-MPCC) framework that enables optimal adjustment of objective weights based on predefined long-horizon performance costs. To implement Diff-MPCC for online objective weight adaptation, we propose a Pacejka-informed machine learning model that is trained in a supervised manner using data generated by Diff-MPCC to tune the objective weights. Simulation results demonstrate that AD-MPCC reliably ensures safety and achieves faster lap times compared to baseline controllers in both single-surface and multiple-surface scenarios.

**arXiv ID:** 2607.00141
</details>

<details>
<summary><strong>Obstacle-Aware Autonomous Coverage and Navigation for Outdoor Robots</strong> - Leonardo Gargani, Matteo Frosi, Matteo Matteucci - [[pdf]](https://arxiv.org/pdf/2609.01384)</summary>

**Abstract:** Long-duration outdoor coverage with autonomous platforms remains challenging beyond classical planning: deployments face localization drift in open spaces, obstacles in cluttered sites, controller feasibility in turn-heavy maneuvers, and persistent autonomy with energy management. We propose a unified ROS 2 architecture for single-robot outdoor coverage: a dual-antenna RTK-GNSS fused in an EKF keeps both position and heading accurate across long missions; three controller-aware refinements extend a mature coverage planner; a Nav2-based behavior-tree mission executive coordinates multi-goal execution, layered recovery, cost-aware goal management, and autonomous docking for return-to-charge. In real-world trials across five outdoor areas with varying geometries and obstacle densities, the robot completed every coverage route, sweeping 93.1% to 96.1% of the planned coverage area.

**arXiv ID:** 2609.01384
</details>

<details>
<summary><strong>Trajectory Tracking Control Design for Autonomous Helicopters with Guaranteed Error Bounds</strong> - Philipp Schitz, Johann C. Dauer, Paolo Mercorelli - [[pdf]](https://arxiv.org/pdf/2603.08045)</summary>

**Abstract:** This paper presents a systematic framework for computing formally guaranteed trajectory tracking error bounds for autonomous helicopters based on Robust Positive Invariant (RPI) sets. The approach establishes a closed-loop translational error dynamics which is cast into polytopic linear parameter-varying form with bounded additive and state-dependent disturbances. Ellipsoidal RPI sets are computed, yielding explicit position error bounds suitable as certified buffer zones in upper-level trajectory planning. Three controller architectures are compared with respect to the conservatism of their error bounds and tracking performance. Simulation results on a nonlinear helicopter model demonstrate that all architectures satisfy the derived bounds, while highlighting trade-offs between performance and the conservatism of the computed invariant set.

**arXiv ID:** 2603.08045
</details>

<details>
<summary><strong>When2Talk: When Should a Proactive In-Car Agent Talk?</strong> - Kaiser Hamid, Peihang Li, Nade Liang - [[pdf]](https://arxiv.org/pdf/2609.12503)</summary>

**Abstract:** Proactive in-cabin agents can help passengers understand automated-vehicle (AV) behavior, but communicating every ride event may introduce unnecessary interruptions. We investigated how communication should adapt to event priority and passenger activity. In a mixed-methods within-subject study, 41 participants rode as passenger in a VR simulated fully-automated vehicle. We compared an event-triggered (ET) policy that communicated immediately at every event with a context-sensitive (CS) policy that selected \textit{Immediate}, \textit{Delayed}, or \textit{Silent} communications. CS increased communication appropriateness and substantially reduced perceived interruption. Perceived trust did not differ between policies, although baselines dispositional trust differentiated communication preferences. Findings highlight event consequence, passenger activity, continuing information value, and confirmation need as key considerations for selective in-cabin communication.

**arXiv ID:** 2609.12503
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (35 papers)</h2></summary>

<details>
<summary><strong>Harness or Model? Isolating the Harness Effect in Agentic Coding with a Contamination-Controlled Private Suite</strong> - Mohsen Arjmandi - [[pdf]](https://arxiv.org/pdf/2609.11987)</summary>

**Abstract:** An agentic coding system couples a language model to a harness: the tools, prompts and control flow that turn a chat model into an autonomous software engineer. Vendors ship harnesses tuned to their own models, and practitioners assume the vendor-native pairing solves more tasks. We measure that assumption with paired same-model contrasts on a private, contamination-controlled suite of 256 repository and post-cutoff contest tasks. The same 80 tasks ran under claude-agent-sdk and under deepagents on claude-opus-4-8, and under the openai-codex SDK and deepagents on gpt-5.5, with gemini-3.5-flash and deepseek-v3.2 as side cells. 792 of 800 planned runs were graded by an isolated oracle. Neither contrast resolves an average advantage for either harness: -1.25 pp for Opus 4.8 (48.8% vs 50.0%, task-bootstrap 95% CI [-10.0, +7.5]) and +1.25 pp for GPT-5.5 (55.6% vs 54.4%, CI [-4.4, +6.9]). The Opus average combines opposite strata: the native harness trails by 9.0 pp on the 61 repository tasks and leads by 23.7 pp on the 19 contest tasks (label-permutation p = 0.003). The partition was chosen after seeing the data and needs a designed replication. Correctness and completion also separate: 22 of 81 runs cancelled at the wall-clock ceiling had produced a passing patch. Re-priced from raw per-turn usage at frozen list prices, the neutral harness cost 1.3 to 1.6 times as much per solved task on Opus 4.8 and 1.2 times on GPT-5.5. These are observed-usage estimates. On the Anthropic account 58 runs left no usage record, and allocating that spend to either cell would move the Opus ratio between 0.7 and 2.3, so the billed ordering is unresolved. This revision corrects an August 2026 manuscript whose cost figures rested on a usage-semantics defect in our own telemetry (Section 5.1). We release the orchestrator, grading oracle, reanalysis code and derived aggregates. The tasks stay private.

**arXiv ID:** 2609.11987
</details>

<details>
<summary><strong>BlueLM-GUI Technical Report: A Real-Device-Centric Flywheel for Self-Improving Mobile GUI Agents</strong> - Tong Ye, Kunyang Han, Guozhi Wang, Longqiang Luo, Zhifeng Ding, Yongxiang Zhang, Xiaolei Shen, Yuxuan Zhang, Zhuping Zhang, Tao Xu, Yue Pan, Yucheng Zhao, Yupei Hu, Yuanjiang Ouyang, Danfeng Shen, Runqi Lin, Hongda Cai, Zhaoxiong Wang, Mengjia Yan, Yingjie Zhong, Chen Zhou, Zeyu Zhang, Xuwen Zhu, Penggang Shi, Mingcheng Luo, Ziyang Wu, Min Jin, Mingfu Shen, Zairong Xu, Fan Zhang, Hao Wang, Liang Liu, Zhulin Xie, Lijun Yao, Xiao Liang, Liangmin Wen, Liqiang Feng, Feilong Wu, Min Hu, Min Chen, Guanjing Xiong, Xiaohu Ruan, Xiaoxin Chen - [[pdf]](https://arxiv.org/pdf/2609.12394)</summary>

**Abstract:** Mobile GUI agents are shifting from multi-module frameworks to native models trained end-to-end, yet industrial deployment faces three persistent gaps. Sandbox training produces a distribution mismatch with production environments; expensive real-device failures remain underutilized; and fixed benchmarks saturate, losing the power to guide iteration. We present BlueLM-GUI, a 35B-A3B mobile GUI agent built as a real-device-centric flywheel that closes these gaps through three principles. Every Sample Matters: a dual-track pipeline with Heterogeneous Triple-System Consensus evaluation and an Error Correction \& Derivation Module salvages every trajectory into usable supervision. Every Rollout Is Real: a three-stage recipe---continual pre-training, supervised fine-tuning, and agentic reinforcement learning on hundreds of real phones---grounds every rollout in real production environments, so the capability the model learns transfers directly to deployment. Every Query Evolves: a quota-driven benchmark methodology with three orthogonal axes enables precise attribution and allows the benchmark to be systematically upgraded as the model improves. BlueLM-GUI achieves 87.4 on MobileGUI-VBench, surpassing the best closed-source model by 5.1 points, and 84.9 on AndroidWorld, the best result among open-source models and competitive with closed-source models. These results demonstrate that grounding model training and iterative improvement in both real devices and the three Every principles yields strong, robust, and transferable mobile GUI capability.

**arXiv ID:** 2609.12394
</details>

<details>
<summary><strong>VRL-Bench: Benchmarking agents on computer control tasks under finite trial budgets</strong> - Yu Bai, Yukai Miao, Dawei Wang, Li Chen, Yanyu Ren, Yuqian Shi, Dan Li, Ying Xiong, Chengqiu Tan, Run Zhou, Li Li - [[pdf]](https://arxiv.org/pdf/2609.12404)</summary>

**Abstract:** Learning from trial and error is a promising way to improve language agents on complex tasks such as computer control. Reflexion introduced verbal reinforcement learning, which turns failed trials into text that guides later attempts without updating model parameters. We introduce VRL-Bench, a harness for fair evaluation of trial-and-error learning under finite trial budgets. Across three models on MiniWoB and WebShop, we evaluate updates from several prominent verbal-memory methods spanning Reflexion and later work: each improves observed success over memory-free retry in some settings but reduces it in others. Replay experiments show that using reflection can reduce success rates, revealing a trade-off between exploiting experience and continued exploration. We propose VEX$^2$, a verbal exploration--exploitation scheduler that uses a language model to jointly select policies and allocate the remaining trial budget. VEX$^2$ is the only evaluated update to achieve positive observed success-rate gains over retry in all six settings.

**arXiv ID:** 2609.12404
</details>

<details>
<summary><strong>EvoRS: On-Policy Self-Evolution of Reward Systems for Open-Ended Reinforcement Learning</strong> - Weiyuan Li, Aili Chen, Xintao Wang, Yikai Zhang, Qingqing Dong, Jinghan Xu, Hongru Hou, Wenxuan Zhao, Chengkun Lang, Jun Gao, Yuanli Guo, Hongcheng Guo, Yanghua Xiao, Deqing Yang - [[pdf]](https://arxiv.org/pdf/2609.12459)</summary>

**Abstract:** Open-ended reinforcement learning often relies on rubric-based rewards for tasks without directly verifiable answers. Yet the policy and reward system form a dynamic feedback loop: as the policy optimizes the current reward, an initially useful reward system may become unreliable due to reward hacking or reduced response discriminability. The reward system should therefore evolve rather than remain fixed during training. Existing dynamic-rubric methods adapt evaluation criteria, but reward failures can also arise from scoring mechanisms or signal composition. We introduce EvoRS, a self-evolving RL framework that evolves the reward system from on-policy experience, representing it as an executable Reward-DAG. Specifically, an agentic designer updates this system from on-policy rollouts and reward traces to maintain train-time reliability. Across writing and roleplay, EvoRS achieves the best quality under all three judges, outperforming the policy by \(2.107\) and \(4.767\) points, respectively, while reducing reward hacking and coverage failures and preserving reward informativeness. Ablations confirm that a comprehensive fixed reward system cannot remain reliable in open-ended tasks and must evolve throughout training.

**arXiv ID:** 2609.12459
</details>

<details>
<summary><strong>Pelican-Sim 1.0: A General World Model Simulator for Embodied Intelligence</strong> - Shilong Zou, Shilin Zhang, Yingji Zhang, Yuhang Huang, Yi Zhang, Zeyuan Ding, Han Dong, Junwei Liao, Yong Dai, Jian Tang, Xiaozhu Ju - [[pdf]](https://arxiv.org/pdf/2609.12036)</summary>

**Abstract:** In this technical report, we propose Pelican-Sim 1.0, a general world model simulator for embodied intelligence that predicts future observations from visual context and robot actions to support downstream learning and decision making. The model incorporates four key design features: (1) Unified action representation: a 28-dimensional action value space covering most mainstream embodiments, keeping one model valid across heterogeneous devices. (2) Action-visual injection: URDF- and camera-rendered action videos bridge actions and pixels, giving markedly better controllability across embodiments, scenes, and tasks (PSNR +0.904 over alternative fusion baselines). (3) Sparse mixture-of-experts (MoE): sparse MoE layers add capacity for heterogeneous dynamics and absorb the action modality while reducing inter-modality conflict (FVD -6.530 vs. the dense backbone). (4) Efficient rollout generation: causal adaptation and few-step distillation yield a four-step autoregressive simulator, achieving a 5.67-fold speedup over the 35-step model. Benefiting from these designs, we train on approximately one million real-world and simulated trajectories and obtain large gains in action controllability and video quality: PSNR improves over the strongest evaluated baselines by 4.636 on AgiBotWorld Beta, 2.080 on RoboMIND, and 10.343 on RoboTwin, with the adapted EWMBench DYN score up 0.426 on RoboTwin. Relying on this, four downstream applications on RoboTwin succeed: 500 generated trajectories added to 50 demonstrations per task raise policy success from 70% to 93%; policy evaluation reaches a Pearson correlation of 0.994 across five checkpoints; and relative success gains reach 47.7% for action selection and 20.3% for policy improvement. Qualitative generalization across trajectory, scene, object, embodiment, and viewpoint shifts highlights its potential as a general-purpose world model simulator.

**arXiv ID:** 2609.12036
</details>

<details>
<summary><strong>Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models</strong> - Yuxin Xiao, Sheng Zhang, Chandan Singh, Tristan Naumann, Hoifung Poon, Jianfeng Gao, Xiaodong Liu - [[pdf]](https://arxiv.org/pdf/2609.12277)</summary>

**Abstract:** Electronic health record (EHR) foundation models trained on longitudinal patient trajectories have demonstrated strong performance across diverse clinical prediction tasks. However, their clinical reasoning capabilities remain constrained by next-token prediction on limited and incomplete EHR data. To address this, we propose a reinforcement learning (RL) fine-tuning framework that treats EHR foundation models as generative policies over patient trajectories. We formulate common clinical prediction problems (e.g., hospital readmission) as event-conditioned, time-windowed reasoning tasks. We then design time-aware, rollout-sensitive rewards to account for finite rollout lengths and temporally inconclusive outcomes. We find that RL fine-tuning consistently improves over pre-trained backbones and strong baselines. Notably, it enables smaller models to surpass larger pre-trained models in data-limited regimes and induces positive transfer across tasks. Further analysis shows that RL fine-tuned models generate trajectories with stronger structural and semantic alignment to ground truth and greater downstream utility.

**arXiv ID:** 2609.12277
</details>

<details>
<summary><strong>Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning</strong> - Fernando Palafox, David Fridovich-Keil - [[pdf]](https://arxiv.org/pdf/2609.12278)</summary>

**Abstract:** World models let agents plan by predicting the consequences of their actions, but changes in the environment can make them inaccurate. We study the problem of adapting a world model to an unknown test-time environment, drawn from a known environment family, using only a few episodes of interaction. Existing approaches trade off computational cost against expressivity, i.e., the range of models a method can produce. For example, in-context learning is computationally cheap but limited in expressivity, and gradient-based adaptation is expressive but computationally expensive. We present CLAW (Context-conditioned Low-rank Adaptation of World models), which addresses this tradeoff by using a hypernetwork to generate low-rank (LoRA) adapters at test time. During pretraining, we simulate adaptation to a variety of environments and jointly train the hypernetwork and base world model. At test time, we freeze the base model and use a forward pass of the hypernetwork to generate adapters from a small batch of test-time transitions. We evaluate CLAW in locomotion and manipulation environment families that vary in dynamics, embodiment, and reward. We show that, using only seconds of test-time data, CLAW outperforms gradient-based adaptation and in-context learning during online adaptation. We also show that CLAW avoids overfitting in data-scarce regimes, that its advantage comes from the expressive adapters rather than context conditioning, and that pretraining the hypernetwork jointly with the base model outperforms training it post hoc.

**arXiv ID:** 2609.12278
</details>

<details>
<summary><strong>LettuceVisSim: A Simulator That Generates Lettuce Image Time-series for Vision-Based Reinforcement Learning</strong> - Ziye Zhu, Bert van 't Ooster, Congcong Sun, Eldert van Henten, Sjoerd Boersma - [[pdf]](https://arxiv.org/pdf/2609.12505)</summary>

**Abstract:** Vision-based reinforcement learning holds strong potential for decision-making in controlled environment agriculture (CEA). However, its development is hindered by the scarcity of labelled crop images. To address this gap, LettuceVisSim, a lettuce growth simulator that generates labelled time series of crop images, was developed and validated. The simulator contains a process-based model (PBM) for shoot dry weight dynamics, a canopy layout algorithm for deriving canopy layout representations from shoot dry weight, and a Unity rendering engine for image generation. Five findings support the simulator. First, the PBM reproduced shoot dry weight under dynamic plant-density management with $\mathrm{R}^{2}=0.84$. Second, a piecewise cubic regression mapped shoot dry weight to potential projected area with $\mathrm{R}^{2}=0.94$. Third, the canopy layout representation was validated using 12 experimental datasets each having different dynamic environmental and spacing conditions. It reproduced the ground coverage ratio dynamics observed in measured images, achieving $\mathrm{R}^{2}=0.84$ when driven by measured shoot dry weight and $\mathrm{R}^{2}=0.40$ (0.76 excluding one outlier) when driven by PBM-simulated values. Fourth, the Unity rendering engine converted canopy layout representations into RGB and segmentation images at less than 10~ms. Fifth, a demonstration showed that a lighting-control policy can be learned and applied by observing only crop images that were generated with LettuceVisSim, providing a proof of concept of vision-based reinforcement learning in CEA using LettuceVisSim.

**arXiv ID:** 2609.12505
</details>

<details>
<summary><strong>Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents</strong> - Zhutao Lv, Chenhao Dang, Yi Feng, Yanpei Gong, Xiaolei Wang, Junyan Ye, Conghui He, Weijia Li - [[pdf]](https://arxiv.org/pdf/2609.12533)</summary>

**Abstract:** Real-world Earth observation (EO) agents must translate high-level scientific questions into executable workflows to acquire observations, prepare data, perform domain computations, and derive conclusions from runtime evidence. Existing EO agents typically start from supplied observations, while benchmarks typically provide prepared inputs or candidate answers, leaving full-chain open-world EO execution largely untested. We present Earth-Agent-Pro, an execution-adaptive Plan-and-Execute framework using expert-authored skills to constrain planning and runtime tool use. Workflow-centered structured memory records planned steps, accepted evidence, and their dependencies, enabling repair of only the affected workflow suffix when runtime evidence invalidates a step. Separate large language model adapters use sequence-level supervised fine-tuning for planner workflow composition and node-level group relative policy optimization with locally verifiable rewards for executor tool-argument grounding. Earth-Bench-Pro instantiates 248 expert-curated task cores as 744 questions under three matched regimes. Its 248 Open-World Execution questions span RGB imagery, spectral observations, and remote sensing products, pairing high-level requests with runtime data requirements, executable trajectories, and open-ended answers grounded in execution evidence. With a shared GPT-5 backbone, Earth-Agent-Pro achieves 66.13% LLM-as-Judge accuracy, exceeding ReAct by 20.95 points in this metric and 24.44 points in Tools-In-Order. Joint adapter tuning raises Qwen3.5-9B LLM-as-Judge accuracy from 38.31% to 50.00%, an 11.69-point gain over the untuned configuration. Planning-only evaluation and execution with the reference workflow show that the adapters improve workflow composition and argument grounding, respectively. Code and datasets will be released soon.

**arXiv ID:** 2609.12533
</details>

<details>
<summary><strong>Online Video Agent Harness for Long Video Understanding</strong> - Sen Yang, Boqiang Duan, Jing Yang, Weihao Bo, Jie Liu, Boyuan Tong, Ze Feng, Wenkang Zhang, Jingdong Wang, Hua Wu - [[pdf]](https://arxiv.org/pdf/2609.12818)</summary>

**Abstract:** Long video understanding often behaves like a visual needle-in-a-haystack problem: query-relevant evidence is sparsely distributed across long temporal spans, while packing dense frames into a single VLM context incurs \textit{context rot} and high cost. Existing video agents often rely on query-agnostic offline preprocessing or ad hoc tool sets, which can miss query-specific details and waste computation. In this work, we present VideoXAgent, a purely online video-agent harness for long video understanding that starts from the given video file and user query, plans and decomposes the task, invokes specialized expert tools on demand, and aggregates multimodal evidence to produce a final answer while resolving conflicts among observations. To support this on-demand invocation, we design a suite of heterogeneous expert tools guided by a data-driven taxonomy of atomic capabilities, spanning scripts, VLMs, and domain models (e.g., detection, OCR, ASR, face recognition). The harness further enforces objective evidence prompting and budget-aware control to curb hallucination and non-termination. Across Video-MME-Long, LongVideoBench-Long, LVBench, and MINERVA, VideoXAgent is competitive with frontier LMMs and video agents under a smaller context footprint---about 50k tokens of agent context per sample, even on hour-long videos. In particular, on complex video-reasoning benchmarks such as MINERVA, it matches this level while using only about 15\% of the context of a 1,024-frame dense-packing baseline. Notably, the harness remains effective with a visually weak or even text-only orchestrator, suggesting that strong long-video understanding can emerge from progressive agentic evidence seeking rather than from packing the full video into a single context. Project page: this https URL

**arXiv ID:** 2609.12818
</details>

<details>
<summary><strong>TileNet: Tile-Based CNN-SVM Architecture for Autonomous Unmanned Aerial Systems Inspection of Flat Roofs</strong> - Samuel Dunthorne, Hashim A. Hashim - [[pdf]](https://arxiv.org/pdf/2609.13013)</summary>

**Abstract:** Flat roofs are among the most influential components of the building envelope, governing both structural performance and thermal efficiency, and thereby contributing directly to household energy consumption, carbon emissions, and long-term environmental sustainability. Timely detection of roof defects is essential for reducing heating and cooling losses, preventing moisture-driven degradation such as mold growth, and supporting national climate-change mitigation goals. This paper presents a real-time, Unmanned Aerial System (UAS)-based deep learning framework that autonomously detects defects using live imagery captured during dual-altitude aerial passes. The multi-resolution flight strategy is designed to aid the identification of both small, fine-scale defects and larger structural issues, enabling more comprehensive assessments. To meet the strict computational and power constraints of embedded UAS hardware, the proposed framework integrates a tile-based architecture with a lightweight Convolution Neural Network-Support Vector Machine (CNN-SVM) classifier designed for low-latency onboard inference. The final model-comprising five convolutional layers and four dense layers, the last a linear SVM head, achieved a mean test accuracy of $94.4\%$ ($95\%$ confidence interval $\pm0.4\%$ over three seeds) on a photo-level split ($43,383$ training, $3,869$ validation, and $2,540$ test tiled and augmented images), outperforming GoogLeNet ($89.2\%$) and AlexNet ($79.8\%$). Experimental evaluations using real UAS imagery collected by onsite visits with DJI Matrice 350 RTK drone demonstrate that the system supports rapid, repeatable, and safe roof inspections while reducing human risk, lowering operational costs, and enabling more sustainable building maintenance.

**arXiv ID:** 2609.13013
</details>

<details>
<summary><strong>Groupoid-Based Internal State Representations for Reinforcement Learning with Local Symmetries</strong> - Ben Opperman, Eduardo Alonso, Esther Mondragón - [[pdf]](https://arxiv.org/pdf/2609.13035)</summary>

**Abstract:** Symmetries play a central role in reducing the complexity of reinforcement learning problems, yet most existing approaches rely on fixed group actions or predefined state abstractions. Classical reinforcement learning algorithms typically assume a globally structured Markov decision process with uniformly applicable actions and transitions, an assumption that limits their ability to exploit modularity and local, context-dependent regularities present in many realistic environments. We propose a reinforcement learning framework using groupoids to capture local, state-dependent symmetries and support the dy- namic discovery of equivalence structures during interaction. The agent maintains orbit representatives together with transporters that map raw states to canonical forms, enabling learning and decision-making to be performed in a symmetry-reduced space while preserving local distinctions. Empirical results demonstrate that the proposed groupoid-based approach improves sample efficiency and convergence in dense and large-scale environments exhibiting strong partial symmetries, yielding substantial performance gains over standard Q-learning. These findings show that dynamically exploiting local symmetry provides a practical and mathematically principled route to scalable and generalisable reinforcement learning.

**arXiv ID:** 2609.13035
</details>

<details>
<summary><strong>ASTRIL-MPC: Autonomous Traversal Framework of Articulated Tracked Robots with Language-Guided Neural-Kinematic MPC</strong> - Zhenfeng Gan, Yanbo Chen, Lirong Che, Junbo Tan, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2609.13083)</summary>

**Abstract:** In urban search and rescue, articulated tracked robots (ATRs) must traverse structured but contact-rich environments such as stairwells and cluttered building interiors. Reliable autonomy remains challenging because robot-terrain interaction (RTI) is hybrid and discontinuous, and effective flipper-track coordination is difficult to model analytically. We present ASTRIL-MPC, a language-guided neural kinematics model predictive control (MPC) framework for autonomous traversal. A learned kinematics model predicts short-horizon task-state increments from a height sequence and recent trajectories; NMPC plans with multi-objective costs and strict feasibility constraints; and a large language model (LLM) proposes bounded updates to selected weights and bounds through a safety-checked interface with range clipping, rate limiting, and consistency checks. The compiled predictor enables a full control cycle within 100 ms. Across three traversal tasks and a multi-height generalization setting, ASTRIL-MPC improves an aggregate traversal-quality score by up to 71% over a non-adaptive NMPC and by 67% over a PPO baseline, while eliminating measurable collision impacts during descent. These results indicate that combining learned kinematics, optimization-based planning, and language-guided retuning yields data-efficient and robust autonomy for articulated tracked robots.

**arXiv ID:** 2609.13083
</details>

<details>
<summary><strong>The Agent Incident Registry: Toward Preventing Repeated AI Agent Failures</strong> - Divyanshu Kumar, Rohith HN, Nitin Aravind Birur, Sahil Agarwal, Prashanth Harshangi - [[pdf]](https://arxiv.org/pdf/2609.11030)</summary>

**Abstract:** AI agents increasingly act through tools and delegated authority, but general incident repositories rarely capture the mechanisms needed to compare public failures with agent-security evaluations. We present the Agent Incident Registry (AIR) (Project page: this https URL), a source-linked catalog containing 487 records of agent-related events disclosed from 2022 through 2026. Each record includes supporting evidence, a stable identifier, and missingness-aware labels for causal role, disclosure class, mechanism, and outcome. Among the 336 generative-system records in which the agent acted, 81 involved realized harm (24\%). Realized outcomes concentrate in in-the-wild and safety-failure records, while responsible disclosures and research demonstrations are overwhelmingly demonstrated; the aggregate share therefore characterizes collection composition rather than deployment risk. After initial curation, a second human reviewer checked all 487 records and their existing labels for completeness and correctness. In a deployment-analogue audit, InjecAgent's 1,054 cases occupy three of AIR's twelve surfaces and are all attacker-triggered, whereas AIR contains 92 no-adversary safety failures. AIR supports source-grounded case retrieval and evaluation-scope auditing, not failure-rate or control-efficacy estimation.

**arXiv ID:** 2609.11030
</details>

<details>
<summary><strong>UltraQuant: 4-bit KV Caching for Context-Heavy Agents</strong> - Inesh Chakrabarti, David Limpus, Aditi Ghai Rana, Bowen Bao, Spandan Tiwari, Thiago Crepaldi, Ashish Sirasao - [[pdf]](https://arxiv.org/pdf/2606.20474)</summary>

**Abstract:** Context-heavy agents place substantial pressure on the key-value (KV) cache: long prefixes are reused across many short turns, while concurrency determines whether the serving system can keep GPUs utilized. We study 4-bit KV-cache compression for this setting, using TurboQuant-style rotation and codebook quantization as a quality anchor and vLLM FP8 KV caching as the deployment anchor. We report three contributions. First, we frame 4-bit KV caching around multi-round agent workloads where task quality, cache residency, and serving throughput must be measured jointly. Second, we describe the practical design choices needed to make the 4-bit path robust, including asymmetric K/V treatment, Walsh-Hadamard rotation, QJL removal, and block-scale variants. Third, we present serving optimizations on AMD GPUs, including optimized decode-attention kernels and UltraQuant, an FP4 approximation path that uses FP8 queries, FP4 KV tensors, UE8M0 group scales, and native scaled-MFMA support on CDNA4. On an adaptive-SLO replay of production Claude Code traces, UltraQuant sustains 2.71x (MiniMax-M2.5) and 4.38x (Qwen3-235B) the qualified-request throughput of the BF16 baseline, matching or exceeding hardware FP8 KV while using half the KV bytes. UltraQuant delivers its largest gains in long-context, high-concurrency, memory-constrained serving regimes.

**arXiv ID:** 2606.20474
</details>

<details>
<summary><strong>The Mechanics of a Swarm: A Reproducible External Reconstruction of an Unintended Agent-Coordination Episode on a Third-Party Wiki</strong> - Philipp Lütje - [[pdf]](https://arxiv.org/pdf/2609.12748)</summary>

**Abstract:** Between 24 May and 2 July 2026, autonomous language-model agents running inside a timed research-question evaluation wrote to a third party's public, world-writable wiki. OpenAI acknowledged the incident; independent researchers reconstructed it and published the wiki's archived revision history. We analyse that history (14,591 revisions, 3,103 names, 4,579 pages, 19,913 server events) as a behavioural record, attributing text to the revision that added it rather than to cumulative page content. Under an explicit identity model we reconstruct 907 cohorts and, from a random calendar marker the environment attached to each episode, estimate about 876 episodes (95% interval 774-995; alternative reconstructions span 800-1400). Coordination formats converged within a day, and the schedules created large opportunities for information asymmetry: because episodes of the same question chain ran at different internal-clock rates and started up to 16 h apart, the first report of an item preceded a later cohort's arrival by a median of 3.4 h. The three schedule parameters agents reported share one latent speed scale (78% of log-variance over 15 configurations), and in one task family the last observed activity clusters by reported speed class on the internal clock, compatible with a fixed internal-time horizon. Across the 510 cohorts with an observable, format-dependent progress trace, we find no robust positive association between measured coordination and documented progress, including the few demonstrably given a future answer. Because the export contains neither successful-read logs, harness messages nor ground-truth outcomes, these results do not identify the causal origin of the coordination or its effect. We report four claims from our earlier analysis that did not survive re-examination, and argue that read and outcome logging are requirements for agent-evaluation environments.

**arXiv ID:** 2609.12748
</details>

<details>
<summary><strong>AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization</strong> - Ji Liu, Saptarshi Majumder, Yiqing Huang, Wenwen Ouyang, Umang Pandey, Zeping Li, Chushi Chen, Zihao An, Puyuan Yang, Zekai Li, Sina Rafati, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Sharon Zhou, Emad Barsoum - [[pdf]](https://arxiv.org/pdf/2609.12471)</summary>

**Abstract:** We introduce AMDKernelVault, an open HIP and Triton kernel corpus and training framework for recent AMD CDNA GPUs. Existing LLM-based kernel agents are largely CUDA/NVIDIA-centric and often depend on repeated frontier-LLM calls for generation, reflection, and optimization. To address this gap, we develop HIPKernelGen and TritonKernelGen, agent-driven pipelines that transform PyTorch references into HIP or Triton kernels, compile and validate candidates under ROCm, and latency-profile them on AMD hardware. The corpus contains 62,153 execution-verified HIP kernel samples, 2,377 production-grounded ROCm Libraries QA entries, and 39,893 Triton kernels. We further train Qwen3-8B with supervised fine-tuning and execution-aware reinforcement learning as a demonstration of the corpus's utility. Under fixed evaluation budgets, it achieves the highest correctness among the compared models on PyTorch-to-HIP (34.0% Pass@1), TritonBench-G (33.2% Corr@3), and ROCmBench (41.94% Corr@3), but does not uniformly lead compilation or speed metrics. The corpus and documentation are available at this https URL, and the associated training and kernel-generation code is available at this https URL.

**arXiv ID:** 2609.12471
</details>

<details>
<summary><strong>Expert-Space Exploration in MoE Reinforcement Learning</strong> - Hongyi He, Zhenghao Lin, Xiao Liu, Peng Cheng, Yan Lu, Yeyun Gong - [[pdf]](https://arxiv.org/pdf/2609.13058)</summary>

**Abstract:** Reinforcement learning (RL) has become central to post-training of large language models. Recent advances in RL for Mixture-of-Experts (MoE) models have primarily focused on improving optimization stability and training efficiency, while treating the expert selection as a fixed component. Since routing determines the sparse computation paths that induce output distributions, expert selection offers an additional source of rollout diversity. Through empirical analysis, we find that perturbing expert routing effectively alters model output and increases rollout diversity, which is similar to increasing the decoding temperature. However, direct perturbation can activate unsuitable experts and substantially degrade rollout quality. Motivated by these observations, we introduce Expert-Space Exploration Reinforcement Learning (ESRL), an architecture-aware framework that explicitly explores the expert-routing space of MoE models. ESRL preserves high-confidence experts as anchors, and restricts stochastic routing to a plausible candidate pool, thereby retaining reliable computation paths. The perturbation strength is further adapted according to router entropy to avoid over-perturbation. To mitigate the routing mismatch introduced by perturbation, ESRL records the expert paths used during rollout and replays them during policy optimization. Experiments demonstrate that ESRL achieves the best performance across MoE backbones with top-K, top-1, and shared-expert routing, as well as across mathematics, science, and code tasks without additional sampling or computational cost. Specifically, ESRL on Qwen3-30B-A3B achieves the best among all compared methods, improving average Pass@1 and Pass@8 over GRPO by 3.2 and 4.5 percentage points, respectively. Further analyses of expert utilization and training dynamics provide insights into how exploiting MoE-specific routing structure benefits RL training.

**arXiv ID:** 2609.13058
</details>

<details>
<summary><strong>Text-to-SPARQL Generation with Reinforcement Learning: A GRPO-based Approach on DBLP</strong> - Jann Pfeifer, Debayan Banerjee, Ricardo Usbeck - [[pdf]](https://arxiv.org/pdf/2605.20066)</summary>

**Abstract:** Knowledge graph question answering seeks to translate natural language questions into executable queries over knowledge graphs, but existing approaches often rely on large models or full supervision in the form of gold query annotations. This study examines whether reinforcement learning with outcome-based rewards can train a small instruction-tuned language model to perform zero-shot Text-to-SPARQL generation in the scholarly domain. Group-Relative Policy Optimization (GRPO) is applied to the Qwen3-1.7B model on DBLP-QuAD, using prompts that combine natural language questions with symbolic hints about entities and relations. Training relies on execution feedback, structural constraints, and answer-level rewards, with an additional variant that incorporates gold-query-based shaping. The resulting models are compared to the unmodified zero-shot baseline and to a supervised DoRA-finetuned baseline across answer-level accuracy, execution accuracy, category-wise scores, and generalization to held-out templates. GRPO substantially improves over the zero-shot baseline and exhibits competitive generalization, while supervised DoRA finetuning achieves higher overall accuracy on the same model scale. Ablation analyses indicate that execution-based rewards account for most gains, with additional shaping yielding limited additional benefit, suggesting that outcome-based reinforcement learning is a viable training strategy when gold queries are unavailable for token-level supervision.

**arXiv ID:** 2605.20066
</details>

<details>
<summary><strong>FINESSE: An Agent-Based Simulator and Benchmark Dataset for Multimodal Financial Event Sequences</strong> - Tyler Farnan, Benjamin Eng, Adam Abate, Xirui Hou, Rizal Fathony, Nam H. Nguyen, Senthil Kumar - [[pdf]](https://arxiv.org/pdf/2609.11993)</summary>

**Abstract:** Machine learning research in financial services is limited by the scarcity of representative open-source datasets. Existing resources are often narrowly focused on a single modality or task and fail to reflect the structured, multimodal, and dynamic nature inherent to many problems in financial services.
In this paper, we introduce FINESSE, a Financial Event Sequence Simulation Environment, an agent-based simulation framework for generating synthetic, structured datasets composed of multiple interdependent event streams. Each stream corresponds to a distinct financial behavior such as transactions, payments, account status changes, and policy interventions, each with unique action spaces, schemas and variable types. These streams are coupled through agents' latent evolving states, enabling the simulation of temporally rich interactions.
We also introduce FINESSE-Bench, a benchmark dataset generated by the simulator, supporting four representative tasks: balance forecasting, transaction fraud detection, missed payment prediction, and next event prediction. We report baseline results using methods from time series forecasting, event sequence modeling, temporal graphs, and temporal point processes. We release the FINESSE framework, including the simulator and dataset to accelerate research on structured, multimodal event sequence modeling challenges in financial services.

**arXiv ID:** 2609.11993
</details>

<details>
<summary><strong>Certified Safety Curation: Distribution-Free Guarantees for Safe Offline Reinforcement Learning</strong> - Adam Haroon, Cody Fleming - [[pdf]](https://arxiv.org/pdf/2609.12014)</summary>

**Abstract:** Safe offline reinforcement learning assumes a cost function on every transition. We ask what remains possible when safety can be judged only by comparing short clips and occasionally asking whether an episode exceeded its budget. Certified safety curation answers with a filter-then-clone pipeline: a state-only value trained from segment comparisons scores whole trajectories, Learn-then-Test calibration certifies a selection threshold under a distribution-free $(\alpha, \delta)$ bound on the unsafe fraction of the selection, and behavior cloning follows. We are not aware of prior work certifying the composition of a training set for offline RL or imitation. Oracle controls justify the design: reweighting individual transitions fails even with an exact value, so the value selects whole trajectories. The policies satisfy the cost budget on eleven of fifteen DSRL tasks, one short of cloning the ground-truth safe subset, which needs a label on every trajectory; the uncertified variant reaches twelve. Retrained on the certified selection, the strongest full-label method becomes safe where no setting of its own cost target rescues it. Refusal is predictable: the certificate's probability has a closed form in the purity the pool attains, which the calibration sample estimates and the scorer enters only through.

**arXiv ID:** 2609.12014
</details>

<details>
<summary><strong>Inverting Self-Triggered Control: Adversarial Reinforcement Learning for Sparse Denial-of-Service Attacks</strong> - Adam Haroon, Erick J. Rodríguez-Seda, Tristan Schuler, Cody Fleming - [[pdf]](https://arxiv.org/pdf/2609.12016)</summary>

**Abstract:** Self-triggered reinforcement learning control (RL-STC) learns the sparsest control schedule that preserves Lyapunov-decreasing stability under a Run-Time Assurance (RTA) override. We invert this: an adversarial RL agent learns the sparsest jamming or Denial-of-Service (DoS) schedule that destabilizes the closed loop, with a Lyapunov-increase admissibility predicate mirroring the defender's safety certificate. We prove a plant-property lower bound on the minimum jam count required for an immediate hold-last medium-access-control adversary to force a crash against a self-triggered controller (STC) satisfying a Lyapunov contract, and recover a certificate-level analog of the consecutive-grouping optimality of prior count-budget DoS scheduling as a corollary. This extends the DoS-scheduling count-budget analysis from periodic and linear-time-invariant to STC controllers. Empirically, we train against four fixed defenders per plant (one Linear Quadratic Regulator (LQR) and three RL-STC) on Pendulum, CartPole, and Quadrotor2D. The learned adversary is the only adversary that crashes every defender on every plant at $100\%$: greedy misses Quadrotor2D LQR on $42\%$ of episodes and periodic misses Pendulum LQR on $97\%$. On jam-time-per-failure it beats baselines by up to $2.8\times$, and shows its widest absolute margin on Quadrotor2D LQR. Robustness ablations show that Gaussian observation noise exceeding the initial-state magnitude and position-only observation both preserve $100\%$ failure rate and keep the learned adversary strictly ahead of both baselines on jam-time-per-failure.

**arXiv ID:** 2609.12016
</details>

<details>
<summary><strong>Reinforcement Learning for Syndrome Extraction</strong> - John Zhuoyang Ye, Aarav Pabla, Jens Palsberg - [[pdf]](https://arxiv.org/pdf/2609.12020)</summary>

**Abstract:** A key subtask of quantum error correction is to extract a syndrome that, if nontrivial, signals an error. The number of possible ways to extract a syndrome grows exponentially with the syndrome size, and these implementations vary greatly in fault tolerance, as measured by their logical error rates. This creates a natural search problem: find an implementation with a low logical error rate. Previous work solves this problem but sacrifices either solution quality or scalability. In this paper, we use reinforcement learning and importance sampling to outperform previous work at all scales. Compared with the state of the art automatic scheduling tools AlphaSyndrome and PropHunt, our tool reduces the logical error rate by 25.9\% and 71.7\% on average, respectively, culminating with a reduction of 97.8\% for a surface code with distance 15.

**arXiv ID:** 2609.12020
</details>

<details>
<summary><strong>Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning</strong> - Bereket Sitotaw Kidane, Md Samiul Haque Motayed, Shuo Wang - [[pdf]](https://arxiv.org/pdf/2609.12264)</summary>

**Abstract:** Designing effective chemotherapy regimens is hindered by tumor heterogeneity and drug resistance, which complicate the deployment of patient-specific model-based optimal control across diverse populations. We develop and compare closed-loop deep reinforcement learning (DRL) dosing policies with continuous (TD3) and discrete (DQN) action spaces trained on a high-dimensional heterogeneous tumor model. The DRL policies are benchmarked against a Pontryagin's Maximum Principle (PMP)-derived open-loop benchmark. We assess generalization under parametric heterogeneity using a 100-patient virtual cohort with plus or minus 10 percent uniform perturbations in growth and drug-sensitivity parameters. Across this cohort, TD3 achieves higher average tumor reduction, while DQN yields tighter inter-patient dosing consistency, revealing a clear efficacy-consistency trade-off in this study. Our simulations assume full observation of all tumor subpopulations; translation to sparse and noisy clinical measurements will require partial-observability formulations and/or state estimation. Overall, the results show that simulation-trained DRL can learn state-dependent feedback dosing policies that complement open-loop optimal control benchmarks.

**arXiv ID:** 2609.12264
</details>

<details>
<summary><strong>Curriculum-Based Adversarial Heterogeneous Agent Reinforcement Learning for Autonomous Quad-Copter Landing in Maritime Settings</strong> - Allan Minh-Tam Nguyen, Sree Showrya Kotala, Stefan Banioi-Crijman, Kurt Driessens, Rico Möckel - [[pdf]](https://arxiv.org/pdf/2609.12758)</summary>

**Abstract:** Recovering unmanned aerial vehicles (UAVs) in maritime environments is challenging due to wind turbulence and ship-deck motion, making it a valuable test case for alternative control and learning approaches as conventional landing approaches often become unreliable. We study simulated mid-air capture of quadrotor UAVs by a ship-mounted robotic arm, learning robust cooperative control policies with Heterogeneous-Agent Proximal Policy Optimization (HAPPO) Reinforcement Learning. We train with HAPPO using a curriculum and an adversarial wind agent (HARL-AC) in NVIDIA Isaac Lab, and compare the obtained control policies against those generated through curriculum-based domain randomization and a benchmark trained on a single sea state. In-distribution evaluation on sea states $0/4/5$ shows comparable success for HARL-AC and domain randomization of up to $97.5\%$. On out-of-distribution sea states $7/8/10$, HARL-AC generalizes better, achieving up to $16\%$ higher median success rate at sea state 10, and substantially lower crash rates of up to $14\%$ compared to the domain randomization policy. Furthermore, we show that the adversarially trained policy shows more cautious behavior, slightly increasing timeouts by $<3\%$, but yields safer recovery behavior in severe, unseen conditions.

**arXiv ID:** 2609.12758
</details>

<details>
<summary><strong>Offline Reinforcement Learning for Wind Farm Control: A Wind Tunnel Study under Dynamic Wind Directions</strong> - Yuhan Su, Hongyang Dong, Simone Tamaro, Filippo Campagnolo, Carlo L. Bottasso, Xiaowei Zhao - [[pdf]](https://arxiv.org/pdf/2609.12905)</summary>

**Abstract:** This paper addresses the wind farm power maximization problem in the presence of wind direction changes. Specifically, a model-free Modified Twin Delayed Deep Deterministic Policy Gradient with Behavior Cloning (MTD3-BC) algorithm is proposed to tackle this task through yaw control under varying wind direction conditions. MTD3-BC is an offline reinforcement learning (RL) algorithm that aims to infer good behavior from only a precollected offline dataset. Additionally, to ensure smooth and moderate yaw adjustments, a new action consistency term is introduced into the policy optimization objective. Unlike online RL methods, MTD3-BC does not require extensive interactions with a wind farm simulator during training, significantly reducing computational costs and training time. A wind tunnel experiment is conducted to validate the effectiveness of the algorithm under varying wind directions. The results demonstrate that MTD3-BC successfully mitigates wake effects, delivering farm-level power gains of approximately 10\% over the baseline greedy strategy and performance on par with a data-calibrated model-based wake-steering benchmark, while requiring no wake model and only a small fraction of the training cost of online RL. To our knowledge, this is the first time an offline RL wind farm control policy has been validated and demonstrated experimentally.

**arXiv ID:** 2609.12905
</details>

<details>
<summary><strong>MCRL2: Multi-resource Cross-attention-based Representation Learning-augmented Reinforcement Learning for Cloud Microservice Scheduling</strong> - Tiangang Li, Shi Ying, Xiangbo Tian, Chuan Shi, Ding Xiao - [[pdf]](https://arxiv.org/pdf/2609.13048)</summary>

**Abstract:** Efficient microservice scheduling is crucial for maintaining load balance across nodes in data centers and ensuring high quality of service. However, achieving this in practice remains challenging due to dynamic resource imbalance under fluctuating workloads, nonlinear coupling across multiple resource dimensions, and the heterogeneity of microservice resource demands. While reinforcement learning-based approaches have shown promise, they struggle to capture the complex interdependencies among heterogeneous resources and neglect the importance of learning informative system representations. To address these limitations, we propose MCRL2, a novel reinforcement learning approach augmented with multi-resource cross-attention-based representation learning for microservice scheduling. Specifically, we first propose MCRL, a novel representation learning approach that captures structured and informative interactions among nodes, resources, and microservices via a multi-resource cross-attention mechanism. Then, MCRL2 augments reinforcement learning through MCRL-enhanced actor-critic architecture combined with a maximum entropy objective, improving system state expressiveness and leading to more stable and effective scheduling decisions. Extensive experiments on real production cluster traces demonstrate that MCRL2 significantly outperforms existing baselines in load balancing, scheduling success rate and average completion time across diverse workload patterns.

**arXiv ID:** 2609.13048
</details>

<details>
<summary><strong>A Unified and Constrained View of Regularization-Based Robust Reinforcement Learning</strong> - Amine Andam, Jamal Bentahar, Mustapha Hedabou - [[pdf]](https://arxiv.org/pdf/2609.13050)</summary>

**Abstract:** Regularization-based methods have become a standard approach for training Deep Reinforcement Learning policies against adversarial input perturbations. In this paper, we unify these methods by deriving new upper bounds on the performance gap between the nominal and worst-case policies. Each upper bound is expressed as an existing regularization objective plus a KL-divergence penalty between the nominal and worst-case policies, which further explains why adding a KL penalty improves robustness in practice. Building on these bounds, we formulate robust training as a constrained optimization problem, showing that existing methods correspond to the special case of a fixed Lagrange multiplier. We instead update the multiplier jointly with the policy to automatically tune the regularization weight. Finally, we conduct extensive adversarial evaluations across several continuous control tasks to validate our theoretical analysis.

**arXiv ID:** 2609.13050
</details>

<details>
<summary><strong>CanvasAnneal: Curriculum Reinforcement Learning for Diffusion Language Models</strong> - Blake Olson, Yuhang Song, Emmett McQuinn, Yuan Shangguan - [[pdf]](https://arxiv.org/pdf/2609.13060)</summary>

**Abstract:** Diffusion Language Models (DLMs) offer promising parallel generation capabilities but lag behind autoregressive models in complex reasoning and tool-use tasks. While Reinforcement Learning (RL) has recently been applied to enhance DLMs, standard RL approaches suffer from an exploration bottleneck. To address this, we inject reasoning priors from a stronger teacher model to guide RL exploration. In this paper, we introduce CanvasAnneal, a curriculum-guided diffusion RL framework. During the initial RL phase, we warm-start exploration by injecting teacher-generated reasoning traces into the initial diffusion canvas. As training progresses, we gradually remove this guidance and require the model to generate more of the reasoning trajectory independently. Across mathematical reasoning and tool-use benchmarks, CanvasAnneal improves over standard diffu-GRPO on MATH500, Countdown, and Tau2 and substantially accelerates reward improvement on several tasks, while gains are task-dependent. Our results suggest that structured training-time guidance can alleviate exploration bottlenecks in diffusion RL and speed up convergence on harder tasks.

**arXiv ID:** 2609.13060
</details>

<details>
<summary><strong>Towards Sustainable Hydrogen Systems: Supply Chain Optimization with Model Predictive Control and Reinforcement Learning</strong> - Mahammad Valiyev - [[pdf]](https://arxiv.org/pdf/2609.11933)</summary>

**Abstract:** Hydrogen supply chains are expected to play a central role in future low-carbon energy systems by enabling renewable energy integration, long-duration storage, and decarbonization of industrial and transportation sectors. However, their operation is challenged by renewable generation variability, electricity price fluctuations, uncertain hydrogen demand, and engineering constraints associated with electrolyzers, energy storage, and grid interaction. As hydrogen infrastructure expands toward commercial deployment, operational strategies must balance economic performance, reliability, and sustainability under dynamic and uncertain conditions.
This paper investigates and compares four control approaches for a renewable-powered hydrogen supply chain: a rule-based controller (RBC), model predictive control (MPC), reinforcement learning without forecasts (RL-NF), and reinforcement learning with forecast-augmented observations (RL-F). All methods are evaluated within a unified, physically realistic framework incorporating electrolyzer minimum-load and ramp-rate constraints, battery and hydrogen storage dynamics, grid import limits, and consistent economic assumptions, enabling a fair comparison under identical operating conditions.
Simulation results show that MPC achieves the highest economic performance by exploiting short-term forecasts to coordinate storage, reduce grid dependence, and improve efficiency. RL-NF demonstrates robust and competitive performance without future information, highlighting the capability of learning-based methods to discover effective policies from experience. RL-F does not consistently outperform its no-forecast counterpart, suggesting that forecast uncertainty and increased state complexity can limit forecast-augmented learning. The results provide guidance for selecting operational control strategies in future hydrogen energy systems.

**arXiv ID:** 2609.11933
</details>

<details>
<summary><strong>PEARL: Structural Privacy-Utility Control in Human-Centric CPS via Personalized Early-Exit Deep Reinforcement Learning</strong> - Mojtaba Taherisadr, Salma Elmalaki - [[pdf]](https://arxiv.org/pdf/2403.05864)</summary>

**Abstract:** In human-centric Cyber-Physical Systems (CPS), personalized Deep Reinforcement Learning (DRL) agents must share fine-grained control actions with cloud services, exposing sensitive private states to inference attacks by honest-but-curious adversaries. Static privacy models fail to address the dynamic nature of human interactions. This paper introduces PEARL (Personalized Early-exit Adaptive Reinforcement Learning), a novel framework that addresses this challenge through structural privacy control rather than data perturbation. PEARL deploys a dual-path Early-Exit Deep Q-Network (EE-DQN) at the edge, using Mutual Information (MI) between private states and observable actions to train per-branch binary labels: Utility Confidence Labels (UCL), verifying action quality, and Privacy Confidence Labels (PCL), verifying MI leakage remains below a user-defined threshold. At inference, PEARL selects the shallowest exit branch satisfying both UCL and PCL, structurally limiting shared action descriptive power without noise injection. An MI-based feedback loop tracks behavioral drift and triggers retraining when privacy-utility profiles shift, ensuring long-term robustness. Validated on a personalized smart-home HVAC system and a VR smart classroom, PEARL reduces adversarial state-inference accuracy by 25.67% on average with a controlled 10-16% utility cost, establishing a practical, dynamically enforceable privacy-utility tradeoff.

**arXiv ID:** 2403.05864
</details>

<details>
<summary><strong>High-Fidelity Multi-Body Simulator for Autonomous Racing</strong> - Nicola Musiu, Francesco Iacovacci, Fausto Lupo, Matteo Pini, Giovanni Scapicchi, Francesco Moretti, Eugenio Mascaro, Pietro Musso, Ayoub Raji, Marko Bertogna, Vincenzo Maria Arricale, Angelo Lo Sapio, Alessandro Piccarelli, Garron Fish - [[pdf]](https://arxiv.org/pdf/2609.12795)</summary>

**Abstract:** We present a custom high-fidelity vehicle dynamics simulation environment for testing and validation of Autonomous Racing software. The digital twin of the autonomous vehicle is developed in Dymola, using racecar dynamics modeling libraries to build a complete multi-body model. A 3D road surface, including elevation profiles and curbs, is implemented using the Curved Regular Grid (CRG) standard. The model is exported from Dymola as a Functional Mock-up Unit (FMU) and integrated into a custom software-in-the-loop simulator, where communication interfaces with the autonomous racing stack were developed in C++. A calibration procedure based on experimental data is also presented, along with a validation study to further support the quality of the proposed framework. The simulator runs in real time on a portable computer and provides reliable ground truth for algorithms validation prior to real-world deployment.

**arXiv ID:** 2609.12795
</details>

<details>
<summary><strong>Ego-Dynamics-Augmented World Model for Autonomous Driving with Zero-Shot Cross-Embodiment Adaptation</strong> - Zhidong Wang, Jingsong Liang, Zirui Li, Zhan Chen, Han Yu, Chen Lv - [[pdf]](https://arxiv.org/pdf/2607.13410)</summary>

**Abstract:** End-to-end autonomous driving requires generalization ability across platforms with dissimilar physical characteristics. The chassis defines the physical embodiment of each platform, and real-world fleets span sub-tonne microcars to bus-class vehicles. Consequently, the driving stack must either be retrained per platform or adapt to the underlying chassis dynamics online. World model (WM)-based reinforcement learning offers a sample-efficient path toward end-to-end autonomous driving on egocentric bird's-eye-view (BEV) representations, but its effectiveness hinges on how faithfully the WM captures the ego vehicle's dynamics. This work identifies a structural bottleneck in BEV-based WMs: observation transitions entangle ego-motion with scene dynamics, consuming modeling capacity at the cost of imagination accuracy. This burden is embodiment-dependent: dissimilar chassis produce different observation warps under the same control input. The proposed DynaDreamer addresses this bottleneck by conditioning the WM's latent distributions on a physics-informed ego-dynamics context derived from a lateral dynamics model with a neural tire force formulation. This context is extracted online via a neural-ODE encoder-decoder that simultaneously identifies the underlying chassis parameters. Information-theoretic analysis confirms that this conditioning removes the ego-motion terms from both the WM's transition entropy and its prior-posterior KL divergence. The identified physical parameterization enables zero-shot cross-embodiment adaptation across a dynamically diverse fleet without per-platform retraining. Simulation results show 28% and 43% improvements in driving task success rates over the strongest baseline in urban and highway scenarios, and the advantage over the base Transformer WM reaches up to 73% when extrapolating to unseen chassis.

**arXiv ID:** 2607.13410
</details>

<details>
<summary><strong>Synthetic TLX: Forecasting Human Workload Using Agent Simulation</strong> - Tzu-Sheng Kuo, Carrie J. Cai, Meredith Ringel Morris, Michael Terry - [[pdf]](https://arxiv.org/pdf/2609.12273)</summary>

**Abstract:** Assessing human workload for technology-mediated tasks helps prevent task failure caused by poor technology design. Traditionally, workload is assessed retrospectively using the NASA Task Load Index (TLX) after humans complete a task. What if we could forecast workload before a human attempts a task using agent simulation? We introduce Synthetic TLX, a new paradigm for proactive workload estimation that predicts NASA TLX scores for a given task, unlocking novel interaction opportunities and evaluation methods. To understand its viability, we conducted three experiments comparing human and agent-generated scores to evaluate where they align and diverge. We found agent estimates align with human scores particularly when prompted with a human persona and active task simulation. However, agents and humans diverge in the sources of workload they are sensitive to. Based on our findings, we present three applications to showcase Synthetic TLX's potential and discuss the future of workload-aware human-AI interaction.

**arXiv ID:** 2609.12273
</details>

<details>
<summary><strong>From Review to Reuse: How Post-Task Workflow Can Support Human-AI Agent Interaction</strong> - Zekun Wu, Xinru Wang, Rock Yuren Pang, Chenglong Wang, Anna Maria Feit - [[pdf]](https://arxiv.org/pdf/2609.13136)</summary>

**Abstract:** AI agents can automate tasks by turning a single natural-language request into a multi-step process spanning tools, files, and applications. Users are often left to judge that process from fragmented execution information and the final output. To make the completed process easier to understand, validate, and reuse, we investigate post-task workflows: editable, graph-based representations of an agent's completed execution. We first analyzed 10,803 public workflow templates from n8n to characterize real-world automation practice, then developed Trace2Flow, a research probe that translates agent execution traces into interactive post-task workflows. In a study, participants (N = 20) reviewed agent executions with prompt or agent errors. We found that post-task workflows improved their understanding and error detection over a prompt-only condition, and that validation succeeded mainly when users cross-checked across multiple evidence sources. For follow-up tasks, adapting the workflow matched adapting the prior prompt in success, time, and difficulty, and was often preferred.

**arXiv ID:** 2609.13136
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
