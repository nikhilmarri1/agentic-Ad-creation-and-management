# agentic-Ad-creation-and-management
The platform for small business owners to interact with to create ad campaigns, and manage them to achieve marketing goals

NOTE:**Actively being updated, working on the details**

---------------------------------------------------------------------------------------------------

The high level vision:

-> building a comprehensive agentic platform requires deterministic aspects among the probabilistic behaviour of LLMs. the aspects which can be deterministic should be deterministic

-> the purpose of any agentic platform is to provide the expertise that a person may lack for carrying out the desired work - while those agents should be reliable

-> the ability of agentic systems grows multi-fold depending on the architecture/workflow employed, and tools/deterministic algorithms made available to LLMs or woven into the architecture


-> reliability becomes a major issue when dealing with LLM agents due to several reasons, a few among those:

=> for complex cases, LLMs fail to determine the underlying logic and weigh the requirements and prioritize accordingly

=> LLMs being primarily built on a linear architecture, many a times fail to explore other possible better ways to deal with some problem -- as they inherently follow the most probable connection and go down that path. Hence, building something popular/existing feels like an instant with LLMs, while dealing with some innovative methods feels like a hassle.

=> the probability nature of the LLMs can be effected by various factors resulting in hallucinations, non-compliance with instructions, etc

---------------------------------------------------------------------------------------------------

**Some of the architectural decisions made, following the vision mentioned above:**

Modular architecture:
divide the overall goal into sections, managed by respective agents.
divide tasks in each section into multiple small components, and assign dedicated LLM nodes or deterministic algorithms as required

Guardrails and reflection:
verification nodes are included at all the viable positions in the architecture, after LLM nodes, to catch any non-compliance before taking it downstream in the workflow -- and correct it using reflection to the same node, without repeating the whole workflow

ML algorithms:
Employing ML algorithms for large-scale analysis and inference, and python code for math and logic implementation tools


