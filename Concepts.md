# LangChain Concepts Guide

This guide explains the core concepts of LangChain, LangGraph, and LangSmith in simple, easy-to-understand terms with practical examples.

---

## Table of Contents
1. [What is LangChain?](#what-is-langchain)
2. [What is LangGraph?](#what-is-langgraph)
3. [What is LangSmith?](#what-is-langsmith)
4. [How They Work Together](#how-they-work-together)

---

## What is LangChain?

### Simple Definition
**LangChain** is a framework that makes it easier to build applications powered by Large Language Models (LLMs) like ChatGPT, Claude, or GPT-4. Think of it as a toolkit that connects different AI components together.

### Why Do We Need LangChain?
Without LangChain, building with LLMs is like building with raw LEGO bricks—you have to handle everything manually. LangChain provides pre-built components that simplify this process.

### Core Components

#### 1. **Models (LLMs)**
The AI brains that generate responses.

```python
from langchain_openai import ChatOpenAI

# Initialize an LLM
llm = ChatOpenAI(model="gpt-4")

# Generate a response
response = llm.invoke("What is Python?")
print(response)
```

#### 2. **Prompts**
Templates that format your input to the LLM for better results.

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer this question: {question}"
)

# Use it
formatted_prompt = prompt.invoke({"question": "What is machine learning?"})
```

#### 3. **Chains**
Sequences of connected components that work together. Like a pipeline.

```python
from langchain_core.runnables import RunnableSequence

# Create a simple chain: prompt → LLM → output
chain = prompt | llm

# Run it
result = chain.invoke({"question": "Explain AI in 1 sentence"})
print(result)
```

#### 4. **Memory**
Remembers previous conversation history so the AI can understand context.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# Add conversation history
memory.save_context(
    {"input": "Hello"}, 
    {"output": "Hi there!"}
)

# Retrieve memory
print(memory.load_memory_variables({}))
```

#### 5. **Tools/Retrieval**
External data sources and tools your AI can use (search engines, databases, APIs).

```python
# Example: RAG (Retrieval Augmented Generation)
# 1. Search for relevant documents
# 2. Pass them to the LLM for accurate responses
# This prevents the AI from making up information
```

### Real-World Example: Customer Support Bot

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Step 1: Set up LLM
llm = ChatOpenAI(model="gpt-4")

# Step 2: Create a prompt template for customer support
template = """You are a friendly customer support agent. 
Help the customer with their issue: {user_message}

Previous context: {chat_history}
"""

prompt = ChatPromptTemplate.from_template(template)

# Step 3: Add memory
memory = ConversationBufferMemory()

# Step 4: Create a chain
support_chain = prompt | llm

# Step 5: Use it
response = support_chain.invoke({
    "user_message": "My order hasn't arrived",
    "chat_history": "Customer is upset about late delivery"
})

print(response)
```

---

## What is LangGraph?

### Simple Definition
**LangGraph** is a library for building **agentic workflows**—applications where AI makes decisions and takes actions based on conditions. It helps you create workflows with multiple steps, decision points, and loops.

### Why LangGraph?
LangChain's chains are linear (A → B → C). LangGraph lets you create complex workflows with loops, conditions, and multiple paths (like flowcharts).

### Key Concepts

#### 1. **Nodes**
Individual steps or tasks in your workflow.

```python
# Example: Different nodes for different tasks
def research_node(state):
    return "Research completed"

def write_node(state):
    return "Article written"

def review_node(state):
    return "Review completed"
```

#### 2. **Edges**
Connections between nodes that define the workflow flow.

```python
# Edges define how to move from one node to another
# research → write → review
```

#### 3. **State**
Data that flows through the workflow, updated at each step.

```python
# State is like a container that holds information
workflow_state = {
    "topic": "AI",
    "research_data": None,
    "draft": None,
    "final_article": None
}
```

#### 4. **Conditions**
Decision points that determine which node to go to next.

```python
# Example: Route based on sentiment
def route_feedback(state):
    sentiment = analyze_sentiment(state["user_feedback"])
    if sentiment == "positive":
        return "thank_customer"
    else:
        return "support_node"
```

### Simple LangGraph Example: Content Writer Workflow

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Step 1: Define the state
class ContentState(TypedDict):
    topic: str
    outline: str
    draft: str
    final_content: str

# Step 2: Define node functions
def create_outline(state):
    """Generate an outline for the topic"""
    outline = f"Outline for {state['topic']}:\n1. Introduction\n2. Main Points\n3. Conclusion"
    return {"outline": outline}

def write_draft(state):
    """Write the draft content"""
    draft = f"Draft for {state['topic']} based on: {state['outline']}"
    return {"draft": draft}

def review_and_finalize(state):
    """Review and finalize the content"""
    final = f"Final content: {state['draft']} (Reviewed)"
    return {"final_content": final}

# Step 3: Create the graph
graph = StateGraph(ContentState)

# Step 4: Add nodes
graph.add_node("outline", create_outline)
graph.add_node("draft", write_draft)
graph.add_node("review", review_and_finalize)

# Step 5: Add edges (define flow)
graph.add_edge(START, "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "review")
graph.add_edge("review", END)

# Step 6: Compile and run
workflow = graph.compile()

initial_state = {"topic": "Machine Learning"}
result = workflow.invoke(initial_state)
print(result["final_content"])
```

### Advanced Example: Agent with Decision Making

```python
# An agent that decides what to do based on user input
def router_node(state):
    user_input = state["message"]
    
    # Decision logic
    if "search" in user_input.lower():
        return "search"
    elif "calculate" in user_input.lower():
        return "calculator"
    else:
        return "general_qa"

graph.add_node("router", router_node)
graph.add_node("search", search_function)
graph.add_node("calculator", calculator_function)
graph.add_node("general_qa", qa_function)

# Conditional edges based on router output
graph.add_conditional_edges("router", router_node)
```

---

## What is LangSmith?

### Simple Definition
**LangSmith** is a platform for **monitoring, debugging, and evaluating** LLM applications. It's like a dashboard and debugger for your AI applications.

### Why LangSmith?
LLM applications are unpredictable. LangSmith helps you:
- Track what's happening inside your application
- Debug when things go wrong
- Evaluate performance and quality
- Improve continuously

### Key Features

#### 1. **Tracing**
Records every step of your LLM application execution.

```python
from langsmith import traceable

@traceable
def my_chatbot(user_message):
    # Every call to this function is automatically traced
    response = llm.invoke(user_message)
    return response

# When you run my_chatbot(), LangSmith logs:
# - Input: the user message
# - Output: the AI response
# - Latency: how long it took
# - Tokens used: cost information
# - Errors: if anything fails
```

**In the LangSmith Dashboard, you'll see:**
```
┌─────────────────────────────────────┐
│ Trace: my_chatbot                   │
│ ├─ Input: "What is AI?"            │
│ ├─ llm.invoke()                     │
│ │  ├─ Model: gpt-4                 │
│ │  ├─ Tokens: 145                  │
│ │  └─ Duration: 2.3s               │
│ └─ Output: "AI is..."              │
└─────────────────────────────────────┘
```

#### 2. **Debugging**
Find and fix issues in your application.

```python
# Without LangSmith: Black box
response = chain.invoke({"question": "What is AI?"})
# ❌ Got wrong answer... but why?

# With LangSmith: Full visibility
# ✓ See exact prompt sent to LLM
# ✓ See LLM response before processing
# ✓ See any data transformations
# ✓ See where the error occurred
```

#### 3. **Evaluation & Scoring**
Test your application's quality.

```python
from langsmith import evaluate

def check_answer_quality(prediction, reference_label):
    """Custom evaluation function"""
    if len(prediction) > 10:
        return {"score": 1.0}  # Good
    else:
        return {"score": 0.0}  # Poor

# Run evaluation on your test dataset
results = evaluate(
    my_chatbot,
    data=test_dataset,
    evaluators=[check_answer_quality]
)

# Get results like:
# ✓ Success Rate: 92%
# ✓ Average Latency: 1.2s
# ✓ Average Cost: $0.002
```

#### 4. **Datasets & Testing**
Manage test data and run experiments.

```python
from langsmith import create_dataset

# Create a test dataset
dataset = create_dataset("qa_dataset")

# Add test cases
dataset.add_example(
    inputs={"question": "What is Python?"},
    outputs={"expected_answer": "A programming language"}
)

# Run experiments
experiment = dataset.run(chain, experiment_name="v1")
experiment.compare(previous_experiment)
# See: Which version performed better?
```

#### 5. **Analytics & Insights**
Understand your application's performance over time.

```
LangSmith Dashboard shows:
- Total API calls: 10,000
- Average latency: 1.5s
- Error rate: 2%
- Most common failures: "Timeout errors"
- Cost per request: $0.001
- User satisfaction: 4.2/5 ⭐
```

### LangSmith Setup Example

```python
import os
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Step 1: Set API key (get from https://smith.langchain.com)
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my_project"

# Step 2: Create your application
llm = ChatOpenAI(model="gpt-4")

@traceable
def answer_question(question):
    return llm.invoke(question)

# Step 3: Use your application
result = answer_question("What is LangChain?")

# Step 4: Check LangSmith dashboard
# Everything is automatically logged!
# Go to: https://smith.langchain.com/projects
```

---

## How They Work Together

### The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your AI Application                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      LangChain Layer                      │   │
│  │  (Building blocks for AI: Prompts, LLMs, Memory, Tools) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LangGraph Layer                        │   │
│  │     (Orchestrates complex workflows with decisions)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                               ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  LangSmith Dashboard                      │   │
│  │    (Monitors, debugs, and evaluates everything)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Real-World Example: Customer Support Agent

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
import os

# Setup LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "customer_support"

# ═════════════════════════════════════════════════════════════
# Part 1: LangChain - Build individual components
# ═════════════════════════════════════════════════════════════

llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory()

# ═════════════════════════════════════════════════════════════
# Part 2: LangGraph - Orchestrate the workflow
# ═════════════════════════════════════════════════════════════

from typing import TypedDict

class SupportState(TypedDict):
    message: str
    category: str
    response: str

def categorize_message(state):
    """Step 1: Categorize customer message"""
    categories = ["billing", "technical", "general"]
    # Use LLM to categorize
    category = llm.invoke(f"Categorize: {state['message']}")
    return {"category": str(category)}

def generate_response(state):
    """Step 2: Generate appropriate response"""
    # Use memory and LLM
    response = llm.invoke(f"{state['message']} (Category: {state['category']})")
    memory.save_context({"input": state['message']}, {"output": response})
    return {"response": str(response)}

# Build the workflow
graph = StateGraph(SupportState)
graph.add_node("categorize", categorize_message)
graph.add_node("respond", generate_response)

graph.add_edge(START, "categorize")
graph.add_edge("categorize", "respond")
graph.add_edge("respond", END)

workflow = graph.compile()

# ═════════════════════════════════════════════════════════════
# Part 3: LangSmith automatically traces everything
# ═════════════════════════════════════════════════════════════

@traceable
def customer_support(user_message):
    return workflow.invoke({"message": user_message})

# Run it
result = customer_support("My billing is wrong!")

# ✓ LangSmith now shows:
#   - Categorization step took 0.8s
#   - LLM received correct prompt
#   - Response quality metrics
#   - Any errors that occurred
```

---

## Quick Comparison

| Aspect | LangChain | LangGraph | LangSmith |
|--------|-----------|-----------|-----------|
| **Purpose** | Build LLM components | Orchestrate workflows | Monitor & debug |
| **Use Case** | Basic LLM calls | Complex decision flows | Track performance |
| **Example** | Chat prompt + LLM | If-else routing | Dashboard view |
| **Output** | Response text | Workflow result | Metrics & insights |

---

## Summary

- **LangChain**: The toolkit - provides building blocks (LLMs, prompts, memory)
- **LangGraph**: The conductor - orchestrates complex workflows with decisions
- **LangSmith**: The observatory - monitors everything and helps you improve

Together, they form a complete framework for building, orchestrating, and maintaining production-grade LLM applications.

---

## Next Steps

1. **Learn LangChain**: Master prompts, chains, and memory
2. **Build with LangGraph**: Create multi-step workflows
3. **Monitor with LangSmith**: Set up tracing and evaluation
4. **Iterate**: Use insights to improve your application

Happy learning! 🚀
