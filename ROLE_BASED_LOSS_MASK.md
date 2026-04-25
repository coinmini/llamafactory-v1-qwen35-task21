# Role-Based Loss Mask for ShareGPT Format

## Background

In SFT (Supervised Fine-Tuning), not all tokens should contribute to the loss. Only tokens the model needs to learn to generate should be included. User inputs, system prompts, and tool responses should be masked (`IGNORE_INDEX = -100`).

## Problem: Position-Based Pairing Breaks Agent Conversations

The original LlamaFactory uses **odd/even position** to decide which messages are prompts (masked) and which are responses (loss computed):

```
Position 0 (even) -> prompt (masked)     must be user/observation
Position 1 (odd)  -> response (loss)     must be assistant/function_call
Position 2 (even) -> prompt (masked)     must be user/observation
Position 3 (odd)  -> response (loss)     must be assistant/function_call
```

This enforces strict alternating order. Any data that doesn't conform is **dropped**.

Real agent conversations with tool calling don't follow this pattern:

```
user          -> "Check weather in Beijing and Shanghai"
assistant     -> "Sure, let me look that up"
function_call -> get_weather(location="Beijing")
function_call -> get_weather(location="Shanghai")   <- consecutive function_calls
observation   -> {"temp": 25, "weather": "sunny"}
observation   -> {"temp": 22, "weather": "cloudy"}  <- consecutive observations
assistant     -> "Beijing 25C sunny, Shanghai 22C cloudy"
```

The original implementation drops this data because consecutive `function_call` violates the alternating rule.

## Solution: Role-Based Loss Mask

Instead of checking position, use the **role** to decide the loss mask:

| Role | Loss Mask | Reason |
|------|-----------|--------|
| `user` | Masked (no loss) | User input, model doesn't need to learn this |
| `observation` | Masked (no loss) | Tool return values, external data |
| `assistant` | **Compute loss** | Model needs to learn how to respond |
| `function_call` | **Compute loss** | Model needs to learn how to call tools |

**Core principle: the model learns "what to say" and "what tools to call" -> compute loss; external inputs -> masked.**

## Implementation

Three files are modified:

### 1. converter.py - Remove Alternating Validation

```python
# Before: position must alternate
odd_tags = (user_tag, observation_tag)
even_tags = (assistant_tag, function_tag)
accept_tags = (odd_tags, even_tags)
if message[role_tag] not in accept_tags[turn_idx % 2]:  # odd/even check
    broken_data = True

# After: only check role is valid
all_tags = (user_tag, assistant_tag, observation_tag, function_tag)
if message[role_tag] not in all_tags:
    broken_data = True
```

Also relaxed message count validation:

```python
# Before: enforced even/odd message count
if (not ranking and len(aligned_messages) % 2 != 0) or (ranking and len(aligned_messages) % 2 == 0):
    broken_data = True

# After: only reject empty
if len(aligned_messages) == 0:
    broken_data = True
```

### 2. template.py - Dynamic Role-Based Pairing

Both `Template.encode_multiturn` and `ReasoningTemplate.encode_multiturn` are updated.

```python
# Before: fixed stride-2 pairing
return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]

# After: role-based accumulation
pairs = []
source_ids = []
for i, message in enumerate(messages):
    if message["role"] in (Role.USER, Role.OBSERVATION):
        source_ids += encoded_messages[i]       # accumulate as masked source
    else:  # Role.ASSISTANT or Role.FUNCTION
        pairs.append((source_ids, encoded_messages[i]))  # pair: (masked, loss)
        source_ids = []
return pairs
```

### 3. supervised.py - Relax Prompt Length Validation

```python
# Before: prompt length must be odd
if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:

# After: only check response count
if len(examples["_response"][i]) != 1:
```

## Examples

### Simple Conversation (unchanged behavior)

```
user -> assistant
Pairs: [(user_ids, assistant_ids)]
         masked     loss
```

### Single Tool Call

```
user -> function_call -> observation -> assistant
Pairs: [(user_ids, fc_ids), (obs_ids, assistant_ids)]
         masked   loss      masked    loss
```

### Parallel Tool Calls

```
user -> assistant -> fc -> fc -> obs -> obs -> assistant
Pairs: [(user_ids, asst_ids), ([], fc_ids), ([], fc_ids), (obs+obs_ids, asst_ids)]
         masked    loss      empty loss    empty loss      masked       loss
```

### Multi-Step Agent Loop

```
user -> asst -> fc -> obs -> asst -> fc -> obs -> assistant
Pairs: [(user_ids, asst_ids), ([], fc_ids), (obs_ids, asst_ids), ([], fc_ids), (obs_ids, asst_ids)]
         masked    loss      empty loss    masked    loss       empty loss     masked    loss
```

## Compatibility

- For standard `user -> assistant` conversations, behavior is **identical** to the original implementation.
- Only affects conversations with `function_call` / `observation` roles in non-alternating patterns.
- Works with both `Template` and `ReasoningTemplate` (Qwen3.5 thinking mode).
