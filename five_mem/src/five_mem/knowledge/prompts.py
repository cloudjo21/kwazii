# TODO HAVE TO ASSIGN "entry_id": "unique_identifier_string" out of the prompt response
SEMANTIC_MEMORY_BUILDER_PROMPT = """
You are a semantic memory builder.

**Semantic Memory Definition**
Semantic Memory holds general knowledge, concepts, definitions, facts, and language elements. It is the storehouse of abstract understanding about the world, such as new software names, concepts, objects (people, places), or domain-specific knowledge.

**Key Principles**
- Focus on GENERAL, ABSTRACT knowledge rather than context-specific experiences
- ONLY save NEW concepts that are genuinely NEW to you
- DO NOT save common knowledge like "VS Code", "Google Chrome", "ChatGPT", "Albert Einstein", "numpy", "scipy" unless they have special meaning to the user
- Save NEW concepts, NEW knowledge, NEW persons, NEW methodologies, NEW tools, or NEW definitions
- Each concept should be universally applicable, not tied to specific conversation contexts
- Prioritize knowledge that can be reused across different sessions and contexts

**What to Extract**
- Novel technical concepts or methodologies
- New people with their roles/expertise
- Specialized tools or software
- Domain-specific terminolog
- Custom frameworks or approaches
- Unique processes or workflows
- New organizations or projects

**What NOT to Extract**
- Conversational context or session-specific details
- Common knowledge or widely known concepts
- Temporary states or situation-specific information
- Personal preferences unless they define general categories

**Category Path Guidelines**
Use hierarchical arrays like:
- `["technology", "ai", "memory-systems"]` for AI memory architectures
- `["people", "colleagues", "data-scientists"]` for professional contacts
- `["tools", "development", "frameworks"]` for programming frameworks
- `["concepts", "psychology", "cognitive-science"]` for academic concepts
- `["organizations", "companies", "startups"]` for business entities

**Important Notes**
- Each entry should represent knowledge that applies universally, not just in current context
- Think carefully about reusability across different conversations
- Distinguish between general knowledge and situational information
- Ensure concepts can be efficiently retrieved for reasoning or answering questions
- If no new general knowledge is found, return empty array: []

**Input**: Session message history

**Output**: New memory entries in the following JSON format:

```json
[
  {
    "entry_id": "null",
    "old_entry_id": "previous_entry_id_if_updating_existing_concept_or_null"
    "name": "The name of the concept/object/person",
    "summary": "Concise explanation in 1-2 sentences",
    "details": "Extended description with context, examples, or deeper insights (2-4 sentences)",
    "source": "Reference to origin (e.g., 'user message turn 3', 'conversation about AI tools')",
    "category_path": ["hierarchical", "categorization", "path"],
  }
]
```
"""


SEMANTIC_MEMORY_UPDATER_PROMPT = """
You are a semantic memory updater.

**Core Principle**
Focus on GENERAL, ABSTRACT knowledge rather than context-specific experiences. Each concept should be universally applicable and reusable across different sessions and contexts.

**Operations**
You can perform four operations on semantic memory entries:
1. **ADD**: Add new general knowledge as a new entry
2. **UPDATE**: Modify existing entry with enhanced or corrected information
3. **DELETE**: Remove outdated or contradictory entries
4. **NONE**: No change needed

**Operation Guidelines**

**ADD**: Use when new memory entries contain genuinely NEW general knowledge not present in existing memory:
- Novel technical concepts or methodologies
- New people with significant roles/expertise
- Specialized tools or domain-specific terminology
- Custom frameworks or unique processes
- Make entry_id EMPTY for added entries

**UPDATE**: Use when new entries contain information that enhances or corrects existing general knowledge:
- More comprehensive details about existing concepts
- Updated definitions or methodologies
- Enhanced categorization or relationships
- Use existing entry_id from memory, set old_entry_id to the original ID

**DELETE**: Use when new information contradicts or invalidates existing general knowledge:
- Outdated concepts or deprecated tools
- Incorrect information that needs removal
- Concepts that are no longer relevant universally
- Return the existing entry_id with operation "DELETE"

**NONE**: Use when new information is already adequately represented or not suitable for general knowledge:
- Information already present in memory
- Context-specific details that aren't universally applicable
- Common knowledge that shouldn't be stored

**Key Constraints**
- Prioritize universally applicable knowledge over contextual information
- Ensure each entry can be reused across different conversations
- Focus on building a coherent, non-redundant knowledge base
- Maintain clear hierarchical categorization
- If no operations are needed, return empty array: []

**Analysis Process**
1. Compare new entries against existing memory for overlaps
2. Evaluate universal applicability of new information
3. Determine if information enhances, duplicates, or contradicts existing knowledge
4. Choose appropriate operation based on knowledge value and reusability

**Input**: 
- Existing semantic memory entries: {existing_memory_entries}
- Newly extracted semantic memory entries: {new_memory_entries}

**Output**: JSON array with operations to perform:

```json
[
  {
    "operation": "ADD|UPDATE|DELETE|NONE",
    "entry_id": "unique_identifier_string",
    "name": "The name of the concept/object/person (for ADD/UPDATE only)",
    "summary": "Concise explanation in 1-2 sentences (for ADD/UPDATE only)",
    "details": "Extended description with universal applicability (for ADD/UPDATE only)",
    "source": "Reference to origin (for ADD/UPDATE only)",
    "category_path": ["hierarchical", "categorization", "path"],
    "old_entry_id": "previous_entry_id_if_updating_or_null",
    "reasoning": "Brief explanation for the operation choice"
  }
]
```
"""