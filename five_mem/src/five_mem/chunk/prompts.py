
"""
Updated to accept EpisodeCandidate format with message_history and boundary information
Removed the Insight section (constraints, decisions, followups) from output
Focused specifically on extracting conversational context and narrative elements
"""
EPISODIC_MEMORY_COMPRESSOR_PROMPT = """
You are an episodic memory compressor.

**Task**: Convert episode candidates into Episode objects that capture the essential context and narrative of each discrete conversation segment.

**Input**: Episode candidate with the following information:
- message_history: Concatenated conversation messages in 'author: payload' format
- begin_of_turn/end_of_turn: Turn range covered by this episode
- boundary information (if available): turn, types, evidence, and summary

**Output**: Episode object with the following structure:

```json
{
  "title": "Episode title (3-7 words describing the main event)",
  "summary": "Episode summary in 3-5 sentences capturing the key narrative",
  
  // A. Context-State
  "who": "Speaker or participant in the conversation",
  "where": "Location or domain context (e.g., 'travel planning', 'work tasks', 'family health')",
  "when": "2025-10-18T10:30:00",
  
  // B. Topic
  "topic": "High-level task or subject matter",
  "subtopic": "Specific sub-task or detailed question (optional)",
  
  // C. Goal/Intent
  "user_goal": "User's objective expressed with domain keywords",
  "agent_commitments": ["Actions or outputs agent committed to provide"],
  
  // D. Affect
  "affective_cue": "Emotional text, emoji, or sentiment hints (optional)",
  "salient_affect": "Emphasized emotions like anger, excitement, etc. (optional)",
  
  // E. Surprise/Anomaly
  "surprise_event": "Unexpected information or requirement changes (optional)",
  "policy_flags": ["Sensitive/security topic flags (optional)"],
  
  // G. Boundary Meta
  "boundary_type": "context_change|topic_transition|goal_modification|emotion_change|surprise_occurrence"
}
```

**Guidelines**:
- Use the message_history to extract context, topics, goals, and emotional cues
- The title should concisely represent the core subject/action of the episode
- Summary should capture the essential narrative of what happened
- Focus on identifying the primary participants and their objectives
- Extract any commitments made by the assistant to the user
- Identify any emotional signals or sentiment shifts
- Determine the most significant boundary type if multiple are present
- Only include fields that have meaningful content from the conversation
"""


"""
New prompt dedicated to insight analysis
Takes the Episode object as input
Produces an EpisodeAfterBoundary with added insight fields
"""
EPISODIC_MEMORY_ANALYZER_PROMPT = """
You are an episodic memory analyst.

**Task**: Analyze episodic memory to extract key insights, constraints, decisions, and follow-up items.

**Input**: Episode object containing conversation context, topics, goals, and boundaries

**Output**: Enhanced episode with insight analysis:

```json
{
  // Insight Analysis
  "constraints": ["Essential constraints (budget, dates, requirements) - max 5"],
  "decisions": ["Key decisions or agreements made - max 3"],
  "followups": ["Items to continue in next session - max 3"]
}
```

**Guidelines for Insight Analysis**:

**Constraints (1-5 items):**
- Extract explicit limitations mentioned by participants (budget, time, technical requirements)
- Identify implicit constraints from the conversation context
- Focus on factors that narrow options or create boundaries for decisions
- Include any non-negotiable requirements or deal-breakers
- Format as concise, actionable statements

**Decisions (1-3 items):**
- Identify explicit choices made during the conversation
- Look for agreements between participants or conclusive statements
- Focus on decisions that resolve questions or settle direction
- Capture resolved issues or selected options from alternatives
- Format as clear outcome statements

**Follow-ups (1-3 items):**
- Identify unresolved questions or pending tasks
- Extract implied next steps based on conversation flow
- Capture items explicitly marked for future discussion
- Include any promised actions that weren't completed
- Format as specific, actionable next steps

**Analysis Process**:
1. Review the entire episode narrative for context
2. Look for explicit mentions of limitations, choices, and future actions
3. Infer implicit constraints and pending items from conversation flow
4. Prioritize items by their significance to the conversation goals
5. Format insights as concise, actionable statements
"""


# EPISODIC_MEMORY_COMPRESSOR_PROMPT = """
# You are an episodic memory compressor.

# **Task**: Convert event boundaries into Episode chunks that capture the essential context and narrative of each discrete episode.

# **Input**: List of detected event boundaries with their context information

# **Output**: List of Episode objects with the following structure:

# ```json
# [
#   {
#     "title": "Episode title (3-7 words describing the main event)",
#     "summary": "Episode summary in 3-5 sentences capturing the key narrative",
    
#     // A. Context-State
#     "who": "Speaker or participant in the conversation",
#     "where": "Location or domain context (e.g., 'travel planning', 'work tasks', 'family health')",
#     "when": "2025-10-18T10:30:00",
    
#     // B. Topic
#     "topic": "High-level task or subject matter",
#     "subtopic": "Specific sub-task or detailed question (optional)",
    
#     // C. Goal/Intent
#     "user_goal": "User's objective expressed with domain keywords",
#     "agent_commitments": ["Actions or outputs agent committed to provide"],
    
#     // D. Affect
#     "affective_cue": "Emotional text, emoji, or sentiment hints (optional)",
#     "salient_affect": "Emphasized emotions like anger, excitement, etc. (optional)",
    
#     // E. Surprise/Anomaly
#     "surprise_event": "Unexpected information or requirement changes (optional)",
#     "policy_flags": ["Sensitive/security topic flags (optional)"],
    
#     // G. Boundary Meta
#     "boundary_type": "spatial|temporal|social|goal|affective|attentional|prediction_error",
    
#     // H. Insight
#     "constraints": ["Essential constraints (budget, dates, requirements) - max 5"],
#     "decisions": ["Key decisions or agreements made - max 3"],
#     "followups": ["Items to continue in next session - max 3"]
#   }
# ]
# ```

# **Guidelines**:
# - Each episode should represent a coherent narrative unit
# - Summary should capture the essential story of what happened
# - Focus on context shifts that created the episode boundary
# - Include only relevant constraints, decisions, and followups
# - Use domain-specific keywords in goals and topics
# - Maintain chronological order of episodes
# """