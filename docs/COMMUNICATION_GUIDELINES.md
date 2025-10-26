# Communication & Workflow

Ask clarifying questions for ambiguous or complex requests using the `AskUserQuestion` tool (max 4 options per question). Examples:

- **Ambiguous:** "optimize this" → Ask: speed, memory, or readability?
- **Complex:** "add authentication" → Ask: JWT, OAuth, session duration?
- **Continuation:** "continue" → Ask: which specific task?

Skip questions for trivial commands like "run tests" or "format code".

Never create or amend commits without explicit permission—the user manages git operations.
