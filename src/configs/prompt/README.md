# File Naming Conventions

The name of the file is on the format `<with/without role>_<aggregation_type>.yaml`:


**Roles**:

- `woR` (without role) -> the agent is not aware of it's own role
- `wR` (with role) -> the agent is aware of it's role
  
**Aggregation:**

- `C` (counts) -> just provides the counts of the neighbours that agree/disagree/unsure (e.g., 4 of your friends)
- `L` (list of neighbours) -> provides a list of neighbours that agree/disagree/unsure (Biomedic, Engineer agree ....)

**Example**:

`wOR_L` config includes the prompt that does not provide the role for the agent, but does provide the roles of it's neighbours