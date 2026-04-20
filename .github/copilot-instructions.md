## Code Standards

### Required Before Each Commit
- Run `black . && ruff check .` before committing any changes to ensure proper code formatting

### Docstring Style
- Write docstrings using **third-person singular** verbs (e.g., "Returns the value.", "Computes the output shape.", not "Return the value." or "Compute the output shape.")
- Use **`Returns:`** as the section header for return value documentation, never `Return:`
- One-liner docstrings must end with a period

### Imports
- Avoid using aliases when importing as much as possible

### Private members
- Do not make everything private.

### Exception
- Do not use try/except when something may fail. Use a proper test to avoid this situation.
