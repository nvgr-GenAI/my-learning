# Feature Development Pattern

This document describes the pattern used by the software engineer and Copilot while working on a feature.

## Context

A software developer is going to be working in concert with Copilot to implement a feature in this software project.

## Documentation

- Base documentation folder for feature development: `.copilot/feat`
- Dependencies folder: `.copilot/deps`
- Documentation format: Markdown
- Key documents:
  - `lessons-learned.md`: Captures key lessons learned during feature implementation. It is where we will store new information which is discovered during the course of implementation. Lessons learned should only include new lessons.
  - `.copilot/deps/`: Contains documentation on dependencies, integrations, and interfaces.
- Feature documentation structure: `.copilot/feat/US[ticket-number]-[feature-name]/`
  - `intro.md`: Feature introduction and context.
    - Create place holder sections like Introduction, Goal, Context, Implementation Guidelines, Dependcies
  - `implementation-plan.md`: Detailed implementation milestones.
    - Each milestone will include a section describing the expected outcome.
    - The implementation plan is high level.
    - The implementation plan should not contain code, technical specifications, or directory structures. Prose only.
    - Each implementation milestone should include only a single discrete testable change. Break complex implementations into multiple smaller milestones, where each milestone can be independently verified.
  - `implementation-progress.md`: Track progress of implementation.
    - Checklists make the progress report easy to understand.

## Software Developer and Copilot Working Together

The feature will be implemented by Copilot with direction and feedback given by the software engineer. Focus on understanding the work to be done before getting to the code.

## Roles and Responsibilities

### Software Engineer's Role

The software engineer plays an important role in the implementation pattern. The software engineer:

- Provides feedback on changes made by Copilot.
- Validates milestone completion.
- Provides context and makes git commits.
- Gets documentation for Copilot.
- Decides when to continue to the next activity.

### Copilot's Role

Copilot manages the code. This pattern is single writer and Copilot is the writer.

- Writes the code and keeps documents up to date.
- Asks the software engineer for input.

## General Pattern

1. Copilot creates the docs(`intro.md`, `implementation-paln.md`, `implementation-progress.md`) for the new feature.
2. The software engineer writes the intro file.
3. The software engineer can take copilot help to refine the intro file.
4. Copilot checks its understanding of the intro file with the software engineer until consensus is reached.
5. Copilot creates an implementation plan document.
6. The software engineer reviews the implementation plan and provides feedback.
7. The software engineer and Copilot work together to refine the implementation plan until the software engineer is satisfied with the plan.
8. Copilot waits for explicit approval from the software engineer before starting the implementation of any milestone.
9. Copilot begins implementation work, starting with the first milestone.
10. Copilot runs builds and tests to verify the correctness of its changes.
11. After Copilot finishes implementation, the software engineer checks the code and application.
12. The software engineer provides feedback to Copilot.
13. Copilot makes updates to the implementation based on the software engineer's feedback.
14. The software engineer and Copilot continue the milestone refinement cycle until the software engineer is satisfied.
15. The milestone is considered complete and Copilot updates the implementation progress and lessons learned docs.
16. At the software engineer's direction, Copilot begins work on the next milestone.
17. This pattern continues until all milestones are complete.
18. The software engineer can ask copilot to generate pullrequest summary after completion of a milestone or entire feature.
19. Copilot will generate pullrequest summary by taking pr template reference or create one based on best practices.

## MCP Git Integration Best Practices

When using Model Context Protocol (MCP) for Git operations, it's important to follow these best practices to avoid conflicts:

- **Synchronize Before MCP Operations**: Always pull the latest changes from remote before asking Copilot to perform Git operations through MCP.
- **Clear Workflow Separation**: Decide whether a specific feature/change will be managed through local Git commands or MCP, not both simultaneously.
- **Local-First vs. Remote-First**: 
  - Local-First: Make changes locally and push them manually (traditional Git workflow)
  - Remote-First: Let MCP handle the Git operations directly on the remote repository
- **Avoid Parallel Work**: Don't make local changes to files while also having MCP modify the same files remotely.
- **Pull After MCP Operations**: After MCP performs Git operations (branch creation, commits, etc.), always pull changes to synchronize your local repository.
- **Communication Protocol**: Document whether a branch is being managed via MCP or local Git to avoid confusion among team members.
- **Branch Ownership**: Designate certain branches to be managed exclusively via one method (MCP or local Git).

## Refining the Intro File

- After the initial writing of the `intro.md` file, the software engineer can take help from Copilot to refine the file with more technical details.
- This includes adding implementation guidelines, dependencies, and any other relevant technical context.

## Post Milestone Completion

- After each milestone completion, Copilot should ask for user approval to:
  - Commit the changes.
  - Proceed to the next milestone.
  - Create a pull request template.
- The software engineer will make the final decision on the next action.

## Post Feature Completion

- Update project-level docs such as the README file.
- Create a pull request content file in the feature directory if requested.

## Commit Messages

- Use Conventional Commits for all commit messages.

## Review and Approval Process

- Feedback Loop: Thorough review of the implementation plan by the software engineer.
- Approval Confirmation: Explicit confirmation before starting any milestone.

## Documentation Updates

- Ensure all relevant documentation is up-to-date before starting implementation.
- Regularly update the `implementation-progress.md` document.

## Implementation Phase

- Milestone Verification: Run builds and tests to verify changes.
- Feedback Integration: Promptly integrate feedback from the software engineer.

## Communication and Collaboration

- Regular Check-ins: Discuss progress, challenges, and next steps.
- Clear Communication: Promptly seek feedback or approval.

## Best Practices

### Commit Best Practices

- Atomic Commits: Each commit represents a single logical change.
- Descriptive Commit Messages: Explain the "why" behind the change.

### Pull Request Best Practices

- Small and Focused PRs: Focus on a single feature or bug fix.
- Template Usage: Use a consistent pull request template.
- Review Checklist: Verify code quality, tests, and adherence to standards.
- Link Issues: Link the pull request to relevant issues or tasks.

### Naming Conventions

- File Names: Use kebab-case or snake_case for file names.
- Variable and Function Names: Follow language-specific naming conventions.

### Folder Structure Best Practices

- Feature-Based Organization: Group files by feature or module.
- Separation of Concerns: Separate components, styles, and utilities.
- Static Assets: Use a `static/` or `assets/` folder for images and fonts.

## Pre-Commit Hooks and Best Practices

- Adhere to pre-commit hooks defined in the codebase.
- Follow language-specific best practices to maintain code quality.
