---
title: Copilot Feature Development Pattern
---

## Introduction

This document provides a guide to effectively using the **Copilot Feature Development Pattern**. This pattern is designed to streamline the collaborative process between software engineers and GitHub Copilot for feature development. It establishes a structured approach to ensure clarity, consistency, and efficient progress tracking from feature conception through implementation.

The core of this pattern is documented in the `feature-development-pattern.md` file. This page serves as an accessible reference within our documentation site, summarizing key aspects and providing actionable guidance.

## The Core Pattern Document

For a comprehensive understanding of the methodology, roles, and detailed steps involved, please refer to the main pattern document hosted in our repository:

[View the Copilot Feature Development Pattern](https://github.com/nvgr-GenAI/my-learning/blob/master/docs/templates/feature-development-pattern.md)

This document outlines:

- The roles and responsibilities of the Software Engineer and Copilot.
- The standardized documentation structure (e.g., `intro.md`, `implementation-plan.md`, `implementation-progress.md`).
- The iterative workflow for planning, implementing, and reviewing milestones.

Further sections on this page will elaborate on specific aspects such as folder structure, example prompts, and the advantages of adopting this pattern.

## Standardized Folder Structure

To maintain consistency and clarity across projects utilizing the Copilot Feature Development Pattern, a specific folder structure is prescribed. All documentation related to a feature being developed with AI assistance should reside within the `.copilot/` directory at the root of the project.

The main subdirectories and key files are:

- **`.copilot/feat/`**: This is where all feature-specific documentation lives.
  - Each feature gets its own subdirectory: `US[ticket-number]-[feature-name]/`
    - `intro.md`: Feature introduction, goals, context.
    - `implementation-plan.md`: Detailed milestones for the feature.
    - `implementation-progress.md`: Tracks progress against milestones.
- **`.copilot/deps/`**: This directory stores markdown documentation related to project dependencies, integrations, and external interfaces that Copilot might need to understand for feature development. For example, API contracts or documentation for a third-party library would go here.
- **`.copilot/lessons-learned.md`**: This crucial file, typically located at the root of the `.copilot` directory, captures key insights, challenges overcome, and solutions discovered during the implementation of features. It serves as an evolving knowledge base. Regularly updating this file helps refine the development process and provides valuable context for Copilot and team members on future tasks. For instance, if a particular approach to API integration proved problematic and a better way was found, that lesson would be documented here.

Below is a visual representation of this structure:

![Feature Development Pattern Folder Structure](/img/feature-development-pattern.png)

This structured approach ensures that all relevant information for a feature is centralized and follows a predictable pattern, making it easier for team members (and Copilot) to navigate and understand the development lifecycle of each feature.

## Interaction Workflow & Prompts

This section summarizes the key prompts, roles, and workflow between the Software Engineer and Copilot, based on the usage guidelines found in the core pattern document. It includes additional clarifications for a comprehensive understanding of how to apply the pattern.

### 1. Starting a New Feature

- **Software Engineer:**
  - Clearly define the feature requirements, scope, and ticket number.
  - Initiate by prompting Copilot:

    ```plaintext
    Please review the feature development pattern document. We are going to start working on a new feature:
    [feature-number]-[feature-name]
    ```

- **Copilot:**
  - Creates the directory `.copilot/feat/US[ticket-number]-[feature-name]/`.
  - Creates blank `intro.md`, `implementation-plan.md`, and `implementation-progress.md` files within this directory.

### 2. Populating Feature Introduction (`intro.md`)

- **Software Engineer:**
  - Populates the `intro.md` file with:
    - **Introduction:** Brief overview of the feature.
    - **Goal:** What the feature aims to achieve.
    - **Context:** Relevant background, dependencies, or existing systems information.
    - **Implementation Guidelines:** High-level technical considerations, constraints, or preferred approaches.

### 3. Requesting and Refining the Implementation Plan

- **Software Engineer:**
  - Once `intro.md` is complete, prompt Copilot:

    ```plaintext
    I have filled out the intro doc. I am ready to start planning.
    ```

- **Copilot:**
  - Reviews `intro.md` and any relevant documents in `.copilot/deps/`.
  - Drafts an `implementation-plan.md`, breaking the feature into discrete, testable milestones with clear descriptions and expected outcomes.
- **Software Engineer:**
  - Thoroughly review the `implementation-plan.md`.
  - Provide feedback directly in the chat for revisions. For example:

    ```plaintext
    Feedback:
    - Milestone 1 seems too broad, can we split it?
    - Please add a milestone for setting up the database schema.
    ```

  - Iterate with Copilot until the plan is satisfactory.

### 4. Milestone Implementation Cycle

- **Software Engineer: Approve and Start a Milestone**
  - Once satisfied with the `implementation-plan.md`, prompt Copilot for the first (or next) milestone:

    ```plaintext
    I have reviewed the implementation plan. It looks good. Let's start the first milestone.
    ```

    (For subsequent milestones: `The [previous milestone name] milestone looks good. Let's move on to implementing milestone [current milestone name/number].`)
- **Copilot: Implement the Milestone**
  - Writes code and makes necessary changes for the approved milestone.
  - May ask clarifying questions.
  - Runs builds and automated tests (if applicable and configured) to verify changes.
- **Software Engineer: Review and Validate Milestone**
  - Review code changes made by Copilot.
  - Manually test the application/feature component.
  - Provide feedback (e.g., "This logic needs adjustment," "Please add more comments here").
  - This review-feedback-update cycle continues until the Software Engineer is satisfied.
- **Copilot: Update `implementation-progress.md`**
  - Once the Software Engineer confirms a milestone is complete, Copilot updates `implementation-progress.md` by checking off the completed milestone.
- **Software Engineer: Document Learnings (If Any)**
  - If significant insights or challenges were encountered, instruct Copilot:

    ```plaintext
    During Milestone X, we discovered that [specific learning]. Please add this to lessons-learned.md.
    ```

### 5. Feature Completion and Wrap-up

- **Software Engineer: Final Review**
  - Once all milestones are complete, conduct a final review of the entire feature.
- **Copilot: Assist with Documentation Updates**
  - If needed, ask Copilot to help update project-level documentation (e.g., `README.md`, API docs). Example:

    ```plaintext
    All milestones for US[ticket-number]-[feature-name] are complete. Can you help generate a summary of the work done for a pull request description?
    ```

- **Software Engineer: Version Control**
  - The Software Engineer is responsible for managing git commits, pull requests, and merging the feature.

These prompts and workflow steps form the core of the collaborative process with Copilot for feature development.

## Advantages of Using This Pattern

Adopting the Copilot Feature Development Pattern offers several key advantages for individuals and teams:

- **Enhanced Clarity and Consistency:** The standardized folder structure and document templates (`intro.md`, `implementation-plan.md`, `implementation-progress.md`) ensure that feature development is approached consistently across the project. This clarity benefits both human developers and Copilot, leading to a shared understanding of goals and progress.

- **Improved Collaboration between Engineer and AI:** By defining clear roles and interaction points (as outlined in the Key Interaction Prompts), the pattern fosters a more effective and predictable collaboration between the software engineer and Copilot. The engineer guides, reviews, and approves, while Copilot assists with generation, research, and updates.

- **Efficient Progress Tracking:** With `implementation-plan.md` detailing the milestones and `implementation-progress.md` tracking their completion, it becomes straightforward to monitor the status of a feature. This transparency is valuable for the developer, team leads, and stakeholders.

- **Systematic Knowledge Capture and Transfer:** The `lessons-learned.md` file serves as a crucial repository for insights, solutions to challenges, and best practices discovered during development. This not only helps in refining future interactions with Copilot for similar tasks but also acts as a valuable knowledge base for the entire team.

- **Focused and Contextual AI Assistance:** Providing Copilot with structured information through the `intro.md` and `deps/` documentation allows it to generate more relevant, context-aware code and suggestions. The iterative planning and feedback loop ensures Copilot's efforts are aligned with the engineer's expectations.

- **Streamlined Onboarding and Handovers:** When new team members join or when features need to be handed over, the consistent documentation structure significantly reduces the learning curve. All the necessary context, plans, and progress are readily available in a predictable format.

- **Iterative and Agile Approach:** The pattern encourages breaking down features into small, manageable, and testable milestones. This iterative approach aligns well with agile development principles, allowing for regular feedback and course correction.

By implementing this pattern, teams can harness the power of AI tools like GitHub Copilot more effectively, leading to a more organized, efficient, and collaborative development experience.
