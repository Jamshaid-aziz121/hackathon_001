---
id: 1
title: Create Book Constitution
stage: constitution
date: 2026-01-01
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: jamshaid
command: sp.constitution
labels: [constitution,book,ai,robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
project:
  name: AI / Spec-Driven Book on Physical AI & Humanoid Robotics
  platform: Docusaurus → GitHub Pages
  tools:
    - Spec-Kit Plus
    - Claude Code

core_principles:
  - name: Educational Clarity
    description: >
      Content must progress from beginner concepts to advanced implementations
      while remaining understandable to readers with a computer science background.

  - name: Technical Accuracy
    description: >
      All AI, robotics, cloud-native, and agentic system content must be
      technically correct, current, and aligned with real-world systems.

  - name: Practical Outcomes
    description: >
      The book must emphasize hands-on learning through runnable examples,
      simulations, and real implementation patterns.

  - name: Ethical & Responsible AI
    description: >
      Safety, robustness, and responsible AI practices must be explicitly
      addressed, especially in physical and humanoid systems.

standards:
  content:
    - All content must be original and source-traceable
    - All factual and technical claims must be verifiable
    - Code examples must run successfully or be clearly marked as pseudocode

  robotics:
    - Concepts must align with ROS2
    - URDF must be used where applicable
    - Real-world control systems and simulation environments must be respected

  agentic_ai:
    - Content must reflect current production practices
    - Experimental or speculative patterns must be clearly labeled

  citations:
    allowed_sources:
      - Official documentation
      - Peer-reviewed research
      - Reputable technical publications
    prohibited_sources:
      - Marketing material
      - Unverifiable blogs
      - Unsourced claims

  tone:
    style: Mentor → student
    characteristics:
      - Respectful
      - Direct
      - Instructional

structure:
  chapter_requirements:
    - Start from a clear specification
    - Learning objectives
    - Concept explanation
    - Practical examples
    - Step-by-step implementation
    - Final exercise or challenge

  technical_modules:
    must_include:
      - Inputs
      - Outputs
      - System architecture
      - Code implementation
      - Failure modes
      - Safety considerations

verification:
  - All non-obvious claims must be cited
  - Contradictory research findings must be acknowledged
  - Claims must be grounded in authoritative sources

accessibility:
  - Images and figures must include descriptive alt text
  - Diagrams must be labeled and technically accurate
  - Markdown must be fully compatible with Docusaurus
  - Navigation must follow Docusaurus best practices

constraints:
  word_count: 20000–35000
  demos:
    - At least one working demo (simulation acceptable)
  format: Markdown (Docusaurus-compatible)
  deployment: Must deploy cleanly to GitHub Pages

success_criteria:
  - A student can build a working AI or robotics system by following the book
  - All examples are reproducible
  - All claims are verifiable
  - The Docusaurus build passes without errors
  - Theory and practice are balanced
  - Safety and limitations are clearly documented

priority_order:
  - Accuracy
  - Practicality
  - Clarity
  - Speed

governance:
  authority: This constitution supersedes all informal practices
  amendments:
    require:
      - Written justification
      - Approval record
      - Migration or update plan
  compliance:
    - All contributions must be reviewed against this constitution
```

## Response snapshot

I have successfully updated the project constitution based on your input. Here's the summary:

**New version**: 1.0.0 (initial creation)
**Bump rationale**: This is the first version of the constitution, created based on the project requirements for an AI/Spec-Driven Book on Physical AI & Humanoid Robotics.

**Files updated**:
- `.specify/memory/constitution.md` (created/updated with the new constitution)

**Constitution now includes**:
- 6 core principles: Educational Clarity, Technical Accuracy, Practical Outcomes, Ethical & Responsible AI, Content Verification, and Technical Standards
- Standards and Requirements section covering content, robotics, agentic AI, citations, and accessibility
- Development Structure section with chapter requirements and technical modules
- Governance section with amendment procedures

**No files flagged for manual follow-up** - all template files were checked and are compatible with the new constitution.

**Suggested commit message**: `docs: create constitution v1.0.0 for AI/Spec-Driven Book on Physical AI & Humanoid Robotics`

The constitution aligns with all your specified requirements and follows the mentor-to-student tone with a focus on technical accuracy, educational clarity, and practical outcomes for the book on Physical AI & Humanoid Robotics.

## Outcome

- ✅ Impact: Created initial project constitution with 6 core principles and detailed standards for the AI/Spec-Driven Book on Physical AI & Humanoid Robotics
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with comprehensive constitution
- 🔁 Next prompts: Ready for feature specification based on this constitution
- 🧠 Reflection: Process followed the template structure and mapped all user inputs to appropriate sections

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS - All constitution requirements met
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin feature specification based on constitution