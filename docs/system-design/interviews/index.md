# System Design Interview Preparation

Complete guide to acing system design interviews at top tech companies.

---

## What to Expect

### Interview Format

**Duration:** 45-60 minutes

**Structure:**
- 5-10 min: Requirements gathering and clarification
- 10-15 min: High-level design
- 20-30 min: Deep dive into specific components
- 5-10 min: Wrap up, bottlenecks, trade-offs

**What interviewers evaluate:**
- Technical knowledge (databases, caching, scaling)
- Problem-solving approach (structured thinking)
- Communication (can you explain clearly?)
- Trade-off analysis (understanding pros/cons)
- Practical experience (have you built systems?)

### Common Interview Questions by Level

**Easy (entry-level, new grads):**
- Design URL shortener, Pastebin, Rate limiter, Key-value store

**Medium (mid-level, 2-5 years):**
- Design Twitter feed, WhatsApp, Uber, YouTube, Web crawler

**Hard (senior, staff+):**
- Design Netflix, Payment system, Google search, Ad system, Stock exchange

---

## Study Resources

### Core Interview Skills

| Resource | What You'll Learn | Time |
|----------|------------------|------|
| **[The 4-Step Framework](framework.md)** | Structured approach for any design problem | 30 min |
| **[Practice Problems](practice-problems.md)** | 15+ problems from easy to hard | Ongoing |
| **[Communication Tips](communication.md)** | How to present your design effectively | 20 min |
| **[Calculations Guide](calculations.md)** | Back-of-envelope estimation techniques | 30 min |
| **[Common Mistakes](common-mistakes.md)** | What to avoid and how to fix it | 15 min |

### Preparation Roadmap

=== "4-Week Intensive"

    **Timeline:** Preparing for interview in 1 month

    **Week 1: Core Concepts + Framework**
    - Study: [The 4-Step Framework](framework.md)
    - Study: [Fundamentals](../fundamentals/index.md), [Databases](../databases/index.md), [Caching](../caching/index.md)
    - Practice: 3-4 easy problems
    - Focus: [Communication](communication.md) basics
    - Time: 10-12 hours

    **Week 2: Scaling + Practice**
    - Study: [Scalability](../scalability/index.md), [Load Balancing](../load-balancing/index.md), [Messaging](../messaging/index.md)
    - Practice: 4-5 medium problems (Twitter, WhatsApp, Uber)
    - Master: [Calculations](calculations.md)
    - Time: 12-15 hours

    **Week 3: Advanced Concepts**
    - Study: [Distributed Systems](../distributed-systems/index.md), [Consistent Hashing](../consistent-hashing/index.md)
    - Practice: 3-4 medium problems
    - Review: [Common Mistakes](common-mistakes.md)
    - Time: 12-15 hours

    **Week 4: Mock Interviews**
    - Practice: 4-5 timed mock interviews (45 min each)
    - Review recordings, identify weak areas
    - Refine communication and trade-off discussions
    - Time: 10-12 hours

=== "8-Week Comprehensive"

    **Timeline:** Thorough preparation

    Follow 4-week plan but:
    - Spend 2 weeks on each phase
    - Add 3-4 more practice problems per week
    - Study [case studies](../case-studies/index.md) deeply
    - Read company engineering blogs
    - Do weekly mock interviews starting week 3

=== "1-Week Crash Course"

    **Timeline:** Last-minute prep (not recommended, but realistic)

    **Days 1-2:** Learn [4-step framework](framework.md), do 2 easy problems

    **Days 3-4:** Study [calculations](calculations.md), do 2 medium problems (Twitter, Uber)

    **Days 5-6:** Practice [communication](communication.md), do 2 more medium problems

    **Day 7:** 2-3 mock interviews, review [common mistakes](common-mistakes.md)

---

## Essential Topics

### Must-Know for Interviews

Priority topics to study before your interview:

| Priority | Topic | Why Critical | Time |
|----------|-------|--------------|------|
| 🔴 Critical | [Databases](../databases/index.md) | 80% of designs need storage | 1 week |
| 🔴 Critical | [Caching](../caching/index.md) | First optimization technique | 3 days |
| 🔴 Critical | [Scalability](../scalability/index.md) | Core interview concept | 1 week |
| 🟡 Important | [Load Balancing](../load-balancing/index.md) | Common in all systems | 2 days |
| 🟡 Important | [Distributed Systems](../distributed-systems/index.md) | Medium/hard problems | 1 week |
| 🟢 Useful | [Messaging](../messaging/index.md) | Async architectures | 3 days |
| 🟢 Useful | [Consistent Hashing](../consistent-hashing/index.md) | Advanced scaling | 2 days |

---

## Quick Links

### By Interview Stage

**Before the interview:**
- Review: [Essential Topics](#essential-topics) above
- Refresh: [The Framework](framework.md)
- Practice: Pick 2-3 problems from [Practice Problems](practice-problems.md)

**During the interview:**
- Follow: [The 4-Step Framework](framework.md)
- Remember: [Communication Tips](communication.md)
- Use: [Calculation techniques](calculations.md)

**After the interview:**
- Review: [Common Mistakes](common-mistakes.md) - what could you improve?
- Reflect: What went well? What needs work?

---

## Study Plan Recommendations

### Minimum Viable Prep (1 week)
- Master [the framework](framework.md)
- Do 5 problems (2 easy, 3 medium)
- Practice 2 mock interviews

### Good Preparation (4 weeks)
- Follow the [4-week plan](#4-week-intensive) above
- Do 12-15 problems across all difficulty levels
- Practice 4-5 mock interviews

### Excellent Preparation (8 weeks)
- Follow the [8-week plan](#8-week-comprehensive) above
- Do 20-25 problems
- Weekly mock interviews
- Study real system architectures

---

## Additional Resources

### Books
- **System Design Interview** by Alex Xu (best for interviews)
- **Designing Data-Intensive Applications** by Martin Kleppmann (deep knowledge)

### Practice Platforms
- [LeetCode System Design](https://leetcode.com/discuss/interview-question/system-design)
- [Pramp](https://www.pramp.com/) - Free mock interviews
- [interviewing.io](https://interviewing.io/) - Practice with engineers

### Company Blogs
Learn from real systems:
- [Netflix Tech Blog](https://netflixtechblog.com/)
- [Uber Engineering](https://eng.uber.com/)
- [Twitter Engineering](https://blog.twitter.com/engineering)
- [LinkedIn Engineering](https://engineering.linkedin.com/)

---

## Next Steps

**Start here:**

1. **Learn the approach** → Read [The 4-Step Framework](framework.md)
2. **Try a problem** → Pick one from [Practice Problems](practice-problems.md)
3. **Improve communication** → Study [Communication Tips](communication.md)
4. **Master estimation** → Practice [Calculations](calculations.md)
5. **Avoid pitfalls** → Review [Common Mistakes](common-mistakes.md)

**Questions?** Everything you need is linked above. Start with the framework and practice problems.

---

[← Back to System Design](../index.md) | [Learning Path](../learning-path.md) | [Case Studies](../case-studies/index.md)
