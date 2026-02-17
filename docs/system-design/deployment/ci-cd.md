# CI/CD Pipelines

CI/CD -- Continuous Integration and Continuous Delivery/Deployment -- is the practice of automating the journey from a developer's code change to a running system in production. Rather than relying on manual builds, hand-executed test suites, and nerve-wracking weekend deployment rituals, CI/CD encodes the entire release process into a repeatable, auditable pipeline that runs on every commit.

The core insight is simple: if deploying is painful, do it more often. Small, frequent changes are easier to test, easier to debug, and far less likely to cause catastrophic failures than massive quarterly releases. Companies that have embraced this philosophy -- Google, Amazon, Netflix, Etsy -- deploy hundreds or thousands of times per day with lower failure rates than organizations deploying monthly.

```
  Developer        Pipeline                              Production
  ---------        --------                              ----------

  git push  -----> [Build] --> [Test] --> [Scan] --> [Stage] --> [Deploy]
                      |           |          |          |           |
                   Compile     Unit +     Security   Smoke       Canary
                   + Lint    Integration   + Deps    Tests       or Full
                                                                Rollout

                   |<--- Automated (CI) --->|<--- CD: Delivery or Deployment --->|
                   |        ~5 min          |            ~5-15 min               |

  On failure at ANY stage: stop pipeline, notify developer, block merge
```

The pipeline above represents the typical flow. Everything to the left of the staging environment is Continuous Integration -- the automated verification that a change is safe. Everything to the right is Continuous Delivery or Deployment -- getting that verified change into the hands of users. The distinction between those two terms matters, and is covered in detail below.

---

=== "Continuous Integration"

    ## What CI Actually Means

    Continuous Integration is frequently misunderstood as simply "having a build server." In practice, CI is a development discipline: every developer integrates their work into a shared mainline at least once per day, and every integration is verified by automated build and test.

    The key word is "continuous." A team where developers work on feature branches for weeks before merging is not practicing CI, even if they have Jenkins running. Martin Fowler's original definition is explicit: CI requires that developers commit to the mainline frequently -- ideally multiple times per day -- so that integration problems surface within hours rather than weeks.

    **Trunk-based development** is the branching model most aligned with CI. Developers either commit directly to main or use very short-lived feature branches (one to two days maximum). This stands in contrast to GitFlow-style models with long-running develop, release, and feature branches that delay integration.

    Google is perhaps the most extreme example. Their internal system called TAP (Test Automation Platform) runs against a monorepo containing over two billion lines of code. Every commit to Google's single shared trunk triggers affected tests -- roughly 150 million test cases run per day across four million builds. TAP prioritizes tests likely to fail first, providing developers with feedback in minutes even at this enormous scale. The philosophy is that broken tests on the mainline are treated as emergencies, not inconveniences.

    ### The Testing Pyramid

    The testing strategy within CI follows the well-known pyramid shape, where the base is wide (many fast tests) and the top is narrow (few slow tests).

    ```
              /\
             /  \
            /E2E \        ~10% -- browser/API tests, 5-30 min
           /______\
          /        \
         /Integrate \     ~20% -- service + DB tests, 1-5 min
        /____________\
       /              \
      /   Unit Tests   \  ~70% -- pure logic tests, seconds
     /                  \
    /____________________\
    ```

    Unit tests verify individual functions in isolation and run in milliseconds. A mature codebase might have tens of thousands of these. Integration tests verify that components work together -- an API handler correctly reads from a database, a message consumer properly processes events. End-to-end tests drive the entire application through realistic user scenarios, but they are slow, flaky, and expensive to maintain, so teams keep their number small.

    The build-test-lint cycle on every push typically looks like this: the CI server checks out the code, installs dependencies (from cache when possible), runs the linter to catch style and syntax issues, executes the full unit test suite, and builds the artifact. If any step fails, the pipeline stops immediately and the developer is notified. At Spotify, this inner loop is kept under five minutes for most services, because anything longer causes developers to context-switch and stop paying attention to results.

=== "Continuous Delivery vs Deployment"

    ## Two Meanings of CD

    The abbreviation "CD" refers to two related but distinct practices, and confusing them is one of the most common mistakes in system design discussions.

    **Continuous Delivery** means that every change that passes the automated pipeline produces a release candidate that *could* be deployed to production at any time. The deployment itself requires a human decision -- someone clicks a button, approves a pull request, or authorizes a release. The codebase is always in a deployable state, but the organization chooses when to actually deploy.

    **Continuous Deployment** goes one step further: every change that passes all automated checks is deployed to production automatically, with no human gate. If the tests pass, users see the change.

    | Aspect                 | Continuous Delivery         | Continuous Deployment          |
    |------------------------|-----------------------------|--------------------------------|
    | Human approval needed? | Yes -- manual gate          | No -- fully automated          |
    | Deploy frequency       | On-demand (daily to weekly) | Every passing commit           |
    | Risk tolerance         | Lower (manual verification) | Higher (relies on automation)  |
    | Best for               | Regulated, financial, healthcare | SaaS, consumer web, internal tools |
    | Example companies      | Banks, airlines, government | Netflix, Etsy, GitHub          |

    Most organizations practice Continuous Delivery. Continuous Deployment requires extremely high confidence in your test suite and monitoring, because there is no human safety net.

    ### The Deployment Pipeline

    Whether delivery or deployment, the pipeline after CI verification follows a consistent pattern of progressive confidence-building stages.

    ```
    [CI passes] --> [Build Artifact] --> [Deploy to Staging] --> [Smoke Tests]
                         |                                           |
                    Immutable image                           Automated checks
                    tagged with SHA                          against staging env
                         |                                           |
                         v                                           v
                  [Store in Registry] -- on success --> [Deploy to Production]
                                                              |
                                                        Canary or Blue-Green
                                                        with metric monitoring
    ```

    **Approval gates** sit between stages in Continuous Delivery pipelines. These can be manual (a tech lead reviews and approves) or semi-automated (the system checks that staging metrics are healthy for 30 minutes before unlocking the production gate). GitHub uses a system called ChatOps where engineers type a deploy command in Slack, which triggers the production promotion only after staging has been verified.

    **Artifact management** is a critical but often overlooked piece. The artifact built in CI -- a Docker image, a JAR file, a compiled binary -- must be the exact same artifact deployed to staging and production. Rebuilding from source for each environment introduces the risk that the production build differs from what was tested. Companies store these immutable artifacts in registries (Docker Hub, AWS ECR, Artifactory) tagged with the Git commit SHA for full traceability.

=== "Pipeline Design"

    ## Designing an Effective Pipeline

    A well-designed pipeline balances thoroughness with speed. The goal is to catch problems as early and cheaply as possible while keeping the total feedback time short enough that developers stay engaged.

    ### Stage Ordering

    Stages should be ordered from fastest and cheapest to slowest and most expensive. If the linter catches a missing semicolon in 10 seconds, there is no reason to wait for a 5-minute integration test suite to also fail.

    ```
    Stage            Typical Duration    What It Catches
    -----            ----------------    ---------------
    Lint + Format       10-30 sec        Style, syntax, import errors
    Unit Tests           1-3 min         Logic bugs, regressions
    Build / Compile      1-3 min         Compilation errors, type mismatches
    Integration Tests    3-8 min         API contracts, DB interactions
    Security Scan        2-5 min         Known CVEs, vulnerable dependencies
    Deploy to Staging    2-5 min         Configuration issues, env problems
    Smoke Tests          1-3 min         Critical path verification
    Production Deploy    2-10 min        Real-world rollout (canary/blue-green)
    ```

    ### Parallel Execution

    Stages that do not depend on each other should run in parallel. Unit tests, linting, and security scanning can all start simultaneously after checkout. Integration tests might depend on a successful build, but they can run alongside end-to-end tests once the build artifact exists. At LinkedIn, parallelizing their test suites across hundreds of machines reduced pipeline time from 45 minutes to under 10 minutes.

    ### Caching for Speed

    Dependency installation is one of the biggest time sinks in CI. Downloading and installing node_modules, Python virtualenvs, or Maven dependencies on every run wastes minutes. All major CI platforms support caching: store the dependency directory keyed on the lockfile hash, and restore it on subsequent runs. A typical Node.js project goes from a 90-second install to a 5-second cache restore.

    Build layer caching for Docker images follows the same principle. If the Dockerfile layers for OS packages and dependencies have not changed, the CI system reuses cached layers and only rebuilds the application code layer, cutting image build time from minutes to seconds.

    ### Minimal Pipeline Example

    A complete pipeline definition can be surprisingly concise. Here is a GitHub Actions workflow that covers the essential stages:

    ```yaml
    # .github/workflows/ci-cd.yml
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-node@v4
            with: { node-version: 20, cache: npm }
          - run: npm ci && npm run lint && npm test
      deploy:
        needs: test
        if: github.ref == 'refs/heads/main'
        runs-on: ubuntu-latest
        steps:
          - run: echo "Build image, push, deploy to staging, then production"
    ```

    ### GitHub Actions vs GitLab CI

    | Feature              | GitHub Actions               | GitLab CI/CD                  |
    |----------------------|------------------------------|-------------------------------|
    | Config file          | `.github/workflows/*.yml`    | `.gitlab-ci.yml`              |
    | Runner model         | Hosted or self-hosted        | Shared or self-hosted runners |
    | Marketplace          | 20,000+ reusable actions     | Fewer templates, more built-in|
    | Container registry   | GitHub Container Registry    | Built-in per-project registry |
    | Strengths            | Ecosystem, community actions | All-in-one DevOps platform    |

    Both are capable platforms. GitHub Actions dominates open-source projects due to generous free tiers and community actions. GitLab CI appeals to enterprises wanting a single platform for source control, CI/CD, security scanning, and artifact management without third-party integrations.

=== "Best Practices"

    ## Principles That Matter

    ### Fast Feedback Loops

    The single most important property of a CI/CD pipeline is speed. If the pipeline takes 30 minutes, developers stop waiting for results, batch up multiple changes, and lose the fast-feedback benefit that CI was designed to provide. The industry benchmark is under 10 minutes from push to deploy-ready.

    Achieving this requires deliberate effort: parallel test execution, aggressive caching, incremental builds, and ruthless pruning of slow or flaky tests. At Shopify, the engineering team invested heavily in test parallelization and selective test execution (only running tests affected by the changed files), bringing their monolith's CI time from over an hour to under 10 minutes.

    ### Test in Production-Like Environments

    A test that passes in CI but fails in production is worse than no test at all -- it creates false confidence. Staging environments should mirror production as closely as possible: same operating system, same database engine and version, same network topology, same environment variables (with different values).

    Netflix takes this further with their concept of "production-is-staging." They test directly in production using feature flags and traffic shadowing, because they found that no staging environment could faithfully replicate the complexity of their production infrastructure serving 260 million subscribers across 190 countries.

    ### Immutable Artifacts

    Never rebuild an artifact between environments. The Docker image deployed to staging must be bit-for-bit identical to what goes to production. Environment-specific configuration should be injected at runtime through environment variables or config services, not baked into the build. This guarantees that what you tested is exactly what you deploy.

    Tag every artifact with the Git commit SHA. This creates an unbreakable link between running code and source, making it trivial to answer "what version is in production?" and "what changed since the last deploy?"

    ### Infrastructure as Code Integration

    CI/CD pipelines should not only deploy application code but also validate infrastructure changes. Terraform plans, CloudFormation changesets, and Kubernetes manifest diffs should be reviewed in pull requests and applied through the same pipeline, with the same approval gates.

    At HashiCorp, infrastructure changes go through a "speculative plan" in CI that shows exactly what resources will be created, modified, or destroyed. This plan is posted as a pull request comment for review before any changes are applied, treating infrastructure with the same rigor as application code.

    ### Etsy's Deployment Story

    Etsy is one of the most cited examples of CI/CD transformation. In 2011, they deployed roughly twice a week, and each deploy was a multi-hour, anxiety-inducing event. Engineers were afraid to deploy because the blast radius of a failure was enormous.

    They invested in a custom deployment tool called Deployinator that made deploying as simple as clicking a button. They built dashboards showing deployment frequency, error rates, and performance metrics. They established a culture where every engineer deployed on their first day.

    By 2014, Etsy was deploying over 50 times per day to production. Their change failure rate dropped because each deploy was tiny -- a few lines of code rather than weeks of accumulated changes. Mean time to recovery plummeted because rolling back a small change is trivial compared to untangling a massive release. The lesson: small, frequent deploys are not just faster -- they are fundamentally safer.

---

## Key Takeaways

CI/CD is as much a cultural practice as a technical one. The pipeline is just automation -- the real shift is in how teams think about integration, testing, and deployment. Committing frequently to a shared mainline, maintaining a comprehensive automated test suite, and treating the pipeline as a first-class product are what separate high-performing teams from the rest.

The critical numbers to remember: aim for under 10 minutes from push to deploy-ready, follow the 70/20/10 testing pyramid, deploy immutable artifacts tagged with commit SHAs, and prefer many small deploys over few large ones. Google runs 150 million tests per day. Amazon deploys every 11.7 seconds. Etsy went from biweekly deploys to 50+ per day. The pattern is consistent: automation and frequency reduce risk.

| Metric               | Low Performer    | Elite Performer     |
|----------------------|------------------|---------------------|
| Deploy frequency     | Monthly          | Multiple per day    |
| Lead time for change | 1-6 months       | Less than 1 hour    |
| Change failure rate  | 46-60%           | 0-15%               |
| Mean time to recover | 1 week - 1 month | Less than 1 hour    |

*Source: DORA State of DevOps Reports*

---

## Related Topics

- [Deployment Strategies](strategies.md) -- Blue-green, canary, and rolling deployments
- [Containers](containers.md) -- Docker and containerization fundamentals
- [Infrastructure as Code](infrastructure.md) -- Terraform, CloudFormation, Pulumi
- [Monitoring](../observability/monitoring.md) -- Observability for deployed services
